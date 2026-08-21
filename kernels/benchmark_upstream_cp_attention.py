"""Compare extracted SGLang/vLLM-Omni CP attention with Chitu baselines."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

# Full Mesh carries a reviewed streaming-KV overlay on top of the tracked
# FlashAttention checkout.  Load that checkout before the packaged FlashAttention
# version so CuTe DSL Python sources and the overlay stay on the same revision.
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "refs/kernels/flash-attention"))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from chitu_diffusion.parallel import (
    FastAgkvAttention,
    create_agkv_transport,
    create_ulysses_transport,
)
from kernel_adapters.upstream_cp_attention import (
    SGLANG_COMMIT,
    VLLM_OMNI_COMMIT,
    SglangIpcA2AState,
    SglangPackedUlyssesAttention,
    omni_allgather_kv_attention,
    omni_ring_attention,
    omni_ulysses_attention,
    sglang_ipc_ulysses_attention,
)

ALL_METHODS = (
    "sglang_packed_ulysses",
    "sglang_ipc_ulysses",
    "omni_ring",
    "omni_ulysses",
    "omni_agkv",
    "torch_ulysses",
    "torch_agkv",
    "fast_ulysses",
    "fast_agkv",
    "full_mesh",
)
DEFAULT_METHODS = ALL_METHODS[:7]
EXCLUSIVE_FAST_METHODS = {"fast_ulysses", "fast_agkv", "full_mesh"}


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def _cudnn_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return (
            F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
        )


def _full_sequence(tensor: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    batch, local_sequence, heads, head_dim = tensor.shape
    gathered = torch.empty(
        world_size * batch,
        local_sequence,
        heads,
        head_dim,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(gathered, tensor.contiguous())
    return (
        gathered.view(world_size, batch, local_sequence, heads, head_dim)
        .permute(1, 0, 2, 3, 4)
        .reshape(batch, world_size * local_sequence, heads, head_dim)
        .contiguous()
    )


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return _cudnn_attention(query, _full_sequence(key), _full_sequence(value))


def _ulysses_attention(query, key, value, transport):
    transport.reset()
    query, key, value = transport.all_to_all_qkv(query, key, value)
    output = _cudnn_attention(query, key, value)
    return transport.all_to_all(output, 1, 2).contiguous()


def _agkv_attention(query, key, value, transport):
    full_key, full_value = transport.all_gather_kv(key, value)
    return _cudnn_attention(query, full_key, full_value)


def _measure(
    call: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, samples)
    rank_samples = [values for values in gathered if values]
    critical_path = [
        max(values[index] for values in rank_samples) for index in range(iterations)
    ]
    ordered = sorted(critical_path)
    return {
        "median_ms": statistics.median(critical_path),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


def _capture_cuda_trace(
    call: Callable[[], torch.Tensor],
    *,
    label: str,
    warmup: int,
    iterations: int,
) -> None:
    """Capture only steady-state calls when launched under nsys."""

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        # Nsight capture startup inflates the first GPU activity by ~170 us on
        # H20. Pay that tax in an explicitly marked lead-in, then synchronize so
        # iteration_0 is a clean, isolated steady-state timeline.
        torch.cuda.nvtx.range_push(f"{label}/capture_lead_in")
        try:
            call()
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        # Only the selected rank pays profiler instrumentation overhead. Realign
        # all ranks so that its measured collective does not inherit lead-in skew.
        dist.barrier()
        for index in range(iterations):
            torch.cuda.nvtx.range_push(f"{label}/iteration_{index}")
            try:
                call()
            finally:
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    dist.barrier()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _collect_setup_status(error: str | None) -> tuple[bool, str | None]:
    statuses: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(statuses, error)
    failures = [f"rank {rank}: {value}" for rank, value in enumerate(statuses) if value]
    return not failures, "; ".join(failures) if failures else None


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences",
        type=_parse_int_csv,
        default=(4096, 8192, 16384, 32768, 75600),
    )
    parser.add_argument(
        "--methods",
        type=_parse_csv,
        default=DEFAULT_METHODS,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=0,
        help="capture this many steady-state calls via cudaProfilerApi",
    )
    parser.add_argument("--full-mesh-chunks", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument(
        "--full-mesh-buffers",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="rotated symmetric K/V buffers; 0 lets the implementation choose",
    )
    parser.add_argument(
        "--full-mesh-scheduler",
        choices=("fixed", "ready_mask"),
        default="fixed",
    )
    parser.add_argument(
        "--full-mesh-cp2-kernel",
        choices=("fused", "split_merge"),
        default="fused",
    )
    parser.add_argument("--full-mesh-cp2-tile-n", type=int, choices=(64, 96, 128))
    parser.add_argument("--full-mesh-cp2-threads", type=int, choices=(256, 384), default=384)
    parser.add_argument("--full-mesh-cp2-no-intra-overlap", action="store_true")
    parser.add_argument("--full-mesh-cp2-pv-smem", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--measure-single-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    unknown = sorted(set(args.methods) - set(ALL_METHODS))
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")
    selected_fast = sorted(set(args.methods) & EXCLUSIVE_FAST_METHODS)
    if len(selected_fast) > 1:
        parser.error(
            "Fast Ulysses, Fast AGKV, and Full-Mesh each own the process-wide "
            "UlyssesGroup; run them in separate benchmark invocations, got: "
            + ", ".join(selected_fast)
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    if world_size not in {2, 4, 8}:
        raise ValueError("benchmark requires CP2, CP4, or CP8")
    if args.heads % world_size:
        raise ValueError("head count must be divisible by CP degree")
    for sequence in args.sequences:
        if sequence % world_size:
            raise ValueError(f"sequence {sequence} is not divisible by CP{world_size}")

    if set(args.methods) & {"fast_ulysses", "fast_agkv"}:
        local_tensor_bytes = sum(
            args.batch
            * (sequence // world_size)
            * args.heads
            * args.head_dim
            * torch.tensor([], dtype=torch.bfloat16).element_size()
            for sequence in args.sequences
        )
        if (
            "fast_agkv" in args.methods
            and "CHITU_FAST_AGKV_POOL_BYTES" not in os.environ
        ):
            required_pool_bytes = 2 * local_tensor_bytes * world_size + (16 << 20)
            pool_variable = "CHITU_FAST_AGKV_POOL_BYTES"
        elif (
            "fast_ulysses" in args.methods
            and "CHITU_FAST_ULYSSES_POOL_BYTES" not in os.environ
        ):
            required_pool_bytes = 4 * local_tensor_bytes + (16 << 20)
            pool_variable = "CHITU_FAST_ULYSSES_POOL_BYTES"
        else:
            required_pool_bytes = None
            pool_variable = None
        if required_pool_bytes is not None and pool_variable is not None:
            alignment = 64 << 20
            os.environ[pool_variable] = str(
                ((required_pool_bytes + alignment - 1) // alignment) * alignment
            )

    implementations: dict[str, Callable] = {}
    unavailable: dict[str, str] = {}
    closers = []
    ipc_state: SglangIpcA2AState | None = None

    def register(
        name: str, setup: Callable[[], tuple[Callable, object | None]]
    ) -> None:
        if name not in args.methods:
            return
        implementation = closer = None
        error = None
        try:
            implementation, closer = setup()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        available, reason = _collect_setup_status(error)
        if not available:
            unavailable[name] = reason or "unavailable"
            return
        implementations[name] = implementation
        if closer is not None:
            closers.append(closer)

    register(
        "sglang_packed_ulysses",
        lambda: (
            SglangPackedUlyssesAttention(dist.group.WORLD),
            None,
        ),
    )

    def setup_ipc():
        nonlocal ipc_state
        if world_size != 2:
            raise RuntimeError("SGLang CUDA-IPC path supports CP2 only")
        ipc_state = SglangIpcA2AState(dist.group.WORLD)
        return (
            lambda query, key, value: sglang_ipc_ulysses_attention(
                query,
                key,
                value,
                ipc_state,
                _cudnn_attention,
            ),
            None,
        )

    register("sglang_ipc_ulysses", setup_ipc)
    register(
        "omni_ring",
        lambda: (
            lambda query, key, value: omni_ring_attention(
                query, key, value, dist.group.WORLD
            ),
            None,
        ),
    )
    register(
        "omni_ulysses",
        lambda: (
            lambda query, key, value: omni_ulysses_attention(
                query,
                key,
                value,
                dist.group.WORLD,
                _cudnn_attention,
            ),
            None,
        ),
    )
    register(
        "omni_agkv",
        lambda: (
            lambda query, key, value: omni_allgather_kv_attention(
                query,
                key,
                value,
                dist.group.WORLD,
                _cudnn_attention,
            ),
            None,
        ),
    )

    def setup_ulysses(name: str):
        transport = create_ulysses_transport(
            name,
            process_group=dist.group.WORLD,
            device=device,
            static_full_world=True,
        )
        return (
            lambda query, key, value: _ulysses_attention(query, key, value, transport),
            transport,
        )

    register("torch_ulysses", lambda: setup_ulysses("torch"))
    register("fast_ulysses", lambda: setup_ulysses("fast_ulysses"))

    def setup_agkv(name: str):
        transport = create_agkv_transport(
            name,
            process_group=dist.group.WORLD,
            device=device,
            static_full_world=True,
        )
        return (
            lambda query, key, value: _agkv_attention(query, key, value, transport),
            transport,
        )

    register("torch_agkv", lambda: setup_agkv("torch"))
    register("fast_agkv", lambda: setup_agkv("fast_agkv"))

    def setup_full_mesh():
        element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
        # Buffers are keyed by shape and stay resident once a shape is seen, so
        # a sweep needs room for every sequence at once, not just the longest.
        global_bytes = sum(
            args.batch * sequence * args.heads * args.head_dim * element_size
            for sequence in args.sequences
        )
        implementation = FastAgkvAttention(
            dist.group.WORLD,
            device,
            # K and V for every rotated buffer, plus slack for flags.
            pool_bytes=2 * (args.full_mesh_buffers or 1) * global_bytes + (64 << 20),
            chunk_count=args.full_mesh_chunks,
            scheduler=args.full_mesh_scheduler,
            buffer_depth=args.full_mesh_buffers or None,
            cp2_kernel=args.full_mesh_cp2_kernel,
            cp2_tile_n=args.full_mesh_cp2_tile_n,
            cp2_num_threads=args.full_mesh_cp2_threads,
            cp2_intra_wg_overlap=not args.full_mesh_cp2_no_intra_overlap,
            cp2_mma_pv_is_rs=not args.full_mesh_cp2_pv_smem,
        )
        return (
            lambda query, key, value: implementation(
                query,
                key,
                value,
                global_sequence=key.shape[1] * world_size,
            ),
            implementation,
        )

    register("full_mesh", setup_full_mesh)

    # Wrap the callable object after setup so every method has one common signature.
    packed = implementations.get("sglang_packed_ulysses")
    if isinstance(packed, SglangPackedUlyssesAttention):
        implementations["sglang_packed_ulysses"] = (
            lambda query, key, value, packed=packed: packed(
                query, key, value, _cudnn_attention
            )
        )

    reports = []
    try:
        for sequence in args.sequences:
            local_sequence = sequence // world_size
            generator = torch.Generator(device=device).manual_seed(
                20260809 + rank + sequence
            )
            query, key, value = [
                torch.randn(
                    args.batch,
                    local_sequence,
                    args.heads,
                    args.head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                for _ in range(3)
            ]
            full_query = full_key = full_value = None
            if args.measure_single_gpu:
                full_query = _full_sequence(query)
                full_key = _full_sequence(key)
                full_value = _full_sequence(value)
                if args.skip_correctness:
                    expected = None
                else:
                    full_output = _cudnn_attention(full_query, full_key, full_value)
                    expected = full_output[
                        :, rank * local_sequence : (rank + 1) * local_sequence
                    ].contiguous()
                    del full_output
            else:
                expected = (
                    None if args.skip_correctness else _reference(query, key, value)
                )
            sequence_report = {
                "global_sequence": sequence,
                "local_sequence": local_sequence,
                "methods": {},
            }
            for name in args.methods:
                call_impl = implementations.get(name)
                if call_impl is None:
                    continue
                dist.barrier()
                actual = call_impl(query, key, value)
                if name == "sglang_ipc_ulysses" and ipc_state is not None:
                    ipc_state.check_timeout()
                method_report = {}
                if expected is not None:
                    difference = (actual - expected).abs()
                    method_report["max_abs_error"] = float(difference.max())
                    method_report["mean_abs_error"] = float(difference.mean())
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=2e-2,
                        atol=2e-2,
                    )
                if args.profile_iterations:
                    _capture_cuda_trace(
                        lambda call_impl=call_impl, query=query, key=key, value=value: (
                            call_impl(query, key, value)
                        ),
                        label=f"{name}/S{sequence}/CP{world_size}",
                        warmup=args.warmup,
                        iterations=args.profile_iterations,
                    )
                dist.barrier()
                method_report["latency"] = _measure(
                    lambda call_impl=call_impl, query=query, key=key, value=value: (
                        call_impl(query, key, value)
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                if expected is not None:
                    # Stateful implementations may switch to a warmed direct
                    # launcher after the first correctness call. Validate the
                    # steady-state path that the latency numbers actually use.
                    steady_actual = call_impl(query, key, value)
                    steady_difference = (steady_actual - expected).abs()
                    method_report["steady_state_max_abs_error"] = float(
                        steady_difference.max()
                    )
                    torch.testing.assert_close(
                        steady_actual,
                        expected,
                        rtol=2e-2,
                        atol=2e-2,
                    )
                if name == "sglang_ipc_ulysses" and ipc_state is not None:
                    ipc_state.check_timeout()
                sequence_report["methods"][name] = method_report
            if args.measure_single_gpu:
                assert full_query is not None
                assert full_key is not None
                assert full_value is not None
                dist.barrier()
                sequence_report["single_gpu_latency"] = _measure(
                    lambda: _cudnn_attention(full_query, full_key, full_value),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                for method_report in sequence_report["methods"].values():
                    median_ms = method_report["latency"]["median_ms"]
                    method_report["speedup"] = (
                        sequence_report["single_gpu_latency"]["median_ms"] / median_ms
                    )
                    method_report["parallel_efficiency"] = (
                        method_report["speedup"] / world_size
                    )
            reports.append(sequence_report)
            del query, key, value, expected
            torch.cuda.empty_cache()

        result = {
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "fast_ulysses_version": _package_version("fast-ulysses"),
            "world_size": world_size,
            "batch": args.batch,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": "bfloat16",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rank_aggregation": "per_iteration_max",
            "full_mesh_chunks": args.full_mesh_chunks,
            "full_mesh_scheduler": args.full_mesh_scheduler,
            "full_mesh_cp2_kernel": args.full_mesh_cp2_kernel,
            "full_mesh_cp2_tile_n": args.full_mesh_cp2_tile_n,
            "full_mesh_cp2_threads": args.full_mesh_cp2_threads,
            "full_mesh_buffers": args.full_mesh_buffers or "auto",
            "fast_ulysses_pool_bytes": os.environ.get("CHITU_FAST_ULYSSES_POOL_BYTES"),
            "fast_agkv_pool_bytes": os.environ.get("CHITU_FAST_AGKV_POOL_BYTES"),
            "attention_backends": {
                "sglang_packed_ulysses": "torch_cudnn",
                "sglang_ipc_ulysses": "torch_cudnn",
                "omni_ring": "aten_efficient",
                "omni_ulysses": "torch_cudnn",
                "omni_agkv": "torch_cudnn",
                "torch_ulysses": "torch_cudnn",
                "torch_agkv": "torch_cudnn",
                "fast_ulysses": "torch_cudnn",
                "fast_agkv": "torch_cudnn",
                "full_mesh": "flash_attn_cute_sm90",
            },
            "upstream_commits": {
                "sglang": SGLANG_COMMIT,
                "vllm_omni": VLLM_OMNI_COMMIT,
            },
            "unavailable": unavailable,
            "results": reports,
        }
        if rank == 0:
            encoded = json.dumps(result, indent=2, sort_keys=True)
            print("UPSTREAM_CP_ATTENTION_RESULT=" + json.dumps(result, sort_keys=True))
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(encoded + "\n", encoding="utf-8")
    finally:
        for closer in reversed(closers):
            closer.close()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
