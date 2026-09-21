"""Weight-free Wan attention benchmark through Chitu's native CP processor."""

from __future__ import annotations

import argparse
import json
import os
import socket
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_wan import WanAttention

from chitu_diffusion.models.wan.attention import WanCpAttnProcessor, _apply_rotary
from chitu_diffusion.parallel.cp import (
    AsyncAgkvTransport,
    AsyncTaggedUlyssesTransport,
    UlyssesTopology,
)
from chitu_diffusion.parallel.cp.agkv_transport import create_agkv_transport
from chitu_diffusion.parallel.cp.fast.adapters.wan import PreparedWanAgkv
from chitu_diffusion.parallel.cp.fast.runtime.backend import FastUlyssesAllToAll
from chitu_diffusion.parallel.cp.ulysses_transport import create_ulysses_transport
from tools.benchmarks.fast_cp_frameworks.common import git_revision, measure

MODELS = {
    "wan1.3b": {"heads": 12, "head_dim": 128, "hidden": 1536},
    "wan14b": {"heads": 40, "head_dim": 128, "hidden": 5120},
}
METHODS = ("nccl_agkv", "fast_agkv", "nccl_ulysses", "fast_ulysses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--serialized-async", action="store_true")
    parser.add_argument("--no-rotary", dest="rotary", action="store_false")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(rotary=True)
    return parser.parse_args()


class SerializedAsyncWanCpAttnProcessor(WanCpAttnProcessor):
    """Use native async transport kernels without projection overlap."""

    def _fast_async_attention(self, attn, hidden_states, rotary_emb):
        topology = self.parallel.active_ulysses
        transport = topology.transport
        if not isinstance(transport, AsyncTaggedUlyssesTransport):
            raise TypeError("async tagged Ulysses transport is required")
        transport.reset()

        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary(query, *rotary_emb)
            key = _apply_rotary(key, *rotary_emb)
        query_handle = transport.begin_all_to_all(
            query, 2, 1, tag="chitu_cp_0", barrier=False
        )
        key_handle = transport.begin_all_to_all(
            key, 2, 1, tag="chitu_cp_1", barrier=True
        )
        key = key_handle.wait()
        query = query_handle.wait()

        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        value = transport.all_to_all_tagged(value, 2, 1, tag="chitu_cp_2")
        return self.attention.finish_ulysses(
            query,
            key,
            value,
            topology=topology,
            output_tag="chitu_cp_3",
        )

    def _fast_agkv_attention(self, attn, hidden_states, rotary_emb):
        transport = self.parallel.active_agkv_transport
        if not isinstance(transport, AsyncAgkvTransport):
            raise TypeError("asynchronous AGKV transport is required")

        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            key = _apply_rotary(key, *rotary_emb)
        gather = transport.begin_all_gather_kv(key, value)
        full_key, full_value = gather.wait()

        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary(query, *rotary_emb)
        output = self.attention.finish_agkv(query, full_key, full_value)
        return output, query.dtype


def rank_max(value: float, device: torch.device) -> float:
    result = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.serial and args.serialized_async:
        raise ValueError("--serial and --serialized-async are mutually exclusive")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if args.sequence <= 0 or args.sequence % world:
        raise ValueError("global sequence must be positive and divisible by CP degree")

    config = MODELS[args.model]
    mode = "agkv" if args.method.endswith("agkv") else "ulysses"
    transport_name = "fast_" + mode if args.method.startswith("fast") else "torch"
    if args.serial:
        if mode == "agkv":
            os.environ["CHITU_FAST_AGKV_ASYNC"] = "off"
        else:
            os.environ["CHITU_FAST_ULYSSES_ASYNC_CE"] = "0"
    if args.method == "fast_ulysses":
        # The optimized Ulysses arm is the serial NVSHMEM path. Its compute
        # kernel is selected independently by the attention planner.
        os.environ["CHITU_FAST_ULYSSES_ASYNC_CE"] = "0"

    transport = None
    topology = None
    if mode == "agkv" and args.method != "fast_agkv":
        transport = create_agkv_transport(
            transport_name,
            process_group=dist.group.WORLD,
            device=device,
            fast_eligible=True,
        )
    elif mode == "ulysses":
        transport = create_ulysses_transport(
            transport_name,
            process_group=dist.group.WORLD,
            device=device,
            fast_eligible=True,
        )
        topology = UlyssesTopology(
            lane_ranks=tuple(range(world)),
            rank=rank,
            degree=world,
            process_group=dist.group.WORLD,
            transport=transport,
        )

    parallel = SimpleNamespace(
        active=SimpleNamespace(width=world, process_group=dist.group.WORLD),
        active_ulysses=topology,
        active_agkv_transport=transport if mode == "agkv" else None,
    )
    torch.manual_seed(args.seed)
    processor_class = (
        SerializedAsyncWanCpAttnProcessor
        if args.serialized_async
        else WanCpAttnProcessor
    )
    processor = processor_class(
        parallel,
        mode=mode,
        planned_attention=args.method == "fast_ulysses",
    )
    attention = WanAttention(
        dim=config["hidden"],
        heads=config["heads"],
        dim_head=config["head_dim"],
        processor=processor,
    ).to(device=device, dtype=torch.bfloat16)
    attention.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed + 1 + rank)
    local_sequence = args.sequence // world
    hidden = torch.randn(
        1,
        local_sequence,
        config["hidden"],
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    full_rotary = None
    local_rotary = None
    if args.rotary:
        positions = torch.arange(args.sequence, dtype=torch.float32, device=device)
        inverse_frequency = 1.0 / (
            10000
            ** (
                torch.arange(0, config["head_dim"], 2, device=device).float()
                / config["head_dim"]
            )
        )
        angles = torch.outer(positions, inverse_frequency).repeat_interleave(2, dim=-1)
        full_rotary = tuple(
            tensor.to(torch.bfloat16).view(1, args.sequence, 1, config["head_dim"])
            for tensor in (angles.cos(), angles.sin())
        )
        token_start = rank * local_sequence
        local_rotary = tuple(
            tensor[:, token_start : token_start + local_sequence]
            for tensor in full_rotary
        )

    prepared_slot = None
    prepared_runtime = None
    if args.method == "fast_agkv":
        width = config["hidden"]
        pool_bytes = 14 * 2 * args.sequence * width * 2 + (64 << 20)
        prepared_runtime = FastUlyssesAllToAll(
            dist.group.WORLD,
            device,
            pool_bytes=pool_bytes,
        )
        prepared_slot = PreparedWanAgkv(
            prepared_runtime,
            attention,
            hidden,
            local_rotary,
            projection_backend="cublas",
            tag=f"wan_{args.model}_s{args.sequence}_cp{world}",
        )
        attention.set_processor(prepared_slot)

    def call() -> torch.Tensor:
        if prepared_slot is None:
            return attention(hidden, rotary_emb=local_rotary)
        return prepared_slot(
            attention,
            hidden,
            rotary_emb=local_rotary,
            overlap=not args.serial,
            overlap_q=not (args.serial or args.serialized_async),
        )

    # Use the same module parameters and dense native processor as the reference.
    hidden_parts = [torch.empty_like(hidden) for _ in range(world)]
    dist.all_gather(hidden_parts, hidden.contiguous())
    full_hidden = torch.cat(hidden_parts, dim=1)
    active_processor = attention.processor
    attention.set_processor(processor.native)
    expected_full = attention(full_hidden, rotary_emb=full_rotary)
    expected = expected_full[:, rank * local_sequence : (rank + 1) * local_sequence]
    attention.set_processor(active_processor)
    actual = call()
    delta = actual.float() - expected.float()
    accuracy = {
        "rtol": args.rtol,
        "atol": args.atol,
        "max_abs": rank_max(float(delta.abs().max()), device),
        "relative_l2": rank_max(
            float(delta.norm() / expected.float().norm().clamp_min(1e-12)), device
        ),
    }
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)

    timing = measure(
        call,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        device=device,
        profile=args.profile,
    )
    hostnames: list[str | None] = [None] * world
    dist.all_gather_object(hostnames, socket.gethostname())
    async_capable = prepared_slot is not None
    if args.method.startswith("fast") and prepared_slot is None:
        async_capable = bool(
            transport.async_enabled
            if mode == "agkv"
            else transport.async_ce_enabled
        )
    schedule = (
        "sync"
        if args.serial
        else "serialized_ce"
        if args.serialized_async
        else "native"
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "complete",
        "framework": "chitu",
        "model": args.model,
        "method": args.method,
        "boundary": "native Wan attention input through output projection",
        "semantics": "BF16 dense non-causal Wan self-attention",
        "shape": {
            "batch": 1,
            "global_sequence": args.sequence,
            "local_sequence": local_sequence,
            "heads": config["heads"],
            "head_dim": config["head_dim"],
            "hidden": config["hidden"],
            "dtype": "bfloat16",
            "cp_degree": world,
        },
        "implementation": {
            "parallelism": mode,
            "transport": "NVSHMEM Fast CP" if args.method.startswith("fast") else "NCCL",
            "attention_backend": (
                "Chitu KV-split attention planner"
                if args.method.startswith("fast")
                else "Diffusers/PyTorch native SDPA dispatch"
            ),
            "async_transport": async_capable and not args.serial,
            "async_overlap": async_capable and schedule == "native",
            "forced_serial": schedule != "native",
            "schedule": schedule,
            "rotary": args.rotary,
            "optimized_components": (
                [
                    "NVSHMEM direct-slot KV",
                    "CE all-gather/Q projection overlap",
                    "KV-split attention planner",
                    "CUDA graph compute segments",
                ]
                if prepared_slot is not None
                else [
                    "serial NVSHMEM",
                    "SM/TMA/CE transfer selection",
                    "KV-split attention planner",
                ]
                if args.method == "fast_ulysses"
                else []
            ),
            "attention_plan": (
                prepared_slot.metadata["attention_plan"]
                if prepared_slot is not None
                else asdict(processor.attention.attention_plan)
                if processor.attention.attention_plan is not None
                else None
            ),
            "transfer_paths": (
                transport.selected_transfers
                if args.method == "fast_ulysses"
                else {"all_gather": "ce"}
                if prepared_slot is not None
                else {}
            ),
            "transfer_tuning_ms": (
                transport.transfer_tuning_ms
                if args.method == "fast_ulysses"
                else {}
            ),
        },
        "accuracy": accuracy,
        "timing": timing,
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "framework_revision": git_revision(args.source_root),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "world_size": world,
            "nodes": sorted({hostname for hostname in hostnames if hostname}),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "arguments": vars(args) | {"output": str(args.output)},
    }
    report["arguments"]["source_root"] = (
        str(args.source_root) if args.source_root is not None else None
    )
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True), flush=True)

    del expected_full, full_hidden
    if prepared_slot is not None:
        prepared_slot.close()
    if transport is not None:
        transport.close()
    if prepared_runtime is not None:
        prepared_runtime.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
