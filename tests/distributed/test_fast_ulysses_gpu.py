from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from chitu_diffusion.parallel import (
    EpeParallelContext,
    ImageSelfAttention,
    create_ulysses_transport,
)

_RESULTS: list[dict[str, object]] = []

# Shapes model the tensor entering the first Ulysses all-to-all:
# (name, local sequence at CP4, attention heads, head dimension).
_MODEL_SHAPES = (
    ("flux1-1024", 1152, 24, 128),
    ("flux2-klein-1024", 1152, 24, 128),
    ("qwen-image-1024", 1152, 24, 128),
    ("wan-1.3b-480p", 8320, 12, 128),
    ("wan-14b-480p", 8320, 40, 128),
    ("z-image-1024", 1152, 30, 128),
)


@pytest.fixture(scope="session", autouse=True)
def _distributed_cuda_session():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2 or not torch.cuda.is_available():
        yield
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        yield
    finally:
        if dist.get_rank() == 0:
            output = os.environ.get("CHITU_FAST_ULYSSES_BENCH_OUTPUT")
            if output:
                path = Path(output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(_RESULTS, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        dist.barrier()
        if owned:
            dist.destroy_process_group()


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _measure_roundtrip(
    transport,
    tensor: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> tuple[torch.Tensor, list[float], int]:
    for _ in range(warmup):
        transport.reset()
        output = transport.all_to_all(tensor, 2, 1)
        output = transport.all_to_all(output, 1, 2)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    events = []
    output = tensor
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        transport.reset()
        start.record()
        output = transport.all_to_all(tensor, 2, 1)
        output = transport.all_to_all(output, 1, 2)
        stop.record()
        events.append((start, stop))
    torch.cuda.synchronize()
    samples = [start.elapsed_time(stop) for start, stop in events]
    return output, samples, torch.cuda.max_memory_allocated()


@pytest.mark.gpu
@pytest.mark.distributed
def test_torch_ulysses_matches_full_attention() -> None:
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != 4
        or not torch.cuda.is_available()
        or not dist.is_initialized()
    ):
        pytest.skip("run with torchrun on exactly four GPUs")
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device=device).manual_seed(20260805)
    query = torch.randn(
        1, 128, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    key = torch.randn(
        1, 128, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    value = torch.randn(
        1, 128, 8, 64, device=device, dtype=torch.bfloat16, generator=generator
    )
    start = rank * 32
    stop = start + 32
    parallel = EpeParallelContext.from_torchrun(
        allowed_widths=(1, 4),
        init_process_group=False,
        owns_process_group=False,
        ulysses_degree=4,
        ulysses_transport="torch",
    )
    try:
        output = ImageSelfAttention("ulysses")(
            query[:, start:stop],
            key[:, start:stop],
            value[:, start:stop],
            lane_process_group=parallel.active.process_group,
            ulysses_topology=parallel.active_ulysses,
        )
        reference = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)
        torch.testing.assert_close(
            output,
            reference[:, start:stop],
            rtol=2e-2,
            atol=2e-2,
        )
    finally:
        dist.barrier()
        parallel.close()


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("model_name", "cp4_sequence", "heads", "head_dim"),
    _MODEL_SHAPES,
)
def test_fast_ulysses_model_shape_transport_benchmark(
    model_name: str,
    cp4_sequence: int,
    heads: int,
    head_dim: int,
) -> None:
    if (
        int(os.environ.get("WORLD_SIZE", "1")) < 2
        or not torch.cuda.is_available()
        or not dist.is_initialized()
    ):
        pytest.skip("run with torchrun on a multi-GPU node")

    world_size = dist.get_world_size()
    if heads % world_size:
        pytest.skip(f"{model_name} has {heads} heads, incompatible with CP{world_size}")
    local_sequence = max(128, cp4_sequence * 4 // world_size)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device=device).manual_seed(20260805)
    tensor = torch.randn(
        1,
        local_sequence,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    transports = {
        name: create_ulysses_transport(
            name,
            process_group=dist.group.WORLD,
            device=device,
            static_full_world=True,
        )
        for name in ("torch", "fast_ulysses")
    }
    try:
        forward_outputs = {}
        for name, transport in transports.items():
            transport.reset()
            forward_outputs[name] = transport.all_to_all(tensor, 2, 1)
        torch.testing.assert_close(
            forward_outputs["fast_ulysses"],
            forward_outputs["torch"],
            rtol=0,
            atol=0,
        )

        outputs = {}
        rank_samples = {}
        peak_memory = {}
        for name, transport in transports.items():
            dist.barrier()
            output, samples, peak = _measure_roundtrip(
                transport,
                tensor,
                warmup=5,
                iterations=20,
            )
            outputs[name] = output
            rank_samples[name] = samples
            peak_memory[name] = peak
        torch.testing.assert_close(
            outputs["fast_ulysses"],
            outputs["torch"],
            rtol=0,
            atol=0,
        )

        gathered: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(
            gathered,
            {
                "samples": rank_samples,
                "peak_memory": peak_memory,
            },
        )
        if dist.get_rank() == 0:
            summary: dict[str, object] = {
                "model": model_name,
                "world_size": world_size,
                "shape": [1, local_sequence, heads, head_dim],
                "dtype": "bfloat16",
                "transports": {},
            }
            transport_summary = summary["transports"]
            assert isinstance(transport_summary, dict)
            for name in transports:
                all_samples = [
                    sample
                    for rank_result in gathered
                    if rank_result is not None
                    for sample in rank_result["samples"][name]
                ]
                all_peaks = [
                    rank_result["peak_memory"][name]
                    for rank_result in gathered
                    if rank_result is not None
                ]
                transport_summary[name] = {
                    "median_ms": statistics.median(all_samples),
                    "p95_ms": _percentile(all_samples, 0.95),
                    "max_peak_memory_bytes": max(all_peaks),
                    "pool_bytes": transports[name].pool_bytes,
                }
            fast_median = transport_summary["fast_ulysses"]["median_ms"]
            torch_median = transport_summary["torch"]["median_ms"]
            summary["fast_speedup_vs_torch"] = torch_median / fast_median
            _RESULTS.append(summary)
            print("FAST_ULYSSES_RESULT=" + json.dumps(summary, sort_keys=True))
    finally:
        for transport in reversed(tuple(transports.values())):
            transport.close()
