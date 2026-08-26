from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

from chitu_diffusion.parallel.cp.nccl.agkv import all_gather_sequence_pair


H3_GLOBAL_HEADS = 56
H3_GLOBAL_SEQUENCE_768P_5S = 21_824
H3_HEAD_DIM = 128


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def _measure(
    call: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[float], int]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        stop.record()
        events.append((start, stop))
    torch.cuda.synchronize()
    return (
        [start.elapsed_time(stop) for start, stop in events],
        torch.cuda.max_memory_allocated(),
    )


def _worst_rank_samples(local_samples: list[float]) -> list[float]:
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_samples)
    return [
        max(float(rank_samples[index]) for rank_samples in gathered if rank_samples)
        for index in range(len(local_samples))
    ]


def _fast_gather(
    group,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    use_ce: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    suffix = "ce" if use_ce else "kernel"
    return torch.ops.fast_ulysses.all_gather_kv_4d(
        group._group,
        key,
        value,
        f"agkv_key_{suffix}",
        f"agkv_value_{suffix}",
        use_ce,
    )


def _benchmark_shape(
    *,
    global_sequence: int,
    local_heads: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    from fast_ulysses import UlyssesGroup

    world_size = dist.get_world_size()
    if global_sequence % world_size:
        raise ValueError("global sequence must be divisible by CP degree")
    shape = (1, global_sequence // world_size, local_heads, H3_HEAD_DIM)
    generator = torch.Generator(device=device).manual_seed(20260825 + dist.get_rank())
    key, value = (
        torch.randn(
            shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        for _ in range(2)
    )
    output_bytes = key.numel() * key.element_size() * world_size
    # Kernel and Copy Engine paths use distinct K/V tags: four gathered buffers.
    required_pool = 4 * output_bytes + (16 << 20)
    alignment = 64 << 20
    pool_bytes = ((required_pool + alignment - 1) // alignment) * alignment
    group = UlyssesGroup(
        process_group=dist.group.WORLD,
        device=device,
        initial_pool_bytes=pool_bytes,
    )
    calls = {
        "nccl": lambda: all_gather_sequence_pair(key, value, dist.group.WORLD),
        "fast_kernel": lambda: _fast_gather(group, key, value, use_ce=False),
        "fast_ce": lambda: _fast_gather(group, key, value, use_ce=True),
    }
    try:
        reference = calls["nccl"]()
        for name in ("fast_kernel", "fast_ce"):
            candidate = calls[name]()
            torch.testing.assert_close(candidate[0], reference[0], rtol=0, atol=0)
            torch.testing.assert_close(candidate[1], reference[1], rtol=0, atol=0)

        results: dict[str, object] = {}
        for name, call in calls.items():
            dist.barrier()
            local_samples, peak_memory = _measure(
                call,
                warmup=warmup,
                iterations=iterations,
            )
            samples = _worst_rank_samples(local_samples)
            if dist.get_rank() == 0:
                median_ms = statistics.median(samples)
                remote_bytes = (
                    2
                    * key.numel()
                    * key.element_size()
                    * (world_size - 1)
                )
                results[name] = {
                    "median_ms": median_ms,
                    "p95_ms": _percentile(samples, 0.95),
                    "min_ms": min(samples),
                    "effective_remote_GBps": remote_bytes / median_ms / 1_000_000,
                    "peak_memory_bytes": peak_memory,
                    "samples_ms": samples,
                }
        if dist.get_rank() == 0:
            nccl_ms = float(results["nccl"]["median_ms"])
            return {
                "global_sequence": global_sequence,
                "local_kv_shape": list(shape),
                "operation": "K/V sequence all-gather",
                "backends": results,
                "fast_kernel_over_nccl_speedup": nccl_ms
                / float(results["fast_kernel"]["median_ms"]),
                "fast_ce_over_nccl_speedup": nccl_ms
                / float(results["fast_ce"]["median_ms"]),
            }
        return {}
    finally:
        group.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--sequences",
        type=int,
        nargs="+",
        default=(4096, 8192, 16384, H3_GLOBAL_SEQUENCE_768P_5S),
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    world_size = dist.get_world_size()
    if 8 % world_size:
        raise ValueError("CP degree must divide the eight-GPU H3 topology")
    tp_degree = 8 // world_size
    local_heads = H3_GLOBAL_HEADS // tp_degree

    rows = [
        _benchmark_shape(
            global_sequence=sequence,
            local_heads=local_heads,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        for sequence in args.sequences
    ]
    if dist.get_rank() == 0:
        payload = {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "world_size": world_size,
            "tp_degree": tp_degree,
            "cp_degree": world_size,
            "dtype": "bfloat16",
            "head_dim": H3_HEAD_DIM,
            "local_heads": local_heads,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
