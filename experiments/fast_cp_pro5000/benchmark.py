from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from chitu_diffusion.parallel.cp import create_ulysses_transport


H3_GLOBAL_HEADS = 56
H3_GLOBAL_SEQUENCE_768P_5S = 21_824
H3_HEAD_DIM = 128


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def _run_cycle(transport, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    transport.reset()
    query, key, value = transport.all_to_all_qkv(*tensors)
    return transport.all_to_all(query, 1, 2)


def _measure(
    transport,
    tensors: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[float], int]:
    for _ in range(warmup):
        _run_cycle(transport, tensors)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_cycle(transport, tensors)
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


def _benchmark_shape(
    *,
    global_sequence: int,
    local_heads: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    world_size = dist.get_world_size()
    if global_sequence % world_size:
        raise ValueError("global sequence must be divisible by CP degree")
    if local_heads % world_size:
        raise ValueError("TP-local heads must be divisible by CP degree")

    shape = (
        1,
        global_sequence // world_size,
        local_heads,
        H3_HEAD_DIM,
    )
    generator = torch.Generator(device=device).manual_seed(20260825)
    tensors = tuple(
        torch.randn(
            shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        for _ in range(3)
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
        references = _run_cycle(transports["torch"], tensors)
        candidate = _run_cycle(transports["fast_ulysses"], tensors)
        torch.testing.assert_close(candidate, references, rtol=0, atol=0)

        results: dict[str, object] = {}
        for name, transport in transports.items():
            dist.barrier()
            local_samples, peak_memory = _measure(
                transport,
                tensors,
                warmup=warmup,
                iterations=iterations,
            )
            samples = _worst_rank_samples(local_samples)
            if dist.get_rank() == 0:
                median_ms = statistics.median(samples)
                remote_bytes = (
                    4
                    * tensors[0].numel()
                    * tensors[0].element_size()
                    * (world_size - 1)
                    / world_size
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
            torch_ms = float(results["torch"]["median_ms"])
            fast_ms = float(results["fast_ulysses"]["median_ms"])
            return {
                "global_sequence": global_sequence,
                "local_tensor_shape": list(shape),
                "cycle": "QKV heads-to-sequence plus output sequence-to-heads",
                "backends": results,
                "fast_over_nccl_speedup": torch_ms / fast_ms,
                "fast_latency_change_percent": (fast_ms / torch_ms - 1.0) * 100.0,
            }
        return {}
    finally:
        for transport in transports.values():
            transport.close()


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
