"""Measure Fast Ulysses all-to-all bandwidth on its production tensor layout."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

from chitu_diffusion.parallel import create_ulysses_transport


def _measure(
    call: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    dist.barrier()

    local_samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        stop.record()
        stop.synchronize()
        local_samples.append(start.elapsed_time(stop))

    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_samples)
    rank_samples = [samples for samples in gathered if samples]
    critical_path = [
        max(samples[index] for samples in rank_samples) for index in range(iterations)
    ]
    ordered = sorted(critical_path)
    return {
        "median_ms": statistics.median(critical_path),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


def _with_bandwidth(latency: dict[str, float], remote_bytes: int) -> dict[str, float]:
    median_ms = latency["median_ms"]
    remote_gbps = remote_bytes / median_ms / 1e6
    return {
        **latency,
        "remote_payload_bytes_per_rank": remote_bytes,
        "remote_payload_GBps_per_rank": remote_gbps,
        "send_plus_receive_GBps_per_rank": 2.0 * remote_gbps,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-sequence", type=int, default=75600)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--require-world-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.require_world_size:
        raise ValueError(f"expected {args.require_world_size} ranks, got {world_size}")
    if args.global_sequence % world_size:
        raise ValueError("global sequence must be divisible by world size")
    if args.heads % world_size:
        raise ValueError("head count must be divisible by world size")

    local_sequence = args.global_sequence // world_size
    generator = torch.Generator(device=device).manual_seed(20260811 + rank)
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
    local_tensor_bytes = query.numel() * query.element_size()
    remote_bytes_per_a2a = local_tensor_bytes * (world_size - 1) // world_size

    torch_transport = create_ulysses_transport(
        "torch",
        process_group=dist.group.WORLD,
        device=device,
        static_full_world=True,
    )
    expected_forward = torch_transport.all_to_all(query, 2, 1)
    torch_transport.close()

    required_pool = 4 * local_tensor_bytes + (64 << 20)
    alignment = 64 << 20
    os.environ.setdefault(
        "CHITU_FAST_ULYSSES_POOL_BYTES",
        str(((required_pool + alignment - 1) // alignment) * alignment),
    )
    transport = create_ulysses_transport(
        "fast_ulysses",
        process_group=dist.group.WORLD,
        device=device,
        static_full_world=True,
    )

    def one_forward_a2a() -> torch.Tensor:
        transport.reset()
        return transport.all_to_all(query, 2, 1)

    def four_a2a_chain() -> torch.Tensor:
        transport.reset()
        exchanged_query, _, _ = transport.all_to_all_qkv(query, key, value)
        return transport.all_to_all(exchanged_query, 1, 2)

    try:
        actual_forward = one_forward_a2a()
        torch.testing.assert_close(actual_forward, expected_forward, rtol=0.0, atol=0.0)
        actual_round_trip = four_a2a_chain()
        torch.testing.assert_close(actual_round_trip, query, rtol=0.0, atol=0.0)

        single = _with_bandwidth(
            _measure(
                one_forward_a2a,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            remote_bytes_per_a2a,
        )
        chain = _with_bandwidth(
            _measure(
                four_a2a_chain,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            4 * remote_bytes_per_a2a,
        )
    finally:
        transport.close()

    result = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "fast_ulysses_version": _package_version("fast-ulysses"),
        "world_size": world_size,
        "batch": args.batch,
        "global_sequence": args.global_sequence,
        "local_sequence": local_sequence,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "local_tensor_bytes": local_tensor_bytes,
        "remote_fraction": (world_size - 1) / world_size,
        "correctness": "bitwise versus Torch/NCCL forward and Fast round trip",
        "single_forward_a2a": single,
        "four_a2a_chain": chain,
        "chain_over_single": chain["median_ms"] / single["median_ms"],
    }
    if rank == 0:
        encoded = json.dumps(result, indent=2, sort_keys=True)
        print("FAST_ULYSSES_A2A_BANDWIDTH=" + json.dumps(result, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
