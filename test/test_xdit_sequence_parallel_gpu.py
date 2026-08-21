"""xDiT/xFuser sequence-parallel correctness and latency benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _measure(call, *, warmup: int, iterations: int) -> dict[str, float]:
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
    flattened = [value for rank_samples in gathered if rank_samples for value in rank_samples]
    ordered = sorted(flattened)
    return {
        "median_ms": statistics.median(flattened),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
    }


def _full_sequence(tensor: torch.Tensor) -> torch.Tensor:
    pieces = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(pieces, tensor)
    return torch.cat(pieces, dim=1)


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    full_key = _full_sequence(key)
    full_value = _full_sequence(value)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            full_key.transpose(1, 2),
            full_value.transpose(1, 2),
        ).transpose(1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-sequence", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ulysses-degree", type=int, required=True)
    parser.add_argument(
        "--backend",
        choices=("fa", "fa3", "torch_cudnn"),
        default="fa",
    )
    parser.add_argument("--pack-qkv", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.global_sequence % world_size:
        raise ValueError("xDiT equal sequence chunks require S_global divisible by world size")
    if world_size % args.ulysses_degree:
        raise ValueError("world size must be divisible by ulysses degree")
    ring_degree = world_size // args.ulysses_degree
    if args.heads % args.ulysses_degree:
        raise ValueError("head count must be divisible by ulysses degree")

    torch.cuda.set_device(local_rank)
    from xfuser.core.distributed import (  # noqa: PLC0415
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(
        sequence_parallel_degree=world_size,
        ulysses_degree=args.ulysses_degree,
        ring_degree=ring_degree,
        backend="nccl",
    )

    from xfuser.core.long_ctx_attention import (  # noqa: PLC0415
        xFuserLongContextAttention,
    )
    from yunchang.kernels import AttnType  # noqa: PLC0415

    backend = {
        "fa": AttnType.FA,
        "fa3": AttnType.FA3,
        "torch_cudnn": AttnType.TORCH_CUDNN,
    }[args.backend]
    if ring_degree > 1 and backend is AttnType.TORCH_CUDNN:
        raise ValueError("xDiT TORCH_CUDNN does not return LSE required by Ring merge")

    local_sequence = args.global_sequence // world_size
    torch.manual_seed(3000 + rank)
    shape = (args.batch, local_sequence, args.heads, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    attention = xFuserLongContextAttention(
        attn_type=backend,
        use_pack_qkv=args.pack_qkv,
    )

    def call() -> torch.Tensor:
        return attention(
            None,
            query=query,
            key=key,
            value=value,
            causal=False,
        )

    actual = call()
    max_abs_error = None
    if not args.skip_correctness:
        expected = _reference(query, key, value)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
        max_abs_error = float((actual - expected).abs().max())

    timing = _measure(call, warmup=args.warmup, iterations=args.iterations)
    result = {
        "device": torch.cuda.get_device_name(),
        "xfuser_version": importlib.metadata.version("xfuser"),
        "yunchang_version": importlib.metadata.version("yunchang"),
        "world_size": world_size,
        "ulysses_degree": args.ulysses_degree,
        "ring_degree": ring_degree,
        "backend": args.backend,
        "pack_qkv": args.pack_qkv,
        "batch": args.batch,
        "global_sequence": args.global_sequence,
        "local_sequence": local_sequence,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "latency": timing,
        "max_abs_error": max_abs_error,
    }
    if rank == 0:
        encoded = json.dumps(result, indent=2)
        print(encoded)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
