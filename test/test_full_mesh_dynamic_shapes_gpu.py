"""Dynamic-shape correctness cycle for chunked Full-Mesh attention."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def _gather(tensor: torch.Tensor, lengths: tuple[int, ...]) -> torch.Tensor:
    rank = dist.get_rank()
    pieces = []
    for source, length in enumerate(lengths):
        piece = (
            tensor
            if source == rank
            else torch.empty(
                (tensor.shape[0], length, *tensor.shape[2:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
        )
        dist.broadcast(piece, source)
        pieces.append(piece)
    return torch.cat(pieces, dim=1)


@torch.inference_mode()
def main() -> None:
    root = Path(__file__).parents[1]
    sys.path.insert(0, str(root / "refs/kernels/flash-attention"))
    sys.path.insert(0, str(root))
    from flash_attn.cute.interface import _flash_attn_fwd

    from chitu_diffusion.parallel.fast_cp.agkv import (
        FastAgkvAttention,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--scheduler", choices=("fixed", "ready_mask"), default="fixed")
    parser.add_argument(
        "--sequences",
        default="4096,8192,4096,75600",
        help="comma-separated global sequence cycle",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    sequences = tuple(int(item) for item in args.sequences.split(","))

    attention = FastAgkvAttention(
        dist.group.WORLD,
        device,
        chunk_count=args.chunks,
            scheduler=args.scheduler,
    )
    worst = 0.0
    try:
        for sequence in sequences:
            lengths = attention.shard_lengths(sequence, world)
            if min(lengths) < args.chunks:
                raise ValueError("every shard must contain at least one token per chunk")
            shape = (
                args.batch,
                lengths[rank],
                args.heads,
                args.head_dim,
            )
            generator = torch.Generator(device=device).manual_seed(
                20260810 + rank + sequence
            )
            query, key, value = [
                torch.randn(
                    shape,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                for _ in range(3)
            ]
            full_key = _gather(key, lengths)
            full_value = _gather(value, lengths)
            expected = _flash_attn_fwd(query, full_key, full_value)[0]
            for _ in range(args.repeats):
                actual = attention(
                    query,
                    key,
                    value,
                    global_sequence=sequence,
                )
                error = float((actual.float() - expected.float()).abs().max())
                worst = max(worst, error)
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
            torch.cuda.synchronize()
        values: list[float | None] = [None] * world
        dist.all_gather_object(values, worst)
        if rank == 0:
            print(
                f"dynamic-shape CP{world} C={args.chunks} passed; "
                f"worst_max_abs_error={max(float(item) for item in values):.6g}"
            )
    finally:
        attention.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
