"""Correctness and latency gate for the exact CP attention-state merge.

Run on one or two GPUs.  With two ranks each rank validates an independent
local CP shard, then reduces the worst error so a PASS covers every rank.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# Keep the standalone harness importable when launched as a bare script by
# script/srun_direct.sh rather than through an installed package entry point.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernel_adapters.attention_state_merge import (  # noqa: E402 - imports follow the repository path bootstrap
    merge_attention_states_reference,
    triton_merge_available,
    try_merge_attention_states,
)


def _timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--require-world-size", type=int, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not triton_merge_available():
        raise RuntimeError("requires CUDA and Triton")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if "RANK" in os.environ:
        dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    if args.require_world_size is not None and world != args.require_world_size:
        raise RuntimeError(f"expected world size {args.require_world_size}, got {world}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    generator = torch.Generator(device=device).manual_seed(20260721 + rank)
    shape = (args.batch, args.seq_len, args.heads, args.head_dim)
    original_out = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    # Preserve the production CP layout: [B,H,S] returned by attention is
    # transposed into a non-contiguous [B,S,H,1] running state.
    original_lse = (
        torch.randn(
            (args.batch, args.heads, args.seq_len),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        .transpose(-2, -1)
        .unsqueeze(-1)
    )
    if original_lse.is_contiguous():
        raise RuntimeError("benchmark failed to construct the production strided LSE layout")
    block_out = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    block_lse = torch.randn(
        (args.batch, args.heads, args.seq_len), device=device, dtype=torch.float32, generator=generator
    )
    reference_out, reference_lse = merge_attention_states_reference(
        original_out, original_lse, block_out, block_lse
    )
    candidate_out, candidate_lse = original_out.clone(), original_lse.clone()
    os.environ["CHITU_CP_ATTN_STATE_MERGE"] = "benchmark_only"
    if not try_merge_attention_states(
        candidate_out, candidate_lse, block_out, block_lse, benchmark=True
    ):
        raise RuntimeError("Triton path declined a supported benchmark layout")
    torch.cuda.synchronize()
    max_out = (candidate_out - reference_out).abs().max()
    max_lse = (candidate_lse - reference_lse).abs().max()
    worst = torch.stack((max_out, max_lse))
    if dist.is_initialized():
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"[correctness] world={world} shape={shape} dtype={args.dtype} "
            f"max_abs_out={worst[0].item():.3e} max_abs_lse={worst[1].item():.3e}"
        )
    if worst.max().item() > 2e-5:
        raise AssertionError("merge output exceeds strict float32 tolerance")

    def reference_once() -> None:
        merge_attention_states_reference(original_out, original_lse, block_out, block_lse)

    def triton_once() -> None:
        candidate_out.copy_(original_out)
        candidate_lse.copy_(original_lse)
        if not try_merge_attention_states(
            candidate_out, candidate_lse, block_out, block_lse, benchmark=True
        ):
            raise RuntimeError("Triton path unexpectedly declined benchmark layout")

    reference_ms = _timed(reference_once, args.warmup, args.iters)
    triton_ms = _timed(triton_once, args.warmup, args.iters)
    if rank == 0:
        print(
            f"[latency] reference={reference_ms:.4f}ms triton={triton_ms:.4f}ms "
            f"speedup={reference_ms / triton_ms:.3f}x"
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
