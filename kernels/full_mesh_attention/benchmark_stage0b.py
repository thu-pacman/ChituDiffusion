# SPDX-License-Identifier: Apache-2.0
"""Measure CuTe attention reading pre-populated NVSHMEM symmetric K/V buffers."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

# Chitu's optional attention adapters may import the installed FlashAttention-2
# package. Put the reviewed CuTe checkout first before importing Chitu modules
# so all ranks resolve one coherent ``flash_attn`` package.
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "refs/kernels/flash-attention"))

from chitu_diffusion.parallel.fast_cp.agkv_transport import FastAgkvTransport  # noqa: E402
from chitu_diffusion.parallel.nccl.agkv import TorchAgkvTransport  # noqa: E402

from benchmark_stage0a import _load_cute_attention  # noqa: E402


def _measure_all_ranks(
    call: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    local: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        stop.record()
        stop.synchronize()
        local.append(start.elapsed_time(stop))
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    samples = [sample for rank_samples in gathered if rank_samples for sample in rank_samples]
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size not in {2, 4, 8}:
        raise ValueError("Stage 0b requires CP2, CP4, or CP8")
    if args.sequence % world_size:
        raise ValueError("global sequence must be divisible by CP degree")

    torch.manual_seed(1000 + rank)
    local_sequence = args.sequence // world_size
    query = torch.randn(
        1,
        local_sequence,
        args.heads,
        args.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    local_key = torch.randn_like(query)
    local_value = torch.randn_like(query)

    fallback = TorchAgkvTransport(dist.group.WORLD)
    transport = FastAgkvTransport(
        dist.group.WORLD,
        torch.device("cuda", local_rank),
        fallback=fallback,
    )
    try:
        symmetric_key, symmetric_value = transport.all_gather_kv(local_key, local_value)
        torch.cuda.synchronize()
        ordinary_key = symmetric_key.clone()
        ordinary_value = symmetric_value.clone()
        torch.cuda.synchronize()

        flash_attn_func = _load_cute_attention()

        def symmetric_attention() -> torch.Tensor:
            return flash_attn_func(
                query,
                symmetric_key,
                symmetric_value,
                causal=False,
            )[0]

        def ordinary_attention() -> torch.Tensor:
            return flash_attn_func(
                query,
                ordinary_key,
                ordinary_value,
                causal=False,
            )[0]

        symmetric_output = symmetric_attention()
        ordinary_output = ordinary_attention()
        torch.testing.assert_close(
            symmetric_output,
            ordinary_output,
            rtol=0.0,
            atol=0.0,
        )

        symmetric = _measure_all_ranks(
            symmetric_attention,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        ordinary = _measure_all_ranks(
            ordinary_attention,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        result = {
            "device": torch.cuda.get_device_name(),
            "world_size": world_size,
            "dtype": "bfloat16",
            "shape": {
                "query": list(query.shape),
                "key_value": list(symmetric_key.shape),
            },
            "ordinary": ordinary,
            "nvshmem_symmetric": symmetric,
            "symmetric_overhead_percent": (
                float(symmetric["median_ms"]) / float(ordinary["median_ms"]) - 1
            )
            * 100,
            "max_abs_error": float((symmetric_output - ordinary_output).abs().max()),
        }
        if rank == 0:
            encoded = json.dumps(result, indent=2)
            print(encoded)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(encoded + "\n")
    finally:
        transport.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
