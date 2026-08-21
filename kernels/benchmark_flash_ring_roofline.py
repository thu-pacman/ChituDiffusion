#!/usr/bin/env python3
"""Measure the compute/HBM side of a low-memory Flash Ring roofline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from chitu_diffusion.parallel.fast_cp._cute import load_cute_flash_attn_fwd
from chitu_diffusion.parallel.fast_cp.ring import (
    merge_flash_partials_fp32,
)


def _measure(call, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-sequence", type=int, default=4096)
    parser.add_argument("--cp-degree", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--ce-gbps", type=float, default=390.0)
    parser.add_argument("--ce-descriptor-us", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.global_sequence % args.cp_degree:
        raise ValueError("global sequence must be divisible by CP degree")
    if args.ce_gbps <= 0:
        raise ValueError("CE bandwidth must be positive")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    local_sequence = args.global_sequence // args.cp_degree
    generator = torch.Generator(device=device).manual_seed(20260815)
    query = torch.randn(
        args.batch,
        local_sequence,
        args.heads,
        args.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    key_blocks = [
        torch.randn(
            args.batch,
            local_sequence,
            args.heads,
            args.head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        for _ in range(args.cp_degree)
    ]
    value_blocks = [
        torch.randn(
            args.batch,
            local_sequence,
            args.heads,
            args.head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        for _ in range(args.cp_degree)
    ]
    flash_attn_fwd = load_cute_flash_attn_fwd()
    partial_out = torch.empty_like(query)
    partial_lse = torch.empty(
        args.batch,
        args.heads,
        local_sequence,
        dtype=torch.float32,
        device=device,
    )

    def partial(index: int) -> tuple[torch.Tensor, torch.Tensor]:
        flash_attn_fwd(
            query,
            key_blocks[index],
            value_blocks[index],
            causal=False,
            return_lse=True,
            out=partial_out,
            lse=partial_lse,
        )
        return partial_out, partial_lse

    def shards_only() -> None:
        """Every phase's attention with no merge, isolating the shard split."""
        for index in range(args.cp_degree):
            partial(index)

    def prefetched_ring() -> torch.Tensor:
        running_out = running_lse = None
        for index in range(args.cp_degree):
            block_out, block_lse = partial(index)
            running_out, running_lse = merge_flash_partials_fp32(
                running_out,
                running_lse,
                block_out,
                block_lse,
            )
        return running_out.to(dtype)

    # One attention over gathered K/V is the compute Full-Mesh performs.
    gathered_key = torch.cat(key_blocks, dim=1)
    gathered_value = torch.cat(value_blocks, dim=1)
    gathered_out = torch.empty_like(query)

    def gathered_attention() -> torch.Tensor:
        return flash_attn_fwd(
            query,
            gathered_key,
            gathered_value,
            causal=False,
            out=gathered_out,
        )[0]

    reference = gathered_attention()
    actual = prefetched_ring()
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-3)

    partial_latency = _measure(
        lambda: partial(0),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    shards_latency = _measure(
        shards_only,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    prefetched_latency = _measure(
        prefetched_ring,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    gathered_latency = _measure(
        gathered_attention,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    elements = query.numel()
    local_tensor_bytes = elements * query.element_size()
    state_output_bytes = elements * torch.float32.itemsize
    state_lse_bytes = (
        args.batch * args.heads * local_sequence * torch.float32.itemsize
    )
    state_bytes = state_output_bytes + state_lse_bytes
    kv_bytes_per_phase = 2 * local_tensor_bytes
    transfer_ms = kv_bytes_per_phase / args.ce_gbps / 1e6
    # K and V are separate production descriptors.
    transfer_ms += 2.0 * args.ce_descriptor_us / 1e3
    compute_ms_per_phase = prefetched_latency["median_ms"] / args.cp_degree
    exposed_ms_per_phase = max(0.0, transfer_ms - compute_ms_per_phase)
    hidden_fraction = (
        1.0
        if transfer_ms == 0
        else 1.0 - exposed_ms_per_phase / transfer_ms
    )
    result = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "batch": args.batch,
        "global_sequence": args.global_sequence,
        "local_sequence": local_sequence,
        "cp_degree": args.cp_degree,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "partial_attention": partial_latency,
        "shard_attentions_without_merge": shards_latency,
        "prefetched_phase_state_ring": prefetched_latency,
        "gathered_single_attention": gathered_latency,
        "shard_split_penalty_ms": (
            shards_latency["median_ms"] - gathered_latency["median_ms"]
        ),
        "merge_cost_ms": (
            prefetched_latency["median_ms"] - shards_latency["median_ms"]
        ),
        "ring_over_gathered_fraction": (
            prefetched_latency["median_ms"] / gathered_latency["median_ms"] - 1.0
        ),
        "local_tensor_bytes": local_tensor_bytes,
        "kv_bytes_per_phase": kv_bytes_per_phase,
        "running_state_bytes": state_bytes,
        "running_state_read_write_bytes_per_middle_phase": 2 * state_bytes,
        "assumed_ce_GBps": args.ce_gbps,
        "assumed_ce_descriptor_us": args.ce_descriptor_us,
        "estimated_transfer_ms_per_phase": transfer_ms,
        "measured_compute_ms_per_phase": compute_ms_per_phase,
        "estimated_exposed_ms_per_phase": exposed_ms_per_phase,
        "estimated_hidden_fraction": hidden_fraction,
        "roofline_allows_full_overlap": compute_ms_per_phase >= transfer_ms,
        "correctness": "rtol=2e-2, atol=2e-3 versus one full CuTe attention",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print("FLASH_RING_ROOFLINE=" + json.dumps(result, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
