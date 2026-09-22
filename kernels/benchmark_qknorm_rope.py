#!/usr/bin/env python3
"""Correctness and latency benchmark for the Flux2 fused QK RMSNorm + RoPE."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb

from kernel_adapters.qknorm_rope import fused_inplace

CASES = {"flux2-4096": (1, 4096, 24, 128), "flux2-4608": (1, 4608, 24, 128)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="flux2-4096")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; benchmark skipped.")
    batch, sequence, heads, head_dim = CASES[args.case]
    dtype = getattr(torch, args.dtype)
    q = torch.randn(batch, sequence, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    q_weight, k_weight = torch.randn(2, head_dim, device="cuda", dtype=dtype)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device="cuda") / head_dim))
    angles = torch.arange(sequence, device="cuda", dtype=torch.float32)[:, None] * inv_freq
    cos, sin = (torch.repeat_interleave(value, 2, dim=-1) for value in (angles.cos(), angles.sin()))

    def copy_inputs() -> None:
        q.copy_(q_seed)
        k.copy_(k_seed)

    def run_fused() -> bool:
        copy_inputs()
        return fused_inplace(q, k, q_weight, k_weight, cos, sin, 1e-6)

    q_seed, k_seed = q.clone(), k.clone()

    def run_reference() -> tuple[torch.Tensor, torch.Tensor]:
        q_norm = F.rms_norm(q_seed, (head_dim,), q_weight, 1e-6)
        k_norm = F.rms_norm(k_seed, (head_dim,), k_weight, 1e-6)
        return (
            apply_rotary_emb(q_norm, (cos, sin), sequence_dim=1),
            apply_rotary_emb(k_norm, (cos, sin), sequence_dim=1),
        )

    def latency_ms(function) -> float:
        for _ in range(args.warmup):
            function()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            function()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / args.iters

    with torch.inference_mode():
        if not run_fused():
            raise SystemExit("Kernel unavailable (unsupported runtime or JIT build failure).")
        expected_q, expected_k = run_reference()
        q_error = (q.float() - expected_q.float()).abs()
        k_error = (k.float() - expected_k.float()).abs()
        copy_ms = latency_ms(copy_inputs)
        fused_with_copy_ms = latency_ms(run_fused)
        reference_ms = latency_ms(run_reference)
    fused_ms = max(fused_with_copy_ms - copy_ms, 0.0)
    print(
        json.dumps(
            {
                "case": args.case,
                "shape": [batch, sequence, heads, head_dim],
                "dtype": args.dtype,
                "correctness": {
                    "q_max_abs_error": float(q_error.max()),
                    "q_mean_abs_error": float(q_error.mean()),
                    "k_max_abs_error": float(k_error.max()),
                    "k_mean_abs_error": float(k_error.mean()),
                    "q_cosine_similarity": float(
                        F.cosine_similarity(q.float().flatten(), expected_q.float().flatten(), dim=0)
                    ),
                    "k_cosine_similarity": float(
                        F.cosine_similarity(k.float().flatten(), expected_k.float().flatten(), dim=0)
                    ),
                },
                "copy_ms": copy_ms,
                "fused_with_copy_ms": fused_with_copy_ms,
                "fused_estimated_ms": fused_ms,
                "reference_ms": reference_ms,
                "speedup": reference_ms / fused_ms if fused_ms > 0 else None,
                "timestamp": time.time(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
