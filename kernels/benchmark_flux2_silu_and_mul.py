#!/usr/bin/env python3
"""Correctness and latency gate for Flux2's exact-layout SiLU × mul fusion."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernel_adapters.silu_and_mul import silu_and_mul, triton_available

# Flux2-klein-4B: hidden=24*128=3072, MLP projection=2*(3*3072)=18432.
# 1024px image tokens are 4096 and the Qwen3 context is at most 512 tokens.
PRESETS = {
    "smoke": (1, 512, 18432),
    "flux2-dual-image-1024": (1, 4096, 18432),
    "flux2-dual-text-1024": (1, 512, 18432),
    "flux2-single-1024": (1, 4608, 18432),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="flux2-single-1024")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _latency_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Flux2 SiLU×mul benchmark.")
    if not triton_available():
        raise SystemExit("Triton is unavailable; install a Triton-enabled PyTorch environment.")
    if args.warmup < 0 or args.iters < 1:
        raise SystemExit("--warmup must be non-negative and --iters must be positive.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    batch, sequence, width = PRESETS[args.preset]
    x = torch.randn(batch, sequence, width, device="cuda", dtype=dtype)
    reference_fn = lambda: F.silu(x[..., : width // 2]) * x[..., width // 2 :]
    candidate_fn = lambda: silu_and_mul(x)
    reference, candidate = reference_fn(), candidate_fn()
    if candidate is None:
        raise SystemExit("Fusion was unavailable for a supported CUDA benchmark input.")

    abs_error = (candidate.float() - reference.float()).abs()
    bitwise_equal = bool(torch.equal(candidate, reference))
    reference_ms = _latency_ms(reference_fn, args.warmup, args.iters)
    candidate_ms = _latency_ms(candidate_fn, args.warmup, args.iters)
    major, minor = torch.cuda.get_device_capability()
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": f"{major}.{minor}",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "shape": {"preset": args.preset, "batch": batch, "sequence": sequence, "width": width},
        "dtype": args.dtype,
        "correctness": {
            "bitwise_equal": bitwise_equal,
            "max_abs_error": float(abs_error.max()),
            "mean_abs_error": float(abs_error.mean()),
        },
        "latency_ms": {"pytorch": reference_ms, "triton": candidate_ms, "speedup": reference_ms / candidate_ms},
    }
    print(json.dumps(result, indent=2))
    return 0 if bitwise_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
