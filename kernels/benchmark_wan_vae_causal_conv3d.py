#!/usr/bin/env python3
"""Benchmark exact Wan VAE cache cat+pad data movement on CUDA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernel_adapters.wan_vae_causal_conv3d import (  # noqa: E402 - imports follow the repository path bootstrap
    try_causal_conv3d_cat_pad,
)


@dataclass(frozen=True)
class Case:
    name: str
    channels: int
    frames: int
    height: int
    width: int
    cache_frames: int


CASES = {
    # WanVAE decoder: latent 480x832 -> [B, 384, 1, 60, 104].
    "wan-decode-480p": Case("wan-decode-480p", 384, 1, 60, 104, 2),
    # WanVAE encoder's first feature layer for an actual 480x832 video chunk.
    "wan-encode-480p": Case("wan-encode-480p", 96, 4, 480, 832, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="wan-decode-480p")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def torch_reference(x: torch.Tensor, cache_x: torch.Tensor, padding: list[int]) -> torch.Tensor:
    adjusted = padding.copy()
    adjusted[4] -= cache_x.shape[2]
    return F.pad(torch.cat([cache_x, x], dim=2), adjusted)


def latency_ms(function, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.warmup < 0 or args.iters < 1:
        raise SystemExit("--warmup must be non-negative and --iters must be positive")

    case = CASES[args.case]
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    x = torch.randn((1, case.channels, case.frames, case.height, case.width), device="cuda", dtype=dtype)
    cache_x = torch.randn(
        (1, case.channels, case.cache_frames, case.height, case.width), device="cuda", dtype=dtype
    )
    padding = [1, 1, 1, 1, 2, 0]
    reference = torch_reference(x, cache_x, padding)
    os.environ["CHITU_WAN_VAE_CAT_PAD"] = "triton"
    fused = try_causal_conv3d_cat_pad(x, cache_x, padding)
    if fused is None:
        raise SystemExit("Triton path was unavailable; benchmark refuses to silently measure fallback")
    if not torch.equal(fused, reference):
        raise SystemExit("bitwise correctness failed")

    reference_ms = latency_ms(lambda: torch_reference(x, cache_x, padding), args.warmup, args.iters)
    fused_ms = latency_ms(lambda: try_causal_conv3d_cat_pad(x, cache_x, padding), args.warmup, args.iters)
    result = {
        "case": asdict(case),
        "dtype": args.dtype,
        "reference_ms": reference_ms,
        "triton_ms": fused_ms,
        "speedup": reference_ms / fused_ms,
        "max_abs_error": 0.0,
        "bitwise_equal": True,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
