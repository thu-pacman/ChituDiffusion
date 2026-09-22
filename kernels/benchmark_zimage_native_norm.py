#!/usr/bin/env python3
"""Correctness and latency benchmark for Z-Image RMSNorm Triton fusions."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernel_adapters.zimage_native_norm import (  # noqa: E402 - imports follow the repository path bootstrap
    rmsnorm_scale,
    rmsnorm_tanh_residual,
    triton_available,
)

# Z-Image's 1024px denoise path is dominated by 4096 image tokens.  The
# default dim is configurable because checkpoint variants may use a different
# transformer width.
PRESETS = {"smoke": (1, 512, 3840), "z-image-1024": (1, 4096, 3840)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="z-image-1024")
    parser.add_argument(
        "--dim", type=int, help="Override transformer hidden dimension."
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    return parser.parse_args()


def latency_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def diffusers_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(weight.dtype) * weight


def native_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Mirrors the retained SGLang BF16 reduction.
    mean_square = (
        x.square().to(torch.bfloat16).mean(dim=-1, keepdim=True).to(torch.bfloat16)
    )
    rstd = torch.rsqrt((mean_square + eps).to(torch.float32)).to(torch.bfloat16)
    return (x * rstd).to(torch.bfloat16) * weight


def result(
    operator: str,
    mode: str,
    reference: torch.Tensor,
    candidate: torch.Tensor | None,
    reference_ms: float,
    kernel_ms: float,
    rtol: float,
    atol: float,
) -> dict:
    if candidate is None:
        return {"operator": operator, "mode": mode, "status": "unavailable"}
    reference_fp32, candidate_fp32 = reference.float(), candidate.float()
    absolute_error = (candidate_fp32 - reference_fp32).abs()
    cosine = torch.nn.functional.cosine_similarity(
        candidate_fp32.flatten(), reference_fp32.flatten(), dim=0
    )
    allclose = torch.allclose(candidate_fp32, reference_fp32, rtol=rtol, atol=atol)
    return {
        "operator": operator,
        "mode": mode,
        "status": "ok" if allclose else "incorrect",
        "max_abs_error": float(absolute_error.max()),
        "mean_abs_error": float(absolute_error.mean()),
        "cosine_similarity": float(cosine),
        "allclose": bool(allclose),
        "rtol": rtol,
        "atol": atol,
        "reference_ms": reference_ms,
        "kernel_ms": kernel_ms,
        "speedup": reference_ms / kernel_ms,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Z-Image native norm benchmark.")
    if not triton_available():
        raise SystemExit(
            "Triton is unavailable; install a Triton-enabled PyTorch environment."
        )
    if args.warmup < 0 or args.iters < 1:
        raise SystemExit("--warmup must be non-negative and --iters must be positive.")

    torch.manual_seed(args.seed)
    batch, sequence, default_dim = PRESETS[args.preset]
    dim = args.dim or default_dim
    x = torch.randn(batch, sequence, dim, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(batch, 1, dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(batch, 1, dim, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6

    results = []
    for mode, rmsnorm_reference in (
        ("diffusers_fp32", diffusers_rmsnorm),
        ("native_bf16", native_rmsnorm),
    ):

        def scale_reference_fn():
            return (rmsnorm_reference(x, weight, eps) * scale).to(torch.bfloat16)

        def residual_reference_fn():
            return (
                residual
                + torch.tanh(gate).to(torch.bfloat16)
                * rmsnorm_reference(x, weight, eps)
            ).to(torch.bfloat16)

        def scale_kernel_fn():
            return rmsnorm_scale(x, weight, scale, eps, mode=mode)

        def residual_kernel_fn():
            return rmsnorm_tanh_residual(x, gate, residual, weight, eps, mode=mode)

        scale_reference, residual_reference = (
            scale_reference_fn(),
            residual_reference_fn(),
        )
        scale_output, residual_output = scale_kernel_fn(), residual_kernel_fn()
        scale_reference_ms = latency_ms(scale_reference_fn, args.warmup, args.iters)
        residual_reference_ms = latency_ms(
            residual_reference_fn, args.warmup, args.iters
        )
        scale_kernel_ms = latency_ms(scale_kernel_fn, args.warmup, args.iters)
        residual_kernel_ms = latency_ms(residual_kernel_fn, args.warmup, args.iters)
        results.extend(
            (
                result(
                    "rmsnorm_scale",
                    mode,
                    scale_reference,
                    scale_output,
                    scale_reference_ms,
                    scale_kernel_ms,
                    args.rtol,
                    args.atol,
                ),
                result(
                    "rmsnorm_tanh_residual",
                    mode,
                    residual_reference,
                    residual_output,
                    residual_reference_ms,
                    residual_kernel_ms,
                    args.rtol,
                    args.atol,
                ),
            )
        )

    major, minor = torch.cuda.get_device_capability()
    print(
        json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "environment": {
                    "host": platform.node(),
                    "gpu": torch.cuda.get_device_name(),
                    "compute_capability": f"{major}.{minor}",
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                "shape": {
                    "preset": args.preset,
                    "batch": batch,
                    "sequence": sequence,
                    "dim": dim,
                },
                "results": results,
            },
            indent=2,
        )
    )
    return int(any(item["status"] == "incorrect" for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
