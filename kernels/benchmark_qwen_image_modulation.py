#!/usr/bin/env python3
"""Correctness and latency benchmark for Qwen-Image select-0/1 fusions."""

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

from kernel_adapters.qwen_image_modulation import (
    layernorm_scale_shift_gate_select01,
    layernorm_scale_shift_gate_select01_reference,
    residual_layernorm_scale_shift_gate_select01,
    residual_layernorm_scale_shift_gate_select01_reference,
    triton_available,
)

PRESETS = {"smoke": (1, 128, 3072), "qwen-image-1024": (1, 4096, 3072)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="qwen-image-1024")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    return parser.parse_args()


def latency_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def summarize(
    name: str,
    reference: tuple[torch.Tensor, ...],
    candidate: tuple[torch.Tensor, ...] | None,
    reference_ms: float,
    kernel_ms: float,
    rtol: float,
    atol: float,
) -> dict:
    if candidate is None:
        return {"operator": name, "status": "unavailable"}
    errors = [(actual.float() - expected.float()).abs() for actual, expected in zip(candidate, reference)]
    return {
        "operator": name,
        "status": "ok" if all(torch.allclose(a, b, rtol=rtol, atol=atol) for a, b in zip(candidate, reference)) else "incorrect",
        "max_abs_error": max(float(error.max()) for error in errors),
        "mean_abs_error": sum(float(error.mean()) for error in errors) / len(errors),
        "reference_ms": reference_ms,
        "kernel_ms": kernel_ms,
        "speedup": reference_ms / kernel_ms,
        "rtol": rtol,
        "atol": atol,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not triton_available():
        raise SystemExit("CUDA with Triton is required.")
    batch, sequence, dim = PRESETS[args.preset]
    torch.manual_seed(args.seed)
    x = torch.randn(batch, sequence, dim, device="cuda", dtype=torch.bfloat16)
    residual, residual_gate = torch.randn_like(x), torch.randn_like(x)
    scale0, shift0, gate0, scale1, shift1, gate1 = (
        torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16) for _ in range(6)
    )
    index = torch.randint(0, 2, (batch, sequence), device="cuda", dtype=torch.int32)
    common = (scale0, shift0, gate0, scale1, shift1, gate1, index, 1e-6)
    modulation_reference = lambda: layernorm_scale_shift_gate_select01_reference(x, *common)
    modulation_kernel = lambda: layernorm_scale_shift_gate_select01(x, *common)
    residual_reference = lambda: residual_layernorm_scale_shift_gate_select01_reference(
        x, residual, residual_gate, *common
    )
    residual_kernel = lambda: residual_layernorm_scale_shift_gate_select01(x, residual, residual_gate, *common)
    references = (modulation_reference(), residual_reference())
    candidates = (modulation_kernel(), residual_kernel())
    results = (
        summarize(
            "layernorm_scale_shift_gate_select01",
            references[0],
            candidates[0],
            latency_ms(modulation_reference, args.warmup, args.iters),
            latency_ms(modulation_kernel, args.warmup, args.iters),
            args.rtol,
            args.atol,
        ),
        summarize(
            "residual_layernorm_scale_shift_gate_select01",
            references[1],
            candidates[1],
            latency_ms(residual_reference, args.warmup, args.iters),
            latency_ms(residual_kernel, args.warmup, args.iters),
            args.rtol,
            args.atol,
        ),
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
                "shape": {"batch": batch, "sequence": sequence, "dim": dim},
                "results": results,
            },
            indent=2,
        )
    )
    return int(any(result["status"] == "incorrect" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
