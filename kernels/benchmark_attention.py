#!/usr/bin/env python3
"""Correctness and latency gate for diffusion-shaped attention kernels."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chitu_diffusion.modules.attention.diffusion_attn_backend import (  # noqa: E402 - imports follow the repository path bootstrap
    DiffusionAttnBackend,
)

PRESETS = {
    "smoke": (1, 512, 8, 64),
    "z-image-1024": (1, 4096, 30, 128),
    "qwen-image-1024": (1, 4096, 24, 128),
    "wan-long": (1, 32760, 12, 128),
}


@dataclass
class Result:
    backend: str
    status: str
    latency_ms: float | None = None
    speedup: float | None = None
    max_abs_error: float | None = None
    cosine_similarity: float | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="smoke")
    parser.add_argument("--backends", nargs="+", default=["torch_sdpa", "flash_attn3", "flash_attn"])
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-error", type=float, default=5e-2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def latency_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the kernel benchmark")
    if args.warmup < 0 or args.iters < 1:
        raise SystemExit("--warmup must be non-negative and --iters must be positive")

    torch.manual_seed(args.seed)
    batch, sequence, heads, head_dim = PRESETS[args.preset]
    dtype = getattr(torch, args.dtype)
    q = torch.randn(batch, sequence, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    reference_backend = DiffusionAttnBackend("torch_sdpa")
    with torch.inference_mode():
        reference = reference_backend(q, k, v, causal=False)[0]

    results: list[Result] = []
    baseline_ms = None
    for name in args.backends:
        try:
            backend = DiffusionAttnBackend(name)

            def invoke():
                return backend(q, k, v, causal=False)[0]

            with torch.inference_mode():
                output = invoke()
                elapsed = latency_ms(invoke, args.warmup, args.iters)
            if name == "torch_sdpa":
                baseline_ms = elapsed
            error = None
            cosine = None
            if not args.skip_correctness:
                error = float((output.float() - reference.float()).abs().max())
                cosine = float(
                    torch.nn.functional.cosine_similarity(
                        output.float().flatten(), reference.float().flatten(), dim=0
                    )
                )
            results.append(
                Result(
                    backend=name,
                    status="ok" if error is None or error <= args.max_error else "incorrect",
                    latency_ms=elapsed,
                    max_abs_error=error,
                    cosine_similarity=cosine,
                )
            )
        except (AttributeError, ImportError, NotImplementedError, OSError, RuntimeError, ValueError) as error:
            results.append(Result(backend=name, status="unavailable", error=str(error)))

    if baseline_ms is None:
        baseline = next((item.latency_ms for item in results if item.status == "ok"), None)
    else:
        baseline = baseline_ms
    if baseline:
        for result in results:
            if result.latency_ms:
                result.speedup = baseline / result.latency_ms

    major, minor = torch.cuda.get_device_capability()
    payload = {
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
            "heads": heads,
            "head_dim": head_dim,
            "dtype": args.dtype,
        },
        "results": [asdict(result) for result in results],
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return int(any(result.status == "incorrect" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
