# SPDX-License-Identifier: Apache-2.0
"""Gate the CuTe SM90 attention kernel before adding NVSHMEM dependencies.

The kernel is loaded from the tracked FlashAttention checkout under
``refs/kernels/flash-attention``.  Sync it with
``python script/kernel_watch.py sync flash-attention`` before running.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


PROJECT_ROOT = Path(__file__).parents[2]
FLASH_ATTENTION_ROOT = PROJECT_ROOT / "refs/kernels/flash-attention"
UPSTREAM_COMMIT = "d7e4dba3e568106b0f1b6323b07c1272f53679b3"


def _load_cute_attention() -> Callable[..., tuple[torch.Tensor, ...]]:
    if not FLASH_ATTENTION_ROOT.is_dir():
        raise RuntimeError(
            "FlashAttention checkout is missing; run "
            "`python script/kernel_watch.py sync flash-attention`"
        )
    sys.path.insert(0, str(FLASH_ATTENTION_ROOT))
    from flash_attn.cute.interface import flash_attn_func

    return flash_attn_func


def _measure(call: Callable[[], torch.Tensor], warmup: int, iterations: int) -> dict[str, float]:
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
    }


def _cudnn_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--cp-degrees", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 0a requires an SM90 CUDA device")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise RuntimeError(f"Stage 0a requires SM90, got sm_{major}{minor}")

    flash_attn_func = _load_cute_attention()
    torch.manual_seed(0)
    reports: list[dict[str, object]] = []
    for cp_degree in args.cp_degrees:
        if args.sequence % cp_degree:
            raise ValueError("global sequence must be divisible by every CP degree")
        query_sequence = args.sequence // cp_degree
        shape_q = (1, query_sequence, args.heads, args.head_dim)
        shape_kv = (1, args.sequence, args.heads, args.head_dim)
        query = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)

        def cute_attention() -> torch.Tensor:
            return flash_attn_func(query, key, value, causal=False)[0]

        reference = _cudnn_attention(query, key, value)
        actual = cute_attention()
        torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-3)

        cute = _measure(cute_attention, args.warmup, args.iterations)
        cudnn = _measure(
            lambda: _cudnn_attention(query, key, value),
            args.warmup,
            args.iterations,
        )
        flops = 4 * query_sequence * args.sequence * args.heads * args.head_dim
        reports.append(
            {
                "cp_degree": cp_degree,
                "query_sequence": query_sequence,
                "kv_sequence": args.sequence,
                "cute": {
                    **cute,
                    "tflops": flops / float(cute["median_ms"]) / 1e9,
                },
                "cudnn": {
                    **cudnn,
                    "tflops": flops / float(cudnn["median_ms"]) / 1e9,
                },
                "max_abs_error": float((actual - reference).abs().max()),
            }
        )

    result = {
        "device": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "heads": args.heads,
        "head_dim": args.head_dim,
        "upstream": {
            "repository": "https://github.com/Dao-AILab/flash-attention",
            "commit": UPSTREAM_COMMIT,
            "source": "flash_attn/cute/flash_fwd_sm90.py",
            "license": "BSD-3-Clause",
        },
        "results": reports,
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
