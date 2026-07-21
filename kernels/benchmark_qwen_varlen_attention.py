#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark Qwen-Image single-device masked varlen attention phases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttnBackend
from kernel_adapters.varlen import build_varlen_metadata, pack_qkv, scatter_padded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="flash_attn", choices=["flash_attn", "flash_attn3"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--text-seq", type=int, default=256)
    parser.add_argument("--image-seq", type=int, default=4096)
    parser.add_argument("--valid-text", type=int, default=128)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def timed_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the varlen benchmark.")
    if min(args.batch, args.text_seq, args.image_seq, args.heads, args.head_dim, args.iters) < 1:
        raise SystemExit("All dimensions and --iters must be positive.")
    if not 0 <= args.valid_text <= args.text_seq:
        raise SystemExit("--valid-text must be in [0, --text-seq].")

    dtype = getattr(torch, args.dtype)
    sequence = args.text_seq + args.image_seq
    mask = torch.ones(args.batch, sequence, dtype=torch.bool, device="cuda")
    mask[:, : args.text_seq] = False
    mask[:, : args.valid_text] = True
    q = torch.randn(args.batch, sequence, args.heads, args.head_dim, device="cuda", dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    backend = DiffusionAttnBackend(args.backend)

    metadata = build_varlen_metadata(mask)
    packed_q, packed_k, packed_v = pack_qkv(q, k, v, metadata)
    packed_out = backend(
        packed_q, packed_k, packed_v,
        cu_seqlens_q=metadata.cu_seqlens,
        cu_seqlens_k=metadata.cu_seqlens,
        max_seqlen_q=metadata.max_seqlen,
        max_seqlen_k=metadata.max_seqlen,
        causal=False,
    )[0]

    def varlen_pipeline():
        current_metadata = build_varlen_metadata(mask)
        current_q, current_k, current_v = pack_qkv(q, k, v, current_metadata)
        current_out = backend(
            current_q,
            current_k,
            current_v,
            cu_seqlens_q=current_metadata.cu_seqlens,
            cu_seqlens_k=current_metadata.cu_seqlens,
            max_seqlen_q=current_metadata.max_seqlen,
            max_seqlen_k=current_metadata.max_seqlen,
            causal=False,
        )[0]
        return scatter_padded(current_out, current_metadata)

    def masked_sdpa():
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        return out.masked_fill(~mask[:, :, None, None], 0)

    phases = {
        "metadata_ms": timed_ms(lambda: build_varlen_metadata(mask), args.warmup, args.iters),
        "pack_ms": timed_ms(lambda: pack_qkv(q, k, v, metadata), args.warmup, args.iters),
        "attention_ms": timed_ms(
            lambda: backend(
                packed_q, packed_k, packed_v,
                cu_seqlens_q=metadata.cu_seqlens,
                cu_seqlens_k=metadata.cu_seqlens,
                max_seqlen_q=metadata.max_seqlen,
                max_seqlen_k=metadata.max_seqlen,
                causal=False,
            )[0],
            args.warmup,
            args.iters,
        ),
        "scatter_ms": timed_ms(lambda: scatter_padded(packed_out, metadata), args.warmup, args.iters),
        "pipeline_ms": timed_ms(varlen_pipeline, args.warmup, args.iters),
        "masked_sdpa_ms": timed_ms(masked_sdpa, args.warmup, args.iters),
    }
    phases["phase_sum_ms"] = sum(
        phases[name] for name in ("metadata_ms", "pack_ms", "attention_ms", "scatter_ms")
    )
    phases["speedup_vs_masked_sdpa"] = phases["masked_sdpa_ms"] / phases["pipeline_ms"]
    varlen_output = varlen_pipeline()
    reference_output = masked_sdpa()
    valid_output = mask[:, :, None, None]
    difference = (varlen_output.float() - reference_output.float()).masked_select(valid_output)
    reference_valid = reference_output.float().masked_select(valid_output)
    varlen_valid = varlen_output.float().masked_select(valid_output)
    payload = {
        "backend": args.backend,
        "dtype": args.dtype,
        "shape": {
            "batch": args.batch, "text_seq": args.text_seq, "image_seq": args.image_seq,
            "valid_text": args.valid_text, "heads": args.heads, "head_dim": args.head_dim,
            "num_valid": metadata.num_valid,
        },
        "phases": phases,
        "correctness": {
            "max_abs_error": float(difference.abs().max()),
            "mean_abs_error": float(difference.abs().mean()),
            "cosine_similarity": float(F.cosine_similarity(varlen_valid, reference_valid, dim=0)),
        },
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
