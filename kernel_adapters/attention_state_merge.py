# SPDX-License-Identifier: Apache-2.0
"""Exact CP attention-state merge with a guarded Triton fast path.

Adapted semantically from vLLM's Apache-2.0 ``merge_attn_states`` at commit
``616c9bd0f4efff9761fd762e4247e18b6bd7eb23``:
``vllm/v1/attention/ops/merge_attn_states.py`` and
``csrc/libtorch_stable/attention/merge_attn_states.cu``.

The vLLM CUDA op cannot be called directly here: it requires token-major
``[T,H,D]`` inputs and same-dtype input/output, whereas Chitu CP retains a
float32 running output ``[B,S,H,D]`` and receives BF16/FP16 block outputs.
This compatibility layer preserves Chitu's float32 state contract and only
uses Triton when every layout and dtype invariant is satisfied.

Copyright contributors to the vLLM project.
Licensed under the Apache License, Version 2.0.
Thank you to the vLLM community for publishing the merge algorithm and CUDA
reference implementation.
"""

from __future__ import annotations

import atexit
import importlib.util
import os

import torch

_TRACE_STATS = os.getenv("CHITU_CP_ATTN_STATE_MERGE_TRACE", "0") == "1"
_TRACE_COUNTS = {"calls": 0, "triton": 0, "off": 0, "unsupported": 0, "errors": 0}


def _report_trace_counts() -> None:
    if not _TRACE_STATS:
        return
    rank = os.getenv("RANK", "?")
    print(
        "[cp-attn-state-merge] "
        f"rank={rank} calls={_TRACE_COUNTS['calls']} triton={_TRACE_COUNTS['triton']} "
        f"off={_TRACE_COUNTS['off']} unsupported={_TRACE_COUNTS['unsupported']} "
        f"errors={_TRACE_COUNTS['errors']}",
        flush=True,
    )


atexit.register(_report_trace_counts)

_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
if _TRITON_AVAILABLE:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    @triton.jit
    def _merge_attention_states_kernel(
        out_ptr,
        lse_ptr,
        block_out_ptr,
        block_lse_ptr,
        lse_batch_stride,
        lse_seq_stride,
        lse_head_stride,
        seq_len: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        row = tl.program_id(0)
        token_head = seq_len * num_heads
        batch = row // token_head
        within_batch = row % token_head
        token = within_batch // num_heads
        head = within_batch % num_heads

        # Chitu: out/lse = [B,S,H,D]/[B,S,H,1], block_lse = [B,H,S].
        lse_offset = (
            batch * lse_batch_stride
            + token * lse_seq_stride
            + head * lse_head_stride
        )
        p_lse = tl.load(lse_ptr + lse_offset)
        s_lse = tl.load(block_lse_ptr + (batch * num_heads + head) * seq_len + token)
        neg_inf = -float("inf")
        # Match the upstream contract: any infinite partial LSE represents an
        # empty partial attention state, not a weight of +/- infinity.
        p_lse = tl.where((p_lse == float("inf")) | (p_lse == neg_inf), neg_inf, p_lse)
        s_lse = tl.where((s_lse == float("inf")) | (s_lse == neg_inf), neg_inf, s_lse)
        max_lse = tl.maximum(p_lse, s_lse)
        both_empty = max_lse == neg_inf
        p_exp = tl.exp(p_lse - max_lse)
        s_exp = tl.exp(s_lse - max_lse)
        denom = p_exp + s_exp
        p_scale = p_exp / denom
        s_scale = s_exp / denom

        offsets = tl.arange(0, block_dim)
        mask = offsets < head_dim
        base = row * head_dim + offsets
        p_out = tl.load(out_ptr + base, mask=mask, other=0.0).to(tl.float32)
        s_out = tl.load(block_out_ptr + base, mask=mask, other=0.0).to(tl.float32)
        merged = p_out * p_scale + s_out * s_scale
        tl.store(out_ptr + base, tl.where(both_empty, p_out, merged), mask=mask)
        out_lse = tl.log(denom) + max_lse
        tl.store(lse_ptr + lse_offset, tl.where(both_empty, neg_inf, out_lse))


def merge_mode() -> str:
    """Return the CP merge policy; runtime activation is intentionally disabled."""

    mode = os.getenv("CHITU_CP_ATTN_STATE_MERGE", "off").strip().lower()
    if mode == "triton":
        raise ValueError(
            "CHITU_CP_ATTN_STATE_MERGE=triton is disabled: the real Ring CP gate "
            "showed no steady-state gain and material output drift. Use "
            "'benchmark_only' only in the isolated harness."
        )
    if mode not in {"off", "benchmark_only"}:
        raise ValueError("CHITU_CP_ATTN_STATE_MERGE must be one of: off, benchmark_only")
    return mode


def triton_merge_available() -> bool:
    return _TRITON_AVAILABLE


def _supported(out: torch.Tensor, lse: torch.Tensor, block_out: torch.Tensor, block_lse: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and out.is_cuda
        and out.device == lse.device == block_out.device == block_lse.device
        and out.dtype == torch.float32
        and lse.dtype == torch.float32
        and block_lse.dtype == torch.float32
        and block_out.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and out.ndim == 4
        and lse.shape == (*out.shape[:3], 1)
        and block_out.shape == out.shape
        and block_lse.shape == (out.shape[0], out.shape[2], out.shape[1])
        and out.is_contiguous()
        and block_out.is_contiguous()
        and block_lse.is_contiguous()
        and 1 <= out.shape[-1] <= 256
    )


def try_merge_attention_states(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    *,
    benchmark: bool = False,
) -> bool:
    """Merge in place only for the isolated benchmark harness.

    An unsupported layout, unavailable Triton, or compilation/launch failure
    returns ``False`` so the caller can execute its established eager merge.
    Production callers must leave ``benchmark=False``.
    """

    mode = merge_mode()
    if _TRACE_STATS:
        _TRACE_COUNTS["calls"] += 1
    if mode != "benchmark_only" or not benchmark:
        if _TRACE_STATS:
            _TRACE_COUNTS["off"] += 1
        return False
    if not _supported(out, lse, block_out, block_lse):
        if _TRACE_STATS:
            _TRACE_COUNTS["unsupported"] += 1
        return False
    try:
        block_dim = triton.next_power_of_2(out.shape[-1])
        _merge_attention_states_kernel[(out.shape[0] * out.shape[1] * out.shape[2],)](
            out,
            lse,
            block_out,
            block_lse,
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            seq_len=out.shape[1],
            num_heads=out.shape[2],
            head_dim=out.shape[3],
            block_dim=block_dim,
            num_warps=4,
        )
    except Exception:
        if _TRACE_STATS:
            _TRACE_COUNTS["errors"] += 1
        return False
    if _TRACE_STATS:
        _TRACE_COUNTS["triton"] += 1
    return True


def merge_attention_states_reference(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable float32 reference for Chitu's BSHD/BHS CP state contract."""

    suffix_lse = block_lse.transpose(-2, -1).unsqueeze(-1)
    neg_inf = torch.full((), -torch.inf, device=lse.device, dtype=lse.dtype)
    prefix_lse = torch.where(torch.isinf(lse), neg_inf, lse)
    suffix_lse = torch.where(torch.isinf(suffix_lse), neg_inf, suffix_lse)
    merged_lse = torch.logaddexp(prefix_lse, suffix_lse)
    both_empty = torch.isneginf(merged_lse)
    prefix_scale = torch.exp(prefix_lse - merged_lse)
    suffix_scale = torch.exp(suffix_lse - merged_lse)
    merged_out = out * prefix_scale + block_out.float() * suffix_scale
    return torch.where(both_empty, out, merged_out), merged_lse
