# SPDX-License-Identifier: Apache-2.0
"""Exact Wan VAE causal Conv3D cache-concatenation and padding adapter.

Adapted from SGLang commit ``d6ef68881e263812d4901f632786015005c4d050``:
``python/sglang/jit_kernel/diffusion/causal_conv3d_cat_pad.py`` and
``python/sglang/jit_kernel/diffusion/triton/causal_conv3d_pad.py``.  The
Wan VAE call site is
``python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py``.

This is deliberately only a copy/zero-fill operation.  It neither casts nor
computes on tensor values, so eligible outputs are bitwise equal to
``F.pad(torch.cat([cache_x, x], dim=2), adjusted_padding)``.  Thank you to
the SGLang community for publishing the original Apache-2.0 implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments must retain the torch fallback.
    triton = None
    tl = None


@dataclass(frozen=True)
class CatPadStats:
    cached_calls: int
    supported_calls: int
    fused_calls: int
    fallback_calls: int


_STATS = {"cached_calls": 0, "supported_calls": 0, "fused_calls": 0, "fallback_calls": 0}


def cat_pad_stats() -> CatPadStats:
    """Return process-local counters proving whether the cached path executed."""

    return CatPadStats(**_STATS)


def reset_cat_pad_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0


def mode() -> str:
    """Return the explicit Wan VAE data-movement policy."""

    value = os.getenv("CHITU_WAN_VAE_CAT_PAD", "off").strip().lower()
    if value not in {"off", "triton"}:
        raise ValueError("CHITU_WAN_VAE_CAT_PAD must be one of: off, triton")
    return value


def triton_available() -> bool:
    return triton is not None


def supported(x: torch.Tensor, cache_x: torch.Tensor, padding: tuple[int, ...] | list[int]) -> bool:
    """Whether the layout has SGLang's exact 5D cat+pad contract."""

    if len(padding) != 6:
        return False
    pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_t_left, pad_t_right = padding
    return (
        triton is not None
        and x.is_cuda
        and x.device == cache_x.device
        and x.dtype == cache_x.dtype
        and x.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and x.ndim == cache_x.ndim == 5
        and x.shape[0:2] == cache_x.shape[0:2]
        and x.shape[3:] == cache_x.shape[3:]
        and x.is_contiguous()
        and cache_x.is_contiguous()
        and pad_w_left == pad_w_right
        and pad_h_top == pad_h_bottom
        and pad_t_right == 0
        and cache_x.shape[2] > 0
        and pad_t_left >= cache_x.shape[2]
        and all(value >= 0 for value in padding)
    )


if triton is not None:

    @triton.jit
    def _causal_conv3d_cat_pad_kernel(
        x_ptr,
        cache_ptr,
        out_ptr,
        total,
        channels: tl.constexpr,
        t_size: tl.constexpr,
        h_size: tl.constexpr,
        w_size: tl.constexpr,
        cache_t: tl.constexpr,
        out_t: tl.constexpr,
        out_h: tl.constexpr,
        out_w: tl.constexpr,
        pad_t_left: tl.constexpr,
        pad_h_top: tl.constexpr,
        pad_w_left: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        out_w_index = offsets % out_w
        remaining = offsets // out_w
        out_h_index = remaining % out_h
        remaining = remaining // out_h
        out_t_index = remaining % out_t
        remaining = remaining // out_t
        channel = remaining % channels
        batch = remaining // channels

        in_w = out_w_index - pad_w_left
        in_h = out_h_index - pad_h_top
        source_t = out_t_index - pad_t_left
        valid = (
            mask
            & (in_w >= 0)
            & (in_w < w_size)
            & (in_h >= 0)
            & (in_h < h_size)
            & (source_t >= 0)
            & (source_t < cache_t + t_size)
        )
        from_cache = source_t < cache_t
        x_t = source_t - cache_t
        safe_w = tl.minimum(tl.maximum(in_w, 0), w_size - 1)
        safe_h = tl.minimum(tl.maximum(in_h, 0), h_size - 1)
        safe_x_t = tl.minimum(tl.maximum(x_t, 0), t_size - 1)
        safe_cache_t = tl.minimum(tl.maximum(source_t, 0), cache_t - 1)
        x_offsets = ((batch * channels + channel) * t_size + safe_x_t) * h_size * w_size + safe_h * w_size + safe_w
        cache_offsets = (
            ((batch * channels + channel) * cache_t + safe_cache_t) * h_size * w_size + safe_h * w_size + safe_w
        )
        x_values = tl.load(x_ptr + x_offsets, mask=valid & ~from_cache, other=0.0)
        cache_values = tl.load(cache_ptr + cache_offsets, mask=valid & from_cache, other=0.0)
        tl.store(out_ptr + offsets, tl.where(from_cache, cache_values, x_values), mask=mask)


def try_causal_conv3d_cat_pad(
    x: torch.Tensor, cache_x: torch.Tensor, padding: tuple[int, ...] | list[int]
) -> torch.Tensor | None:
    """Return an exact fused result, or ``None`` for the established torch path."""

    _STATS["cached_calls"] += 1
    is_supported = supported(x, cache_x, padding)
    if is_supported:
        _STATS["supported_calls"] += 1
    if mode() != "triton" or not is_supported:
        _STATS["fallback_calls"] += 1
        return None
    pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_t_left, pad_t_right = padding
    del pad_w_right, pad_h_bottom, pad_t_right
    adjusted_pad_t_left = pad_t_left - cache_x.shape[2]
    try:
        output = torch.empty(
            (
                x.shape[0],
                x.shape[1],
                x.shape[2] + pad_t_left,
                x.shape[3] + 2 * pad_h_top,
                x.shape[4] + 2 * pad_w_left,
            ),
            device=x.device,
            dtype=x.dtype,
        )
        block_size = 256
        _causal_conv3d_cat_pad_kernel[(triton.cdiv(output.numel(), block_size),)](
            x,
            cache_x,
            output,
            output.numel(),
            channels=x.shape[1],
            t_size=x.shape[2],
            h_size=x.shape[3],
            w_size=x.shape[4],
            cache_t=cache_x.shape[2],
            out_t=output.shape[2],
            out_h=output.shape[3],
            out_w=output.shape[4],
            pad_t_left=adjusted_pad_t_left,
            pad_h_top=pad_h_top,
            pad_w_left=pad_w_left,
            block_size=block_size,
        )
    except Exception:
        _STATS["fallback_calls"] += 1
        return None
    _STATS["fused_calls"] += 1
    return output
