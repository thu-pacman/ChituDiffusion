"""Opt-in Qwen-Image LayerNorm/modulation/residual Triton fusions.

SPDX-License-Identifier: Apache-2.0

Adapted from SGLang commit ``429f6b6d157a9d8013903c4e4260b027e7bf5386``:
``python/sglang/jit_kernel/diffusion/triton/scale_shift.py`` and
``python/sglang/jit_kernel/diffusion/norm_scale_shift_native.py``.

Copyright 2025 SGLang Team.

The original Qwen-Image runtime integration was copied and adapted from
FastVideo (Apache-2.0); thank you to both the SGLang and FastVideo
communities. This module is an upstream-kernel compatibility adapter and does
not import SGLang, sgl-kernel, TVM FFI, or CuTeDSL.

The Triton reduction follows the same real-valued LayerNorm expression as
``torch.nn.LayerNorm(elementwise_affine=False)``. It is not promised bitwise
identical: the reduction tree and BF16 store rounding can differ from PyTorch.
Callers must keep it opt-in until their target stack has passed a strict
numerical gate.
"""

from __future__ import annotations

import importlib.util

import torch
import torch.nn.functional as F

_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
if _TRITON_AVAILABLE:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    @triton.jit
    def _layernorm_scale_shift_gate_select01_kernel(
        output_ptr,
        gate_out_ptr,
        x_ptr,
        scale0_ptr,
        shift0_ptr,
        gate0_ptr,
        scale1_ptr,
        shift1_ptr,
        gate1_ptr,
        index_ptr,
        inner_dim,
        seq_len,
        stride_x_row,
        stride_out_row,
        stride_gate_row,
        stride_s0_batch,
        stride_sh0_batch,
        stride_g0_batch,
        stride_s1_batch,
        stride_sh1_batch,
        stride_g1_batch,
        stride_index_batch,
        stride_index_seq,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < inner_dim
        x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / inner_dim
        centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / inner_dim
        # diffusers materializes BF16 LayerNorm output before modulation.
        normalized = ((x - mean) * tl.rsqrt(variance + eps)).to(tl.bfloat16)

        batch = row // seq_len
        sequence = row % seq_len
        select_one = tl.load(index_ptr + batch * stride_index_batch + sequence * stride_index_seq).to(tl.int1)
        if select_one:
            scale = tl.load(scale1_ptr + batch * stride_s1_batch + cols, mask=mask, other=0.0)
            shift = tl.load(shift1_ptr + batch * stride_sh1_batch + cols, mask=mask, other=0.0)
            gate = tl.load(gate1_ptr + batch * stride_g1_batch + cols, mask=mask, other=0.0)
        else:
            scale = tl.load(scale0_ptr + batch * stride_s0_batch + cols, mask=mask, other=0.0)
            shift = tl.load(shift0_ptr + batch * stride_sh0_batch + cols, mask=mask, other=0.0)
            gate = tl.load(gate0_ptr + batch * stride_g0_batch + cols, mask=mask, other=0.0)
        one_plus_scale = (1.0 + scale).to(tl.bfloat16)
        scaled = (normalized * one_plus_scale).to(tl.bfloat16)
        tl.store(
            output_ptr + row * stride_out_row + cols,
            (scaled + shift).to(tl.bfloat16),
            mask=mask,
        )
        tl.store(gate_out_ptr + row * stride_gate_row + cols, gate, mask=mask)

    @triton.jit
    def _residual_layernorm_scale_shift_gate_select01_kernel(
        output_ptr,
        residual_out_ptr,
        gate_out_ptr,
        x_ptr,
        residual_ptr,
        residual_gate_ptr,
        scale0_ptr,
        shift0_ptr,
        gate0_ptr,
        scale1_ptr,
        shift1_ptr,
        gate1_ptr,
        index_ptr,
        inner_dim,
        seq_len,
        stride_x_row,
        stride_residual_row,
        stride_residual_gate_row,
        stride_out_row,
        stride_residual_out_row,
        stride_gate_row,
        stride_s0_batch,
        stride_sh0_batch,
        stride_g0_batch,
        stride_s1_batch,
        stride_sh1_batch,
        stride_g1_batch,
        stride_index_batch,
        stride_index_seq,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < inner_dim
        x = tl.load(x_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row * stride_residual_row + cols, mask=mask, other=0.0)
        residual_gate = tl.load(
            residual_gate_ptr + row * stride_residual_gate_row + cols, mask=mask, other=0.0
        )
        gated = (residual_gate * x).to(tl.bfloat16)
        residual_out = (residual + gated).to(tl.bfloat16)
        residual_out_fp32 = residual_out.to(tl.float32)
        mean = tl.sum(residual_out_fp32, axis=0) / inner_dim
        centered = tl.where(mask, residual_out_fp32 - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / inner_dim
        normalized = ((residual_out_fp32 - mean) * tl.rsqrt(variance + eps)).to(tl.bfloat16)

        batch = row // seq_len
        sequence = row % seq_len
        select_one = tl.load(index_ptr + batch * stride_index_batch + sequence * stride_index_seq).to(tl.int1)
        if select_one:
            scale = tl.load(scale1_ptr + batch * stride_s1_batch + cols, mask=mask, other=0.0)
            shift = tl.load(shift1_ptr + batch * stride_sh1_batch + cols, mask=mask, other=0.0)
            gate = tl.load(gate1_ptr + batch * stride_g1_batch + cols, mask=mask, other=0.0)
        else:
            scale = tl.load(scale0_ptr + batch * stride_s0_batch + cols, mask=mask, other=0.0)
            shift = tl.load(shift0_ptr + batch * stride_sh0_batch + cols, mask=mask, other=0.0)
            gate = tl.load(gate0_ptr + batch * stride_g0_batch + cols, mask=mask, other=0.0)
        one_plus_scale = (1.0 + scale).to(tl.bfloat16)
        scaled = (normalized * one_plus_scale).to(tl.bfloat16)
        tl.store(
            output_ptr + row * stride_out_row + cols,
            (scaled + shift).to(tl.bfloat16),
            mask=mask,
        )
        tl.store(residual_out_ptr + row * stride_residual_out_row + cols, residual_out, mask=mask)
        tl.store(gate_out_ptr + row * stride_gate_row + cols, gate, mask=mask)


def triton_available() -> bool:
    """Return whether the optional Triton implementation is importable."""

    return _TRITON_AVAILABLE


def _valid_inputs(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
) -> bool:
    if (
        not _TRITON_AVAILABLE
        or not x.is_cuda
        or x.dtype is not torch.bfloat16
        or x.ndim != 3
        or not x.is_contiguous()
        or x.shape[-1] > 32768
        or index.shape != x.shape[:2]
        or index.device != x.device
        or index.ndim != 2
        or index.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        return False
    expected = (x.shape[0], x.shape[-1])
    tensors = (scale0, shift0, gate0, scale1, shift1, gate1)
    return all(
        tensor.device == x.device
        and tensor.dtype is x.dtype
        and tensor.shape == expected
        and tensor.is_contiguous()
        for tensor in tensors
    )


def layernorm_scale_shift_gate_select01_reference(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact PyTorch fallback matching diffusers' Qwen-Image block expression."""

    selected = index.to(torch.bool).unsqueeze(-1)
    scale = torch.where(selected, scale1.unsqueeze(1), scale0.unsqueeze(1))
    shift = torch.where(selected, shift1.unsqueeze(1), shift0.unsqueeze(1))
    gate = torch.where(selected, gate1.unsqueeze(1), gate0.unsqueeze(1))
    normalized = F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=eps)
    return normalized * (1 + scale) + shift, gate


def residual_layernorm_scale_shift_gate_select01_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact PyTorch fallback for residual, norm, modulation and gate selection."""

    residual_out = residual + residual_gate * x
    output, gate = layernorm_scale_shift_gate_select01_reference(
        residual_out, scale0, shift0, gate0, scale1, shift1, gate1, index, eps
    )
    return output, residual_out, gate


def layernorm_scale_shift_gate_select01(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse Qwen's select-0/1 LayerNorm/modulation on supported CUDA inputs."""

    if not _valid_inputs(x, scale0, shift0, gate0, scale1, shift1, gate1, index):
        return None
    batch, sequence, dim = x.shape
    block = triton.next_power_of_2(dim)
    output, gate_out = torch.empty_like(x), torch.empty_like(x)
    _layernorm_scale_shift_gate_select01_kernel[(batch * sequence,)](
        output,
        gate_out,
        x,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        dim,
        sequence,
        x.stride(1),
        output.stride(1),
        gate_out.stride(1),
        scale0.stride(0),
        shift0.stride(0),
        gate0.stride(0),
        scale1.stride(0),
        shift1.stride(0),
        gate1.stride(0),
        index.stride(0),
        index.stride(1),
        eps,
        BLOCK_N=block,
        num_warps=4,
    )
    return output, gate_out


def residual_layernorm_scale_shift_gate_select01(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fuse Qwen's residual, LayerNorm and select-0/1 modulation when safe."""

    if (
        not _valid_inputs(x, scale0, shift0, gate0, scale1, shift1, gate1, index)
        or residual.shape != x.shape
        or residual_gate.shape != x.shape
        or residual.device != x.device
        or residual_gate.device != x.device
        or residual.dtype is not x.dtype
        or residual_gate.dtype is not x.dtype
        or not residual.is_contiguous()
        or not residual_gate.is_contiguous()
    ):
        return None
    batch, sequence, dim = x.shape
    block = triton.next_power_of_2(dim)
    output, residual_out, gate_out = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    _residual_layernorm_scale_shift_gate_select01_kernel[(batch * sequence,)](
        output,
        residual_out,
        gate_out,
        x,
        residual,
        residual_gate,
        scale0,
        shift0,
        gate0,
        scale1,
        shift1,
        gate1,
        index,
        dim,
        sequence,
        x.stride(1),
        residual.stride(1),
        residual_gate.stride(1),
        output.stride(1),
        residual_out.stride(1),
        gate_out.stride(1),
        scale0.stride(0),
        shift0.stride(0),
        gate0.stride(0),
        scale1.stride(0),
        shift1.stride(0),
        gate1.stride(0),
        index.stride(0),
        index.stride(1),
        eps,
        BLOCK_N=block,
        num_warps=4,
    )
    return output, residual_out, gate_out
