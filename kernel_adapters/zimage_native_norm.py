"""Optional Triton fusions for the diffusers Z-Image transformer.

Adapted from SGLang's ``zimage_native_norm.py`` at commit
``d6ef68881e263812d4901f632786015005c4d050``.

Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License.  You may obtain a copy at
http://www.apache.org/licenses/LICENSE-2.0

Thank you to the SGLang community for publishing and maintaining the original
diffusion Triton kernels. This is a small API adaptation; it is not a fork of
the SGLang runtime.
"""

from __future__ import annotations

import importlib.util
from typing import Literal

import torch

ZImageNormMode = Literal["diffusers_fp32", "native_bf16"]
_VALID_MODES = {"diffusers_fp32", "native_bf16"}

_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
if _TRITON_AVAILABLE:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    @triton.jit
    def _tanh(x):
        return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0

    @triton.jit
    def _rmsnorm_scale_kernel(
        y_ptr,
        x_ptr,
        weight_ptr,
        scale_ptr,
        x_row_stride,
        scale_row_stride,
        seq_len,
        dim: tl.constexpr,
        eps: tl.constexpr,
        block_dim: tl.constexpr,
        native_bf16: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_dim)
        mask = offsets < dim
        x = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0)
        if native_bf16:
            square = (x * x).to(tl.bfloat16)
            mean_square = (tl.sum(square, axis=0) / dim).to(tl.bfloat16)
            rstd = tl.rsqrt((mean_square + eps).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        else:
            x_fp32 = x.to(tl.float32)
            mean_square = tl.sum(x_fp32 * x_fp32, axis=0) / dim
            rstd = tl.rsqrt(mean_square + eps)
        batch = row // seq_len
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        scale = tl.load(scale_ptr + batch * scale_row_stride + offsets, mask=mask, other=0.0)
        y = (((x * rstd).to(tl.bfloat16) * weight).to(tl.bfloat16) * scale).to(tl.bfloat16)
        tl.store(y_ptr + row * dim + offsets, y, mask=mask)

    @triton.jit
    def _rmsnorm_tanh_residual_kernel(
        y_ptr,
        x_ptr,
        gate_ptr,
        residual_ptr,
        weight_ptr,
        x_row_stride,
        gate_row_stride,
        residual_row_stride,
        seq_len,
        dim: tl.constexpr,
        eps: tl.constexpr,
        block_dim: tl.constexpr,
        native_bf16: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_dim)
        mask = offsets < dim
        x = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0)
        if native_bf16:
            square = (x * x).to(tl.bfloat16)
            mean_square = (tl.sum(square, axis=0) / dim).to(tl.bfloat16)
            rstd = tl.rsqrt((mean_square + eps).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        else:
            x_fp32 = x.to(tl.float32)
            mean_square = tl.sum(x_fp32 * x_fp32, axis=0) / dim
            rstd = tl.rsqrt(mean_square + eps)
        batch = row // seq_len
        gate = tl.load(gate_ptr + batch * gate_row_stride + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + row * residual_row_stride + offsets, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        norm = ((x * rstd).to(tl.bfloat16) * weight).to(tl.bfloat16)
        gated = (_tanh(gate.to(tl.float32)).to(tl.bfloat16) * norm).to(tl.bfloat16)
        tl.store(y_ptr + row * dim + offsets, (residual + gated).to(tl.bfloat16), mask=mask)


def triton_available() -> bool:
    """Whether the optional Triton dependency can be used by this process."""

    return _TRITON_AVAILABLE


def _flat_row_stride(x: torch.Tensor) -> int | None:
    if x.ndim < 2 or x.stride(-1) != 1:
        return None
    row_stride = x.stride(-2)
    expected_stride = row_stride * x.shape[-2]
    for dimension in range(x.ndim - 3, -1, -1):
        if x.stride(dimension) != expected_stride:
            return None
        expected_stride *= x.shape[dimension]
    return row_stride


def _can_use(x: torch.Tensor, weight: torch.Tensor, other: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and x.is_cuda
        and x.dtype is torch.bfloat16
        and weight.device == x.device
        and other.device == x.device
        and weight.dtype is torch.bfloat16
        and other.dtype is torch.bfloat16
        and weight.is_contiguous()
        and x.shape[-1] <= 8192
        and _flat_row_stride(x) is not None
        and _flat_row_stride(other) is not None
    )


def _validate_mode(mode: str) -> ZImageNormMode:
    if mode not in _VALID_MODES:
        raise ValueError(f"Z-Image norm mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
    return mode  # type: ignore[return-value]


def rmsnorm_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    *,
    mode: ZImageNormMode = "diffusers_fp32",
) -> torch.Tensor | None:
    """Fuse BF16 ``RMSNorm(x) * scale`` with an explicit reduction mode.

    ``diffusers_fp32`` reproduces diffusers' FP32 variance reduction.
    ``native_bf16`` reproduces SGLang's experimental BF16 reduction. Returning
    ``None`` lets the adapter retain the corresponding unfused expression.
    """

    mode = _validate_mode(mode)
    if not _can_use(x, weight, scale):
        return None
    dim = x.shape[-1]
    x_rows, scale_rows = x.numel() // dim, scale.numel() // dim
    if x_rows % scale_rows:
        return None
    x_row_stride, scale_row_stride = _flat_row_stride(x), _flat_row_stride(scale)
    if x_row_stride is None or scale_row_stride is None:
        return None
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.get_device_module().device(x.device):
        _rmsnorm_scale_kernel[(x_rows,)](
            output.reshape(-1, dim),
            x,
            weight,
            scale,
            x_row_stride,
            scale_row_stride,
            x_rows // scale_rows,
            dim,
            eps,
            block_dim=triton.next_power_of_2(dim),
            num_warps=8,
            native_bf16=mode == "native_bf16",
        )
    return output


def rmsnorm_tanh_residual(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    mode: ZImageNormMode = "diffusers_fp32",
) -> torch.Tensor | None:
    """Fuse residual/gate/RMSNorm with exactly one tanh and explicit reduction."""

    mode = _validate_mode(mode)
    if not (_can_use(x, weight, gate) and residual.device == x.device and residual.dtype is x.dtype):
        return None
    dim = x.shape[-1]
    x_rows, gate_rows = x.numel() // dim, gate.numel() // dim
    if x_rows % gate_rows:
        return None
    x_row_stride, gate_row_stride, residual_row_stride = (
        _flat_row_stride(x),
        _flat_row_stride(gate),
        _flat_row_stride(residual),
    )
    if x_row_stride is None or gate_row_stride is None or residual_row_stride is None:
        return None
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.get_device_module().device(x.device):
        _rmsnorm_tanh_residual_kernel[(x_rows,)](
            output.reshape(-1, dim),
            x,
            gate,
            residual,
            weight,
            x_row_stride,
            gate_row_stride,
            residual_row_stride,
            x_rows // gate_rows,
            dim,
            eps,
            block_dim=triton.next_power_of_2(dim),
            num_warps=8,
            native_bf16=mode == "native_bf16",
        )
    return output
