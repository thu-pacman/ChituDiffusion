# SPDX-License-Identifier: Apache-2.0
"""Portable padded-varlen helpers for diffusion attention.

Adapted from SGLang's
``python/sglang/jit_kernel/diffusion/triton/varlen_pack_pad.py`` and its
``multimodal_gen/runtime/layers/attention/layer.py`` metadata contract.
The latter is copied and adapted by SGLang from FastVideo; retain both
projects' attribution when moving this implementation.

The public API deliberately has a PyTorch implementation. Triton is optional:
when it is unavailable or the tensors cannot use the fused kernels, these
helpers fall back to exact ``index_select``/``index_copy_`` operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs must remain importable.
    triton = None
    tl = None


@dataclass(frozen=True)
class VarlenMetadata:
    """Metadata for a boolean valid-token mask in padded ``[B, S]`` layout."""

    indices: torch.Tensor
    inverse_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    batch_size: int
    sequence_length: int
    all_valid: bool

    @property
    def num_valid(self) -> int:
        return int(self.indices.numel())


def build_varlen_metadata(valid_mask: torch.Tensor) -> VarlenMetadata:
    """Build packing metadata for a strict boolean ``[batch, sequence]`` mask."""

    if valid_mask.ndim != 2 or valid_mask.dtype is not torch.bool:
        raise ValueError(
            "valid_mask must be a bool tensor with layout [batch, sequence]; "
            f"got dtype={valid_mask.dtype}, shape={tuple(valid_mask.shape)}."
        )
    batch_size, sequence_length = valid_mask.shape
    if batch_size < 1 or sequence_length < 1:
        raise ValueError(
            "valid_mask must have non-zero batch and sequence dimensions; "
            f"got shape={tuple(valid_mask.shape)}."
        )

    valid_lens = valid_mask.sum(dim=1, dtype=torch.int32)
    indices = valid_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    inverse_indices = torch.full(
        (batch_size * sequence_length,), -1, dtype=torch.int32, device=valid_mask.device
    )
    inverse_indices[indices.long()] = torch.arange(
        indices.numel(), dtype=torch.int32, device=valid_mask.device
    )
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=valid_mask.device)
    cu_seqlens[1:] = torch.cumsum(valid_lens, dim=0)
    return VarlenMetadata(
        indices=indices,
        inverse_indices=inverse_indices,
        cu_seqlens=cu_seqlens,
        max_seqlen=int(valid_lens.max().item()),
        batch_size=batch_size,
        sequence_length=sequence_length,
        all_valid=bool(valid_mask.all().item()),
    )


def triton_varlen_available(tensor: torch.Tensor | None = None) -> bool:
    """Whether the fused implementation can run for ``tensor``."""

    return bool(
        triton is not None
        and tensor is not None
        and tensor.is_cuda
        and tensor.dtype in (torch.float16, torch.bfloat16)
        and tensor.ndim == 4
    )


if triton is not None:

    @triton.jit
    def _pack_qkv_kernel(
        q_ptr, k_ptr, v_ptr, q_out_ptr, k_out_ptr, v_out_ptr, indices_ptr,
        feature_size, source_stride, destination_stride, BLOCK: tl.constexpr,
    ):
        output_row = tl.program_id(0)
        source_row = tl.load(indices_ptr + output_row).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        mask = columns < feature_size
        source_offset = source_row * source_stride + columns
        destination_offset = output_row * destination_stride + columns
        tl.store(q_out_ptr + destination_offset, tl.load(q_ptr + source_offset, mask=mask), mask=mask)
        tl.store(k_out_ptr + destination_offset, tl.load(k_ptr + source_offset, mask=mask), mask=mask)
        tl.store(v_out_ptr + destination_offset, tl.load(v_ptr + source_offset, mask=mask), mask=mask)

    @triton.jit
    def _scatter_kernel(
        packed_ptr, padded_ptr, inverse_indices_ptr, feature_size,
        source_stride, destination_stride, BLOCK: tl.constexpr,
    ):
        padded_row = tl.program_id(0)
        packed_row = tl.load(inverse_indices_ptr + padded_row).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        feature_mask = columns < feature_size
        is_valid = packed_row >= 0
        source_offset = tl.where(is_valid, packed_row, 0) * source_stride + columns
        destination_offset = padded_row * destination_stride + columns
        value = tl.load(packed_ptr + source_offset, mask=feature_mask & is_valid, other=0.0)
        tl.store(padded_ptr + destination_offset, value, mask=feature_mask)


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, meta: VarlenMetadata) -> None:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k and v must have identical [batch, sequence, heads, head_dim] shapes.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k and v must share a dtype.")
    if q.shape[:2] != (meta.batch_size, meta.sequence_length):
        raise ValueError(
            "q/k/v padded layout does not match metadata: "
            f"expected {(meta.batch_size, meta.sequence_length)}, got {tuple(q.shape[:2])}."
        )
    if q.device != meta.indices.device:
        raise ValueError("q/k/v and varlen metadata must reside on the same device.")


def pack_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, metadata: VarlenMetadata
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack valid Q/K/V rows, using Triton when available or exact PyTorch otherwise."""

    _validate_qkv(q, k, v, metadata)
    heads, head_dim = q.shape[2:]
    if metadata.num_valid == 0:
        empty = q.new_empty((0, heads, head_dim))
        return empty, empty.clone(), empty.clone()

    if triton_varlen_available(q):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        feature_size = heads * head_dim
        block = triton.next_power_of_2(feature_size)
        packed = [torch.empty((metadata.num_valid, feature_size), dtype=q.dtype, device=q.device) for _ in range(3)]
        q_flat, k_flat, v_flat = (item.view(-1, feature_size) for item in (q, k, v))
        _pack_qkv_kernel[(metadata.num_valid,)](
            q_flat, k_flat, v_flat, *packed, metadata.indices, feature_size,
            q_flat.stride(0), packed[0].stride(0), BLOCK=block,
        )
        return tuple(item.view(metadata.num_valid, heads, head_dim) for item in packed)

    flattened = (item.reshape(-1, heads, head_dim) for item in (q, k, v))
    return tuple(item.index_select(0, metadata.indices) for item in flattened)


def scatter_padded(packed: torch.Tensor, metadata: VarlenMetadata) -> torch.Tensor:
    """Scatter packed rows to padded layout and zero-fill invalid token rows."""

    if packed.ndim != 3 or packed.shape[0] != metadata.num_valid:
        raise ValueError(
            "packed must be [num_valid, heads, head_dim] matching metadata; "
            f"got shape={tuple(packed.shape)}, num_valid={metadata.num_valid}."
        )
    if packed.device != metadata.indices.device:
        raise ValueError("packed tensor and varlen metadata must reside on the same device.")
    heads, head_dim = packed.shape[1:]
    feature_size = heads * head_dim

    if triton_varlen_available(
        packed.new_empty((metadata.batch_size, metadata.sequence_length, heads, head_dim))
    ):
        padded = torch.empty(
            (metadata.batch_size * metadata.sequence_length, feature_size),
            dtype=packed.dtype,
            device=packed.device,
        )
        packed_flat = packed.contiguous().view(-1, feature_size)
        _scatter_kernel[(metadata.batch_size * metadata.sequence_length,)](
            packed_flat, padded, metadata.inverse_indices, feature_size,
            packed_flat.stride(0), padded.stride(0), BLOCK=triton.next_power_of_2(feature_size),
        )
        return padded.view(metadata.batch_size, metadata.sequence_length, heads, head_dim)

    padded = packed.new_zeros((metadata.batch_size * metadata.sequence_length, heads, head_dim))
    if metadata.num_valid:
        padded.index_copy_(0, metadata.indices, packed)
    return padded.view(metadata.batch_size, metadata.sequence_length, heads, head_dim)
