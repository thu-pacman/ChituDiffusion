from __future__ import annotations

from importlib import metadata
from typing import Protocol

import torch
import torch.nn.functional as F
from packaging.version import InvalidVersion, Version

_MIN_FA4_SM120_VERSION = Version("4.0.0b25")


def _fa4_supports_compute_capability(
    capability: tuple[int, int],
    fa4_version: Version | None,
) -> bool:
    if capability != (12, 0):
        return True
    return fa4_version is not None and fa4_version >= _MIN_FA4_SM120_VERSION


class VarlenAttentionBackend(Protocol):
    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_k: int | None = None,
        causal: bool = False,
    ) -> torch.Tensor: ...


def _validate(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None = None,
) -> None:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("varlen Q/K/V must use packed [tokens, heads, dim] layout")
    if key.shape != value.shape:
        raise ValueError("K/V shapes must match")
    if query.shape[1:] != key.shape[1:]:
        raise ValueError("Q/K/V head shapes must match")
    cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional cumulative array")
    if cu_seqlens_k.ndim != 1 or cu_seqlens_k.numel() != cu_seqlens.numel():
        raise ValueError("Q/K cumulative arrays must describe the same segments")
    if int(cu_seqlens[-1].item()) != query.shape[0]:
        raise ValueError("cu_seqlens[-1] must equal the packed query token count")
    if int(cu_seqlens_k[-1].item()) != key.shape[0]:
        raise ValueError("cu_seqlens_k[-1] must equal the packed key token count")


def _segment_ids(cu_seqlens: torch.Tensor, token_count: int) -> torch.Tensor:
    positions = torch.arange(token_count, device=cu_seqlens.device)
    return torch.bucketize(positions, cu_seqlens[1:-1], right=True)


class SdpaVarlenBackend:
    """Portable correctness backend; evaluates each packed document separately."""

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_k: int | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        del max_seqlen, max_seqlen_k
        key_cu = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
        _validate(query, key, value, cu_seqlens, key_cu)
        query_bounds = [int(value) for value in cu_seqlens.tolist()]
        key_bounds = [int(value) for value in key_cu.tolist()]
        output = torch.empty_like(query)
        for q_start, q_stop, k_start, k_stop in zip(
            query_bounds[:-1],
            query_bounds[1:],
            key_bounds[:-1],
            key_bounds[1:],
            strict=True,
        ):
            if q_stop == q_start:
                continue
            segment = F.scaled_dot_product_attention(
                query[q_start:q_stop].transpose(0, 1),
                key[k_start:k_stop].transpose(0, 1),
                value[k_start:k_stop].transpose(0, 1),
                dropout_p=0.0,
                is_causal=causal,
            )
            output[q_start:q_stop] = segment.transpose(0, 1)
        return output


class FlexVarlenBackend:
    """BlockMask backend for document attention without a dense SxS mask."""

    def __init__(self) -> None:
        self._compiled = None

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_k: int | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        del max_seqlen, max_seqlen_k
        key_cu = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
        _validate(query, key, value, cu_seqlens, key_cu)
        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention,
        )

        query_count = query.shape[0]
        key_count = key.shape[0]
        query_document = _segment_ids(cu_seqlens.to(query.device), query_count)
        key_document = _segment_ids(key_cu.to(query.device), key_count)

        def document_mask(batch, head, q_index, kv_index):
            del batch, head
            visible = query_document[q_index] == key_document[kv_index]
            return visible & (q_index >= kv_index) if causal else visible

        block_mask = create_block_mask(
            document_mask,
            B=1,
            H=query.shape[1],
            Q_LEN=query_count,
            KV_LEN=key_count,
            device=str(query.device),
        )
        flex_impl = flex_attention
        if query.is_cuda:
            if self._compiled is None:
                self._compiled = torch.compile(flex_attention, dynamic=True)
            flex_impl = self._compiled
        output = flex_impl(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            block_mask=block_mask,
        )
        return output.squeeze(0).transpose(0, 1).contiguous()


class Fa4VarlenBackend:
    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_k: int | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        key_cu = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
        _validate(query, key, value, cu_seqlens, key_cu)
        if not query.is_cuda:
            raise RuntimeError("FlashAttention-4 requires CUDA")
        from flash_attn.cute.interface import flash_attn_varlen_func

        cu_q = cu_seqlens.to(device=query.device, dtype=torch.int32).contiguous()
        cu_k = key_cu.to(device=query.device, dtype=torch.int32).contiguous()
        result = flash_attn_varlen_func(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=int(max_seqlen),
            max_seqlen_k=int(
                max_seqlen if max_seqlen_k is None else max_seqlen_k
            ),
            causal=causal,
        )
        return result[0] if isinstance(result, tuple) else result


class AutoVarlenBackend:
    def __init__(self) -> None:
        self._fa4_version: Version | None = None
        try:
            from flash_attn.cute.interface import flash_attn_varlen_func  # noqa: F401

            self._fa4_available = True
            try:
                self._fa4_version = Version(metadata.version("flash-attn-4"))
            except (metadata.PackageNotFoundError, InvalidVersion):
                pass
        except ImportError:
            self._fa4_available = False
        self._fa4 = Fa4VarlenBackend()
        self._flex = FlexVarlenBackend()
        self._sdpa = SdpaVarlenBackend()

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_k: int | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        backend: VarlenAttentionBackend
        use_fa4 = False
        if query.is_cuda and self._fa4_available:
            capability = torch.cuda.get_device_capability(query.device)
            use_fa4 = _fa4_supports_compute_capability(
                capability,
                self._fa4_version,
            )
        if use_fa4:
            backend = self._fa4
        elif query.is_cuda:
            backend = self._flex
        else:
            backend = self._sdpa
        return backend.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )


def create_varlen_attention_backend(name: str = "auto") -> VarlenAttentionBackend:
    normalized = name.lower().replace("_", "-")
    if normalized == "auto":
        return AutoVarlenBackend()
    if normalized in {"fa4", "flash-attn-4"}:
        return Fa4VarlenBackend()
    if normalized == "flex":
        return FlexVarlenBackend()
    if normalized == "sdpa":
        return SdpaVarlenBackend()
    raise ValueError(f"unknown varlen attention backend: {name}")

