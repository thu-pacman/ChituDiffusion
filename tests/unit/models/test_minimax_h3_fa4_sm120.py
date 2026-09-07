from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch

from chitu_diffusion.parallel.cp.attention_backend import (
    Fa4VarlenBackend,
    SdpaVarlenBackend,
)


def _sm120_fa4_available() -> bool:
    if find_spec("flash_attn") is None:
        return False
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (12, 0)
        and find_spec("flash_attn.cute.interface") is not None
    )


@pytest.mark.skipif(
    not _sm120_fa4_available(),
    reason="requires FlashAttention-4 on an SM120 GPU",
)
@pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 129])
def test_sm120_fa4_single_document_matches_sdpa(length: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(length)
    query = torch.randn(
        length,
        14,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device="cuda")

    actual = Fa4VarlenBackend().forward_varlen(
        query,
        key,
        value,
        cu_seqlens=cu_seqlens,
        max_seqlen=length,
    )
    expected = SdpaVarlenBackend().forward_varlen(
        query,
        key,
        value,
        cu_seqlens=cu_seqlens,
        max_seqlen=length,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not _sm120_fa4_available(),
    reason="requires FlashAttention-4 on an SM120 GPU",
)
def test_sm120_fa4_h3_padding_document_matches_sdpa() -> None:
    query = torch.randn(128, 14, 128, dtype=torch.bfloat16, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.tensor([0, 80, 128], dtype=torch.int32, device="cuda")

    actual = Fa4VarlenBackend().forward_varlen(
        query,
        key,
        value,
        cu_seqlens=cu_seqlens,
        max_seqlen=80,
    )
    expected = SdpaVarlenBackend().forward_varlen(
        query,
        key,
        value,
        cu_seqlens=cu_seqlens,
        max_seqlen=80,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
