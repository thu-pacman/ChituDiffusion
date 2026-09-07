from __future__ import annotations

from types import SimpleNamespace

import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2ParallelSelfAttention,
)

from chitu_diffusion.models.flux2_klein.attention import (
    Flux2KleinCpAttnProcessor,
    Flux2KleinCpSingleAttnProcessor,
)


class _SingleRankParallel:
    active = SimpleNamespace(process_group=None)
    active_agkv_transport = None
    active_ulysses = None


def test_flux2_klein_double_stream_cp_processor_matches_local_attention() -> None:
    torch.manual_seed(13)
    attention = Flux2Attention(
        query_dim=8,
        heads=2,
        dim_head=4,
        added_kv_proj_dim=8,
        out_dim=8,
    )
    image = torch.randn(1, 5, 8)
    text = torch.randn(1, 3, 8)
    expected = attention(image, encoder_hidden_states=text)

    processor = Flux2KleinCpAttnProcessor(_SingleRankParallel())
    processor.enable(text.shape[1])
    attention.set_processor(processor)
    actual = attention(image, encoder_hidden_states=text)

    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-6)


def test_flux2_klein_single_stream_cp_processor_matches_local_attention() -> None:
    torch.manual_seed(17)
    attention = Flux2ParallelSelfAttention(
        query_dim=8,
        heads=2,
        dim_head=4,
        out_dim=8,
        mlp_ratio=2.0,
    )
    hidden_states = torch.randn(1, 8, 8)
    expected = attention(hidden_states)

    processor = Flux2KleinCpSingleAttnProcessor(_SingleRankParallel())
    processor.enable(3)
    attention.set_processor(processor)
    actual = attention(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
