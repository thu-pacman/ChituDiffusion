from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers import Flux2KleinPipeline
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2ParallelSelfAttention,
)

from chitu_diffusion.models.flux2_klein.attention import (
    Flux2KleinCpAttnProcessor,
    Flux2KleinCpSingleAttnProcessor,
)
from chitu_diffusion.models.flux2_klein.pipeline import Flux2KleinCpPipeline


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


@pytest.mark.parametrize(
    "output_type,return_dict",
    [("pil", True), ("np", False), ("pt", True), ("latent", False)],
)
@pytest.mark.parametrize("leader", [True, False])
def test_flux2_pipeline_routes_final_decode_and_preserves_output_contract(
    monkeypatch, output_type, return_dict, leader
):
    from chitu_diffusion.models.flux2_klein import pipeline as integration

    latent = torch.randn(1, 4, 8, 6)
    calls = []

    def generate(self, image=None, prompt=None, *, output_type="pil", return_dict=True):
        calls.append((image, prompt, output_type, return_dict))
        assert output_type == "latent"
        return (latent,)

    monkeypatch.setattr(Flux2KleinPipeline, "__call__", generate)

    def decode(vae, x, *, topology, enabled):
        assert x is latent
        assert enabled is True
        vae._chitu_vae_decode_stats = {"vae_decode_mode": "layerwise"}
        return x if topology.is_leader else None

    monkeypatch.setattr(integration, "parallel_vae_decode", decode)
    pipeline = object.__new__(Flux2KleinCpPipeline)
    processed = []
    pipeline.__dict__.update(
        vae=SimpleNamespace(),
        _cp_parallel_context=SimpleNamespace(active=SimpleNamespace(is_leader=leader)),
        image_processor=SimpleNamespace(
            postprocess=lambda image, output_type: (
                processed.append(output_type) or [image]
            )
        ),
        maybe_free_model_hooks=lambda: None,
    )
    output = pipeline(None, "a cube", output_type=output_type, return_dict=return_dict)
    assert calls == [(None, "a cube", "latent", False)]
    if output_type == "latent":
        assert output[0] is latent
        assert pipeline.last_vae_stats is None
        assert not processed
    else:
        images = output.images if return_dict else output[0]
        assert len(images) == int(leader)
        assert processed == ([output_type] if leader else [])
        assert pipeline.last_vae_stats["vae_decode_mode"] == "layerwise"
