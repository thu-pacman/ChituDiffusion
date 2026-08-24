from __future__ import annotations

import importlib.metadata
import sys
import types
import warnings

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from chitu_diffusion.models.llada_image import text_encoder_compat as compat


def test_dense_flash_attention_matches_sdpa() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 4, 4, 8)
    key = torch.randn(2, 5, 2, 8)
    value = torch.randn(2, 5, 2, 8)

    actual = compat.flash_attn_func(query, key, value, softmax_scale=0.25)
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale=0.25,
        enable_gqa=True,
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected)


def test_varlen_flash_attention_matches_individual_sequences() -> None:
    torch.manual_seed(1)
    query = torch.randn(5, 2, 4)
    key = torch.randn(7, 2, 4)
    value = torch.randn(7, 2, 4)
    cu_query = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_key = torch.tensor([0, 4, 7], dtype=torch.int32)

    actual = compat.flash_attn_varlen_func(
        query,
        key,
        value,
        cu_query,
        cu_key,
        max_seqlen_q=3,
        max_seqlen_k=4,
        causal=True,
    )
    expected = torch.cat(
        [
            compat.flash_attn_func(
                query[0:2].unsqueeze(0),
                key[0:4].unsqueeze(0),
                value[0:4].unsqueeze(0),
                causal=True,
            ).squeeze(0),
            compat.flash_attn_func(
                query[2:5].unsqueeze(0),
                key[4:7].unsqueeze(0),
                value[4:7].unsqueeze(0),
                causal=True,
            ).squeeze(0),
        ]
    )

    torch.testing.assert_close(actual, expected)


def test_bert_padding_round_trip_and_lengths() -> None:
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    attention_mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)

    unpadded, indices, cumulative, maximum, used_lengths = compat.unpad_input(
        hidden_states, attention_mask
    )
    restored = compat.pad_input(unpadded, indices, batch=2, seqlen=3)

    torch.testing.assert_close(restored[attention_mask], hidden_states[attention_mask])
    torch.testing.assert_close(
        restored[~attention_mask], torch.zeros_like(restored[~attention_mask])
    )
    torch.testing.assert_close(cumulative, torch.tensor([0, 2, 4], dtype=torch.int32))
    torch.testing.assert_close(used_lengths, torch.tensor([2, 2], dtype=torch.int32))
    assert maximum == 2


@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_embedding_matches_manual_rotation(interleaved: bool) -> None:
    values = torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0, 4.0, 9.0]],
                [[5.0, 6.0, 7.0, 8.0, 10.0]],
            ],
            [
                [[11.0, 12.0, 13.0, 14.0, 15.0]],
                [[16.0, 17.0, 18.0, 19.0, 20.0]],
            ],
        ]
    )
    angles = torch.tensor([[0.0, 0.0], [0.2, 0.4], [0.6, 0.8]])
    offsets = torch.tensor([0, 1])
    cos = angles.cos()
    sin = angles.sin()

    actual = compat.apply_rotary_emb(
        values,
        cos,
        sin,
        interleaved=interleaved,
        seqlen_offsets=offsets,
    )

    expected = values.clone()
    for batch_index, offset in enumerate(offsets.tolist()):
        for sequence_index in range(values.shape[1]):
            rotation_cos = cos[offset + sequence_index]
            rotation_sin = sin[offset + sequence_index]
            source = values[batch_index, sequence_index, 0, :4]
            if interleaved:
                rotated_half = torch.tensor(
                    [-source[1], source[0], -source[3], source[2]]
                )
                expanded_cos = rotation_cos.repeat_interleave(2)
                expanded_sin = rotation_sin.repeat_interleave(2)
            else:
                rotated_half = torch.tensor(
                    [-source[2], -source[3], source[0], source[1]]
                )
                expanded_cos = torch.cat([rotation_cos, rotation_cos])
                expanded_sin = torch.cat([rotation_sin, rotation_sin])
            expected[batch_index, sequence_index, 0, :4] = (
                source * expanded_cos + rotated_half * expanded_sin
            )

    torch.testing.assert_close(actual, expected)


def test_fused_moe_forward_matches_explicit_expert_sum() -> None:
    torch.manual_seed(2)
    hidden_states = torch.randn(2, 3, 4)
    selected_experts = torch.tensor(
        [[[0, 1], [1, 2], [2, 0]], [[1, 0], [2, 1], [0, 2]]]
    )
    routing_weights = torch.rand(2, 3, 2)
    gate_weight = torch.randn(3, 5, 4)
    up_weight = torch.randn(3, 5, 4)
    down_weight = torch.randn(3, 4, 5)
    module = nn.Module()
    module.act_fn = F.silu

    actual = compat.fused_moe_forward(
        module,
        num_experts=3,
        routing_weights=routing_weights,
        selected_experts=selected_experts,
        hidden_states=hidden_states,
        fc1_1_weight=gate_weight,
        fc1_2_weight=up_weight,
        fc2_weight=down_weight,
    )

    expected = torch.zeros_like(hidden_states)
    for batch_index in range(hidden_states.shape[0]):
        for token_index in range(hidden_states.shape[1]):
            token = hidden_states[batch_index, token_index]
            for route_index in range(selected_experts.shape[-1]):
                expert_index = int(
                    selected_experts[batch_index, token_index, route_index]
                )
                gate = F.linear(token, gate_weight[expert_index])
                up = F.linear(token, up_weight[expert_index])
                expert_output = F.linear(F.silu(gate) * up, down_weight[expert_index])
                expected[batch_index, token_index] += (
                    expert_output
                    * routing_weights[batch_index, token_index, route_index]
                )

    torch.testing.assert_close(actual, expected)


def _clear_module_family(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    for name in tuple(sys.modules):
        if name == root or name.startswith(f"{root}."):
            monkeypatch.delitem(sys.modules, name)


def test_compatibility_context_registers_and_restores_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_module_family(monkeypatch, "flash_attn")
    _clear_module_family(monkeypatch, "veomni")
    original_flash = types.ModuleType("flash_attn")
    original_veomni = types.ModuleType("veomni")
    monkeypatch.setitem(sys.modules, "flash_attn", original_flash)
    monkeypatch.setitem(sys.modules, "veomni", original_veomni)
    monkeypatch.setattr(
        compat, "_probe_flash_attention", lambda: "forced flash failure"
    )
    monkeypatch.setattr(compat, "_probe_veomni", lambda: "forced veomni failure")
    original_version = importlib.metadata.version

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with compat.llada_text_frontend_compatibility() as status:
            assert status.flash_attention_fallback
            assert status.veomni_fallback
            assert status.fallback_active
            assert sys.modules["flash_attn"] is not original_flash
            assert sys.modules["veomni"] is not original_veomni
            assert importlib.metadata.version("flash_attn") == "2.8.3"
            assert sys.modules["flash_attn"].flash_attn_func is compat.flash_attn_func
            assert (
                sys.modules["veomni.ops"].fused_moe_forward is compat.fused_moe_forward
            )

    assert len(emitted) == 1
    assert "mathematically equivalent PyTorch fallbacks" in str(emitted[0].message)
    assert sys.modules["flash_attn"] is original_flash
    assert sys.modules["veomni"] is original_veomni
    assert "flash_attn.bert_padding" not in sys.modules
    assert "veomni.ops" not in sys.modules
    assert importlib.metadata.version is original_version


def test_compatibility_context_preserves_compatible_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_flash = types.ModuleType("flash_attn")
    native_veomni = types.ModuleType("veomni")
    monkeypatch.setitem(sys.modules, "flash_attn", native_flash)
    monkeypatch.setitem(sys.modules, "veomni", native_veomni)
    monkeypatch.setattr(compat, "_probe_flash_attention", lambda: None)
    monkeypatch.setattr(compat, "_probe_veomni", lambda: None)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with compat.llada_text_frontend_compatibility() as status:
            assert not status.fallback_active
            assert sys.modules["flash_attn"] is native_flash
            assert sys.modules["veomni"] is native_veomni

    assert not emitted
    assert sys.modules["flash_attn"] is native_flash
    assert sys.modules["veomni"] is native_veomni
