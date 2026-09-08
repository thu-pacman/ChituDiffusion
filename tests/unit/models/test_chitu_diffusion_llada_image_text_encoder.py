from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from chitu_diffusion.models.llada_image.text_encoder.configuration import (
    LLaDA2MoeConfig,
)
from chitu_diffusion.models.llada_image.text_encoder.modeling import (
    LLaDA2MoeAttention,
    LLaDA2MoeDecoderLayer,
    LLaDA2MoeModelLM,
    LLaDA2MoeSdpaAttention,
)
from chitu_diffusion.models.llada_image.text_encoder_ops import fused_moe_forward


def _tiny_config() -> LLaDA2MoeConfig:
    return LLaDA2MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        pad_token_id=0,
        first_k_dense_replace=1,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        n_group=2,
        topk_group=1,
        moe_intermediate_size=8,
    )


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

    actual = fused_moe_forward(
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


def test_local_text_encoder_loader_restores_every_sharded_weight(tmp_path) -> None:
    config = _tiny_config()
    expected = LLaDA2MoeModelLM(config).eval()
    assert all(torch.isfinite(value).all() for value in expected.parameters())
    expected.save_pretrained(
        tmp_path,
        safe_serialization=True,
        max_shard_size="20KB",
    )

    actual = LLaDA2MoeModelLM.from_local_pretrained(tmp_path)

    assert actual.training is False
    assert not any(value.is_meta for value in actual.state_dict().values())
    for name, expected_value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], expected_value)


@pytest.mark.parametrize(
    ("implementation", "expected_class"),
    (("eager", LLaDA2MoeAttention), ("sdpa", LLaDA2MoeSdpaAttention)),
)
def test_text_attention_backend_selection_is_explicit(
    implementation: str,
    expected_class: type[torch.nn.Module],
) -> None:
    config = _tiny_config()
    config._attn_implementation = implementation

    layer = LLaDA2MoeDecoderLayer(config, layer_idx=0)

    assert type(layer.attention) is expected_class


def test_text_attention_backend_rejects_unsupported_flash_attention() -> None:
    config = _tiny_config()
    config._attn_implementation = "flash_attention_2"

    with pytest.raises(ValueError, match="supports only.*eager.*sdpa"):
        LLaDA2MoeDecoderLayer(config, layer_idx=0)


def test_transfer_token_selection_matches_threshold_and_topk_rules() -> None:
    active = torch.tensor([[True, True, True, False]])
    confidence = torch.tensor([[0.99, 0.98, 0.20, 1.00]])

    threshold_selected = LLaDA2MoeModelLM._select_transfer_tokens(
        active,
        confidence,
        count=1,
        threshold=0.95,
    )
    topk_selected = LLaDA2MoeModelLM._select_transfer_tokens(
        active,
        confidence,
        count=3,
        threshold=0.995,
    )

    assert torch.equal(threshold_selected, torch.tensor([[True, True, False, False]]))
    assert torch.equal(topk_selected, active)


def test_completed_stop_requires_a_fully_generated_prefix() -> None:
    tokens = torch.tensor([10, 99, 11, 12])
    masked_prefix = torch.tensor([10, 77, 99, 12])

    assert (
        LLaDA2MoeModelLM._completed_stop_offset(
            tokens,
            stop_token=99,
            mask_token=77,
        )
        == 1
    )
    assert LLaDA2MoeModelLM._completed_stop_offset(
        masked_prefix,
        stop_token=99,
        mask_token=77,
    ) == len(masked_prefix)
