from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from chitu_diffusion.models.minimax_h3.qwen3vl_config import (
    MiniMaxH3Qwen3VLConfig,
)
from chitu_diffusion.models.minimax_h3.qwen3vl_encoder import (
    MiniMaxH3Qwen3VLEncoder,
)
from chitu_diffusion.models.minimax_h3.qwen3vl_loader import (
    load_minimax_h3_qwen3vl_weights,
)
from chitu_diffusion.models.minimax_h3.qwen3vl_processor import (
    MiniMaxH3Qwen3VLProcessor,
    build_h3_presentation,
)
from chitu_diffusion.parallel.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from chitu_diffusion.parallel.tensor_parallel import (
    TensorParallelTopology,
    reset_tensor_parallel_topology,
    set_tensor_parallel_topology,
)


def _tiny_config(**overrides) -> MiniMaxH3Qwen3VLConfig:
    values = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        mrope_section=(1, 1, 0),
        max_position_embeddings=64,
    )
    values.update(overrides)
    return MiniMaxH3Qwen3VLConfig(**values)


class _Tokenizer:
    special = {
        "<|vision_start|>": 100,
        "<|vision_end|>": 101,
        "<|image_pad|>": 102,
    }

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    def convert_tokens_to_ids(self, token):
        return self.special[token]


class _ImageProcessor:
    merge_size = 2

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.arange(len(images) * 4).view(-1, 1).float(),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images)),
        }


class _FakeVision(nn.Module):
    def __init__(self, pooled, deepstack):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.pooled = pooled
        self.deepstack = deepstack

    def forward(self, pixel_values, grid_thw):
        assert grid_thw.shape[1] == 3
        return SimpleNamespace(
            pooler_output=self.pooled,
            deepstack_features=self.deepstack,
        )


class _IdentityLayer(nn.Module):
    def forward(self, hidden_states, cos, sin, attention_mask):
        return hidden_states


def test_h3_prompt_only_presentation_is_verbatim() -> None:
    ids, tags = build_h3_presentation(_Tokenizer(), " A\nB ")
    assert ids.tolist() == [ord(char) for char in " A\nB "]
    assert tags.tolist() == [1] * len(ids)


def test_h3_fl2va_presentation_matches_native_contract() -> None:
    ids, tags = build_h3_presentation(
        _Tokenizer(), "P", image_token_counts=(2, 1)
    )
    assert ids.tolist() == [
        ord(":"),
        ord(" "),
        100,
        102,
        102,
        101,
        ord("P"),
        ord(":"),
        ord(" "),
        100,
        102,
        101,
        ord("P"),
    ]
    assert tags.tolist() == [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]


def test_processor_preprocesses_first_last_and_builds_3d_positions() -> None:
    processor = MiniMaxH3Qwen3VLProcessor(
        _Tokenizer(), _ImageProcessor(), spatial_merge_size=2
    )
    output = processor("P", first_image=object(), last_image=object())
    assert output.pixel_values.shape == (8, 1)
    assert output.image_grid_thw.tolist() == [[1, 4, 4], [1, 4, 4]]
    assert output.position_ids.shape == (3, 1, output.input_ids.numel())
    assert int((output.input_ids == 102).sum()) == 8
    assert output.token_type_ids.tolist() == output.token_tags.tolist()
    first_visual = torch.argwhere(output.input_ids == 102).squeeze(1)[:4]
    assert output.position_ids[:, 0, first_visual].tolist() == [
        [3, 3, 3, 3],
        [3, 3, 4, 4],
        [3, 4, 3, 4],
    ]


def test_processor_rejects_grid_placeholder_mismatch() -> None:
    processor = MiniMaxH3Qwen3VLProcessor(
        _Tokenizer(), _ImageProcessor(), spatial_merge_size=2
    )
    ids, _ = build_h3_presentation(_Tokenizer(), "P", image_token_counts=(3,))
    from chitu_diffusion.models.minimax_h3.qwen3vl_processor import (
        build_qwen3vl_position_ids,
    )

    with pytest.raises(ValueError, match="placeholder count"):
        build_qwen3vl_position_ids(
            ids,
            torch.tensor([[1, 4, 4]]),
            image_token_id=102,
            spatial_merge_size=2,
        )


def test_tiny_encoder_returns_unnormalized_layer_prefix() -> None:
    model = MiniMaxH3Qwen3VLEncoder(_tiny_config())
    assert isinstance(model.layers[0].self_attn.qkv_proj, MergedColumnParallelLinear)
    assert isinstance(model.layers[0].self_attn.o_proj, RowParallelLinear)
    assert isinstance(model.layers[0].mlp.gate_up_proj, MergedColumnParallelLinear)
    assert isinstance(model.layers[0].mlp.down_proj, RowParallelLinear)
    assert not hasattr(model, "norm")

    result = model(torch.tensor([1, 2, 3]))
    assert result.shape == (3, 16)
    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()


def test_visual_features_replace_embeddings_and_deepstack_adds() -> None:
    config = _tiny_config(vocab_size=128, image_token_id=102)
    model = MiniMaxH3Qwen3VLEncoder(config)
    model.embed_tokens.weight.data.zero_()
    model.layers = nn.ModuleList([_IdentityLayer(), _IdentityLayer()])
    pooled = torch.full((2, 16), 5, dtype=torch.bfloat16)
    deepstack = [
        torch.full((2, 16), 2, dtype=torch.bfloat16),
        torch.full((2, 16), 3, dtype=torch.bfloat16),
    ]
    model.visual = _FakeVision(pooled, deepstack)
    ids = torch.tensor([1, 102, 102, 2])
    result = model(
        ids,
        pixel_values=torch.zeros(2, 1),
        image_grid_thw=torch.tensor([[1, 2, 4]]),
    )
    torch.testing.assert_close(
        result[1:3], torch.full((2, 16), 10, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(result[[0, 3]], torch.zeros(2, 16).bfloat16())


def test_visual_feature_count_mismatch_is_rejected() -> None:
    config = _tiny_config(vocab_size=128, image_token_id=102)
    model = MiniMaxH3Qwen3VLEncoder(config)
    model.visual = _FakeVision(torch.zeros(1, 16), [])
    with pytest.raises(ValueError, match="does not match 2 image tokens"):
        model(
            torch.tensor([102, 102]),
            pixel_values=torch.zeros(2, 1),
            image_grid_thw=torch.tensor([[1, 2, 4]]),
        )


def test_deepstack_feature_count_mismatch_is_rejected() -> None:
    config = _tiny_config(vocab_size=128, image_token_id=102)
    model = MiniMaxH3Qwen3VLEncoder(config)
    model.layers = nn.ModuleList([_IdentityLayer(), _IdentityLayer()])
    model.visual = _FakeVision(
        torch.zeros(2, 16), [torch.zeros(1, 16)]
    )
    with pytest.raises(ValueError, match="does not match 2 image tokens"):
        model(
            torch.tensor([102, 102]),
            pixel_values=torch.zeros(2, 1),
            image_grid_thw=torch.tensor([[1, 2, 4]]),
        )


def test_production_gqa_is_tp4_shardable() -> None:
    config = MiniMaxH3Qwen3VLConfig()
    config.validate_tensor_parallel(4)
    assert config.num_attention_heads // 4 == 16
    assert config.num_key_value_heads // 4 == 2


def test_streaming_loader_reads_only_rank_local_linear_shards(tmp_path) -> None:
    topology = TensorParallelTopology(
        ranks=(0, 1, 2, 3),
        rank=2,
        rank_in_group=2,
        degree=4,
        process_group=None,
    )
    set_tensor_parallel_topology(topology)
    try:
        config = _tiny_config(
            num_hidden_layers=1,
            num_key_value_heads=4,
            intermediate_size=16,
        )
        model = MiniMaxH3Qwen3VLEncoder(config, device="meta")
        model.visual = nn.Linear(2, 2, device="meta")
        prefix = "model.language_model.layers.0"
        tensors = {
            "model.language_model.embed_tokens.weight": torch.arange(
                32 * 16, dtype=torch.float32
            ).reshape(32, 16),
            f"{prefix}.self_attn.q_proj.weight": torch.arange(
                16 * 16, dtype=torch.float32
            ).reshape(16, 16),
            f"{prefix}.self_attn.k_proj.weight": torch.arange(
                16 * 16, dtype=torch.float32
            ).reshape(16, 16)
            + 1000,
            f"{prefix}.self_attn.v_proj.weight": torch.arange(
                16 * 16, dtype=torch.float32
            ).reshape(16, 16)
            + 2000,
            f"{prefix}.self_attn.o_proj.weight": torch.arange(
                16 * 16, dtype=torch.float32
            ).reshape(16, 16),
            f"{prefix}.self_attn.q_norm.weight": torch.ones(4),
            f"{prefix}.self_attn.k_norm.weight": torch.ones(4),
            f"{prefix}.mlp.gate_proj.weight": torch.arange(
                16 * 16, dtype=torch.float32
            ).reshape(16, 16),
            f"{prefix}.mlp.up_proj.weight": torch.arange(
                16 * 16, dtype=torch.float32
            ).reshape(16, 16)
            + 3000,
            f"{prefix}.mlp.down_proj.weight": torch.arange(
                16 * 16, dtype=torch.float32
            ).reshape(16, 16),
            f"{prefix}.input_layernorm.weight": torch.ones(16),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(16),
            # Deliberately unconsumed checkpoint tensors.
            "model.language_model.norm.weight": torch.ones(16),
            "model.language_model.layers.50.input_layernorm.weight": torch.ones(16),
            "model.visual.weight": torch.arange(4).view(2, 2).float(),
            "model.visual.bias": torch.tensor([7.0, 8.0]),
        }
        shard = tmp_path / "model.safetensors"
        save_file(tensors, shard)
        (tmp_path / "config.json").write_text(json.dumps({}))

        load_minimax_h3_qwen3vl_weights(model, tmp_path, device="cpu")

        qkv = model.layers[0].self_attn.qkv_proj.weight
        torch.testing.assert_close(
            qkv[:4], tensors[f"{prefix}.self_attn.q_proj.weight"][8:12].bfloat16()
        )
        torch.testing.assert_close(
            qkv[4:8], tensors[f"{prefix}.self_attn.k_proj.weight"][8:12].bfloat16()
        )
        assert qkv.shape == (12, 16)
        assert model.layers[0].self_attn.o_proj.weight.shape == (16, 4)
        assert model.layers[0].mlp.down_proj.weight.shape == (16, 4)
        torch.testing.assert_close(
            model.visual.weight, tensors["model.visual.weight"]
        )
    finally:
        reset_tensor_parallel_topology()
