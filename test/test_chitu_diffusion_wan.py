from __future__ import annotations

import pytest
import torch

from chitu_diffusers import WanRequest
from chitu_diffusers.epac.api import DiffusersEPACPipeline
from chitu_diffusers.epac.model_scheduling import EpeSchedulingModule
from chitu_diffusers.models.wan.api import WanEPACPipeline
from chitu_diffusers.models.wan.executor import WanVideoDecoderExecutor
from chitu_diffusers.models.wan.loader import convert_wan_umt5_encoder_state_dict
from chitu_diffusers.models.wan.pipeline import combine_wan_cfg_predictions


def test_wan_request_validates_video_shape() -> None:
    request = WanRequest(prompt="test", num_frames=17, num_steps=4)

    assert request.width == 832
    assert request.height == 480
    assert request.guidance_scale == 6.0
    assert request.output_type == "np"
    with pytest.raises(ValueError, match=r"4n\+1"):
        WanRequest(prompt="test", num_frames=16)


def test_wan_request_builds_diffusers_inputs() -> None:
    request = WanRequest(
        prompt="test",
        negative_prompt="bad",
        num_frames=17,
        num_steps=4,
        guidance_scale=5.0,
        seed=7,
    )

    diffusion = request.to_diffusion_request(device=torch.device("cpu"))

    assert diffusion.inputs["negative_prompt"] == "bad"
    assert diffusion.inputs["num_frames"] == 17
    assert diffusion.inputs["guidance_scale"] == 5.0
    assert diffusion.num_inference_steps == 4


def test_wan_cfg_combination_matches_diffusers_formula() -> None:
    positive = torch.tensor([2.0])
    negative = torch.tensor([-1.0])

    output = combine_wan_cfg_predictions(positive, negative, guidance_scale=6.0)

    assert torch.equal(output, negative + 6.0 * (positive - negative))


def test_wan_uses_shared_api_layer() -> None:
    assert issubclass(WanEPACPipeline, DiffusersEPACPipeline)


def test_wan_umt5_key_mapping_is_complete() -> None:
    checkpoint = {
        "token_embedding.weight": torch.empty(1),
        "norm.weight": torch.empty(1),
    }
    for index in range(24):
        prefix = f"blocks.{index}"
        for name in (
            "norm1.weight",
            "attn.q.weight",
            "attn.k.weight",
            "attn.v.weight",
            "attn.o.weight",
            "norm2.weight",
            "ffn.gate.0.weight",
            "ffn.fc1.weight",
            "ffn.fc2.weight",
            "pos_embedding.embedding.weight",
        ):
            checkpoint[f"{prefix}.{name}"] = torch.empty(1)

    converted = convert_wan_umt5_encoder_state_dict(checkpoint)

    assert len(converted) == 243
    assert "encoder.block.23.layer.1.DenseReluDense.wo.weight" in converted
    assert not checkpoint


def test_wan_executor_profiles_video_tokens_and_state() -> None:
    class FakeParallel:
        rank = 0
        local_rank = 0
        world_size = 1
        allowed_widths = (1,)

    class FakeVae:
        config = type("Config", (), {"scale_factor_temporal": 4})()

    class FakePipeline:
        parallel_context = FakeParallel()
        vae_scale_factor_temporal = 4
        vae = FakeVae()

    executor = WanVideoDecoderExecutor(
        FakePipeline(),
        EpeSchedulingModule(FakeParallel()),
        default_width=832,
        default_height=480,
        default_num_frames=17,
        default_num_steps=50,
    )
    profile = executor.request_profile({"prompt": "test"}, completed_steps=3)

    assert profile.image_tokens == 5 * 30 * 52
    assert profile.attributes["conditions"] == 2
    assert profile.attributes["num_frames"] == 17
    assert profile.attributes["state_bytes"] == 16 * 5 * 60 * 104 * 4
    assert executor._decode_kwargs() == {
        "parallel_vae": True,
        "vae_parallel_halo": 8,
    }


def test_wan_executor_rejects_negative_parallel_vae_halo() -> None:
    class FakeParallel:
        rank = 0
        local_rank = 0
        world_size = 1
        allowed_widths = (1,)

    class FakePipeline:
        parallel_context = FakeParallel()

    with pytest.raises(ValueError, match="halo must be non-negative"):
        WanVideoDecoderExecutor(
            FakePipeline(),
            EpeSchedulingModule(FakeParallel()),
            default_width=832,
            default_height=480,
            default_num_frames=17,
            default_num_steps=50,
            vae_parallel_halo=-1,
        )


def test_wan_executor_rejects_invalid_warmup_frame_profile() -> None:
    class FakeParallel:
        rank = 0
        local_rank = 0
        world_size = 1
        allowed_widths = (1,)

    class FakePipeline:
        parallel_context = FakeParallel()

    with pytest.raises(ValueError, match=r"4n\+1"):
        WanVideoDecoderExecutor(
            FakePipeline(),
            EpeSchedulingModule(FakeParallel()),
            default_width=832,
            default_height=480,
            default_num_frames=17,
            default_num_steps=50,
            warmup_profiles=((480, 832, 32),),
        )
