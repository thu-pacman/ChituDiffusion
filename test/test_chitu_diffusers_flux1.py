from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from chitu_diffusers import EpeRequest, EpeSchedulingModule, Flux1Request
from chitu_diffusers.epac.api import DiffusersEPACPipeline
from chitu_diffusers.models.flux1 import EpeFlux1Pipeline
from chitu_diffusers.models.flux1.api import Flux1EPACPipeline
from chitu_diffusers.models.flux1.executor import Flux1ImageDecoderExecutor
from chitu_diffusers.parallel import ImageContextParallelAttention


def test_flux1_request_requires_packed_latent_compatible_resolution() -> None:
    request = Flux1Request(prompt="test", width=512, height=768, num_steps=4)

    assert request.width == 512
    assert request.height == 768
    assert request.guidance_scale == 3.5

    with pytest.raises(ValueError, match="multiples of 16"):
        Flux1Request(prompt="test", width=510, height=512)
    with pytest.raises(ValueError, match="deadline_ms"):
        Flux1Request(prompt="test", deadline_ms=0)


def test_flux1_request_rejects_reserved_extra_inputs() -> None:
    request = Flux1Request(prompt="test", extra_inputs={"width": 1024})

    with pytest.raises(ValueError, match="reserved fields"):
        request.to_diffusion_request(device=torch.device("cpu"))


def test_joint_first_image_attention_matches_local_sdpa() -> None:
    generator = torch.Generator().manual_seed(17)
    joint = tuple(torch.randn(1, 3, 2, 4, generator=generator) for _ in range(3))
    image = tuple(torch.randn(1, 5, 2, 4, generator=generator) for _ in range(3))
    attention = ImageContextParallelAttention(mode="agkv")

    joint_output, image_output = attention(
        *image,
        *joint,
        lane_process_group=None,
        joint_first=True,
    )

    expected = F.scaled_dot_product_attention(
        torch.cat([joint[0], image[0]], dim=1).transpose(1, 2),
        torch.cat([joint[1], image[1]], dim=1).transpose(1, 2),
        torch.cat([joint[2], image[2]], dim=1).transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
    assert torch.equal(joint_output, expected[:, :3])
    assert torch.equal(image_output, expected[:, 3:])


def test_flux1_usp_fails_before_model_loading() -> None:
    with pytest.raises(NotImplementedError, match="agkv"):
        EpeFlux1Pipeline.from_pretrained("unused", attention_mode="usp")


def test_flux1_uses_shared_api_layer() -> None:
    assert issubclass(Flux1EPACPipeline, DiffusersEPACPipeline)


def test_shared_scheduling_module_rebuilds_for_runtime_policy_changes() -> None:
    class FakeParallel:
        rank = 0
        world_size = 4
        allowed_widths = (1, 2, 4)

    scheduling = EpeSchedulingModule(FakeParallel(), policy="static_cp")
    scheduling.initialize_cost_model(
        [
            {
                "image_tokens": 64,
                "width": width,
                "batch_size": 1,
                "cfg_conditions": 1,
                "latency_ms": float(10 / width),
            }
            for width in (1, 2, 4)
        ]
    )
    request = EpeRequest(
        request_id="shared",
        image_tokens=64,
        total_steps=4,
        current_step=0,
        submitted_at_ms=0.0,
        cfg_conditions=1,
    )

    assert scheduling.plan_phase([request], now_ms=0.0)[0].ranks == (0, 1, 2, 3)
    scheduling.policy = "static_dp"
    assert scheduling.plan_phase([request], now_ms=0.0)[0].ranks == (0,)


def test_flux1_executor_reuses_shared_request_profile_lifecycle() -> None:
    class FakeParallel:
        rank = 0
        local_rank = 0
        world_size = 1
        allowed_widths = (1,)

    class FakePipeline:
        parallel_context = FakeParallel()
        transformer = torch.nn.Linear(4, 4, bias=False)

    parallel = FakePipeline.parallel_context
    executor = Flux1ImageDecoderExecutor(
        FakePipeline(),
        EpeSchedulingModule(parallel),
        default_width=512,
        default_height=512,
        default_num_steps=4,
        parallel_vae=False,
        vae_parallel_halo=4,
    )

    request = executor.normalize_request({"prompt": "test"})
    profile = executor.request_profile(request, completed_steps=1)

    assert (request.width, request.height, request.num_steps) == (512, 512, 4)
    assert profile.shape_key == (512, 512)
    assert profile.completed_steps == 1
    assert profile.image_tokens == 1024
    assert profile.attributes == {
        "batch_size": 1,
        "conditions": 1,
        "state_bytes": 1024 * 64 * 4,
    }
    assert executor._decode_kwargs() == {
        "parallel_vae": False,
        "vae_parallel_halo": 4,
    }


def test_flux1_parallel_vae_uses_decoder_scale_not_packing_scale() -> None:
    pipeline = SimpleNamespace(
        vae=SimpleNamespace(
            config=SimpleNamespace(block_out_channels=(128, 256, 512, 512))
        )
    )

    assert EpeFlux1Pipeline.vae_spatial_scale_factor.fget(pipeline) == 8


def test_flux1_executor_rejects_negative_vae_halo() -> None:
    class FakeParallel:
        rank = 0
        local_rank = 0
        world_size = 1
        allowed_widths = (1,)

    class FakePipeline:
        parallel_context = FakeParallel()

    with pytest.raises(ValueError, match="vae_parallel_halo"):
        Flux1ImageDecoderExecutor(
            FakePipeline(),
            EpeSchedulingModule(FakeParallel()),
            default_width=512,
            default_height=512,
            default_num_steps=4,
            vae_parallel_halo=-1,
        )
