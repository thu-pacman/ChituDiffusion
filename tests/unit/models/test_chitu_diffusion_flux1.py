from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from chitu_diffusion.epe.api import DiffusersEPEPipeline
from chitu_diffusion.epe.scheduling.planner import EpeRequest, EpeSchedulingModule
from chitu_diffusion.models.flux1 import EpeFlux1Pipeline
from chitu_diffusion.models.flux1.api import Flux1Pipeline, Flux1Request
from chitu_diffusion.models.flux1.executor import Flux1ImageDecoderExecutor
from chitu_diffusion.parallel.cp import ImageContextParallelAttention
from chitu_diffusion.parallel.vae import VaeParallelPlacement


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


def test_flux1_uses_shared_api_layer() -> None:
    assert issubclass(Flux1Pipeline, DiffusersEPEPipeline)


def test_flux1_finalize_skips_postprocess_on_nonleader() -> None:
    pipeline = SimpleNamespace(
        decode_request=lambda state: None,
        image_processor=SimpleNamespace(
            postprocess=lambda *args, **kwargs: pytest.fail(
                "nonleader output must not be postprocessed"
            )
        ),
        maybe_free_model_hooks=lambda: None,
    )
    state = SimpleNamespace(latents=torch.empty(1))

    output = EpeFlux1Pipeline.finalize_request(pipeline, state)

    assert output.images is None


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
        vae_placement=VaeParallelPlacement(halo=4, sharded=False),
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
    assert (executor.parallel_vae, executor.vae_parallel_halo) == (False, 4)


def test_flux1_parallel_vae_uses_decoder_scale_not_packing_scale() -> None:
    pipeline = SimpleNamespace(
        vae=SimpleNamespace(
            config=SimpleNamespace(block_out_channels=(128, 256, 512, 512))
        )
    )

    assert EpeFlux1Pipeline.vae_spatial_scale_factor.fget(pipeline) == 8


def test_the_shared_vae_plan_rejects_a_negative_halo() -> None:
    from chitu_diffusion.epe.contracts import VaeParallelPlan

    with pytest.raises(ValueError, match="halo must be non-negative"):
        VaeParallelPlan(halo=-1)
