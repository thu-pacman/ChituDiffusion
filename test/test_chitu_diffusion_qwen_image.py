from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from chitu_diffusion import QwenImageRequest
from chitu_diffusion.epac.api import DiffusersEPACPipeline
from chitu_diffusion.epac.model_scheduling import EpeSchedulingModule
from chitu_diffusion.models.qwen_image.api import QwenImageEPACPipeline
from chitu_diffusion.models.qwen_image.executor import QwenImageDecoderExecutor
from chitu_diffusion.models.qwen_image.pipeline import (
    EpeQwenImagePipeline,
    combine_qwen_cfg_predictions,
)


def test_qwen_image_request_defaults_to_true_cfg() -> None:
    request = QwenImageRequest(prompt="test", width=512, height=768, num_steps=50)

    assert request.negative_prompt == " "
    assert request.true_cfg_scale == 4.0
    assert request.guidance_scale is None

    with pytest.raises(ValueError, match="multiples of 16"):
        QwenImageRequest(prompt="test", width=510, height=512)


def test_qwen_image_request_builds_diffusion_inputs() -> None:
    request = QwenImageRequest(
        prompt="test",
        negative_prompt="bad",
        width=512,
        height=512,
        num_steps=4,
        true_cfg_scale=3.0,
        seed=7,
    )

    diffusion = request.to_diffusion_request(device=torch.device("cpu"))

    assert diffusion.inputs["negative_prompt"] == "bad"
    assert diffusion.inputs["true_cfg_scale"] == 3.0
    assert diffusion.inputs["guidance_scale"] is None
    assert diffusion.num_inference_steps == 4


def test_qwen_cfg_combination_matches_native_formula() -> None:
    positive = torch.tensor([[[3.0, 4.0]]])
    negative = torch.tensor([[[1.0, 2.0]]])

    output = combine_qwen_cfg_predictions(positive, negative, true_cfg_scale=4.0)
    combined = negative + 4.0 * (positive - negative)
    expected = combined * (
        torch.norm(positive, dim=-1, keepdim=True)
        / torch.norm(combined, dim=-1, keepdim=True)
    )

    assert torch.equal(output, expected)


def test_qwen_image_uses_shared_api_layer() -> None:
    assert issubclass(QwenImageEPACPipeline, DiffusersEPACPipeline)


def test_qwen_image_finalize_skips_postprocess_on_nonleader() -> None:
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

    output = EpeQwenImagePipeline.finalize_request(pipeline, state)

    assert output.images is None


def test_qwen_image_executor_profiles_cfg_and_packed_state() -> None:
    class FakeParallel:
        rank = 0
        local_rank = 0
        world_size = 1
        allowed_widths = (1,)

    class FakePipeline:
        parallel_context = FakeParallel()
        transformer = torch.nn.Linear(4, 4, bias=False)

    executor = QwenImageDecoderExecutor(
        FakePipeline(),
        EpeSchedulingModule(FakeParallel()),
        default_width=512,
        default_height=512,
        default_num_steps=50,
        parallel_vae=False,
        vae_parallel_halo=4,
    )
    request = executor.normalize_request({"prompt": "test", "guidance_scale": 2.5})
    profile = executor.request_profile(request, completed_steps=3)

    assert profile.image_tokens == 1024
    assert profile.attributes == {
        "batch_size": 1,
        "conditions": 2,
        "state_bytes": 1024 * 64 * 4,
    }
    assert executor._decode_kwargs() == {
        "parallel_vae": False,
        "vae_parallel_halo": 4,
    }
    assert request.true_cfg_scale == 2.5
