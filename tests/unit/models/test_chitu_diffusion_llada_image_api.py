from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from chitu_diffusion import LLaDAImageRequest
from chitu_diffusion.epe.api import DiffusersEPEPipeline
from chitu_diffusion.epe.scheduling.planner import EpeSchedulingModule
from chitu_diffusion.models.llada_image.api import LLaDAImagePipeline
from chitu_diffusion.models.llada_image.executor import LLaDAImageDecoderExecutor
from chitu_diffusion.models.llada_image.pipeline import LLaDAImageDiffusionPipeline
from chitu_diffusion.parallel.vae import VaeParallelPlacement


class FakeParallel:
    rank = 0
    local_rank = 0
    world_size = 1
    allowed_widths = (1,)


def make_executor() -> LLaDAImageDecoderExecutor:
    transformer = torch.nn.Linear(4, 4, bias=False)
    transformer.config = SimpleNamespace(in_channels=128)
    pipeline = SimpleNamespace(
        parallel_context=FakeParallel(),
        transformer=transformer,
    )
    return LLaDAImageDecoderExecutor(
        pipeline,
        EpeSchedulingModule(FakeParallel()),
        default_width=512,
        default_height=512,
        default_num_steps=20,
        vae_placement=VaeParallelPlacement(halo=4, sharded=False),
    )


class _PromptCacheHarness:
    _encode_default_negative_prompt = (
        LLaDAImageDiffusionPipeline._encode_default_negative_prompt
    )
    encode_prompt = LLaDAImageDiffusionPipeline.encode_prompt

    def __init__(self) -> None:
        self.text_projection = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self._default_negative_prompt_cache = None
        self.encoded_prompts: list[list[str | None]] = []

    def _encode_text(
        self, prompts: list[str | None], max_sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del max_sequence_length
        self.encoded_prompts.append(prompts)
        value = float(len(self.encoded_prompts))
        return (
            torch.full((len(prompts), 2, 3), value),
            torch.ones(len(prompts), 2, dtype=torch.bool),
        )


def test_llada_image_request_preserves_official_defaults() -> None:
    request = LLaDAImageRequest(prompt="test")

    assert request.generation_mode == "text"
    assert request.num_inference_steps == 20
    assert request.num_steps == 20
    assert request.guidance_scale == 4.5
    assert request.max_sequence_length == 2048

    with pytest.raises(ValueError, match="one prompt"):
        LLaDAImageRequest(prompt=["first", "second"])


def test_llada_image_request_validates_mode_specific_image_contract() -> None:
    image = Image.new("RGB", (32, 32))

    with pytest.raises(ValueError, match="image is required"):
        LLaDAImageRequest(prompt="edit", generation_mode="editing")
    with pytest.raises(ValueError, match="must be omitted"):
        LLaDAImageRequest(prompt="test", generation_mode="text", image=image)
    with pytest.raises(ValueError, match="multiples of 32"):
        LLaDAImageRequest(
            prompt="edit",
            generation_mode="editing",
            image=image,
            width=1008,
        )
    with pytest.raises(ValueError, match="one source image"):
        LLaDAImageRequest(
            prompt="edit",
            generation_mode="editing",
            image=[image],
        )


def test_llada_image_request_builds_official_diffusers_inputs() -> None:
    request = LLaDAImageRequest(
        prompt="test",
        generation_mode="vq",
        negative_prompt="bad",
        width=512,
        height=768,
        num_inference_steps=8,
        guidance_scale=5.0,
        seed=7,
        max_sequence_length=1024,
    )

    diffusion = request.to_diffusion_request(device=torch.device("cpu"))

    assert diffusion.inputs["generation_mode"] == "vq"
    assert diffusion.inputs["image"] is None
    assert diffusion.inputs["negative_prompt"] == "bad"
    assert diffusion.inputs["max_sequence_length"] == 1024
    assert diffusion.inputs["guidance_scale"] == 5.0
    assert diffusion.num_inference_steps == 8
    assert diffusion.inputs["generator"].initial_seed() == 7


def test_default_negative_prompt_condition_is_cached_in_inference() -> None:
    pipeline = _PromptCacheHarness()
    prompt_embeds = torch.zeros(1, 2, 3)
    prompt_mask = torch.ones(1, 2, dtype=torch.bool)

    with torch.inference_mode():
        first = pipeline.encode_prompt(
            None,
            None,
            True,
            1,
            prompt_embeds,
            prompt_mask,
            None,
            None,
            2048,
            torch.device("cpu"),
        )
        second = pipeline.encode_prompt(
            None,
            None,
            True,
            1,
            prompt_embeds,
            prompt_mask,
            None,
            None,
            2048,
            torch.device("cpu"),
        )

    assert pipeline.encoded_prompts == [[None]]
    torch.testing.assert_close(first[2], second[2])
    torch.testing.assert_close(first[3], second[3])
    assert pipeline._default_negative_prompt_cache is not None
    assert pipeline._default_negative_prompt_cache[1][0].device.type == "cpu"


def test_default_negative_prompt_cache_is_bounded_and_invalidated() -> None:
    pipeline = _PromptCacheHarness()
    prompt_embeds = torch.zeros(1, 2, 3)
    prompt_mask = torch.ones(1, 2, dtype=torch.bool)

    with torch.inference_mode():
        for max_sequence_length in (1024, 2048):
            pipeline.encode_prompt(
                None,
                None,
                True,
                1,
                prompt_embeds,
                prompt_mask,
                None,
                None,
                max_sequence_length,
                torch.device("cpu"),
            )

    assert pipeline.encoded_prompts == [[None], [None]]
    assert pipeline._default_negative_prompt_cache is not None
    assert pipeline._default_negative_prompt_cache[0][0] == 2048


def test_explicit_negative_prompt_bypasses_default_cache() -> None:
    pipeline = _PromptCacheHarness()

    with torch.inference_mode():
        pipeline.encode_prompt(
            None,
            "low quality",
            True,
            1,
            torch.zeros(1, 2, 3),
            torch.ones(1, 2, dtype=torch.bool),
            None,
            None,
            2048,
            torch.device("cpu"),
        )

    assert pipeline.encoded_prompts == [["low quality"]]


def test_llada_image_uses_shared_api_layer() -> None:
    assert issubclass(LLaDAImagePipeline, DiffusersEPEPipeline)


def test_llada_image_requires_local_checkpoint_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="local Diffusers checkpoint directory"):
        LLaDAImageDiffusionPipeline.from_pretrained(tmp_path / "missing")


def test_llada_image_executor_normalizes_alias_and_profiles_cfg() -> None:
    executor = make_executor()

    request = executor.normalize_request(
        {"prompt": "test", "num_steps": 8, "guidance_scale": 1.0}
    )
    profile = executor.request_profile(request, completed_steps=3)

    assert request.num_inference_steps == 8
    assert profile.total_steps == 8
    assert profile.image_tokens == 1024
    assert profile.attributes == {
        "batch_size": 1,
        "conditions": 1,
        "state_bytes": 0,
        "generation_mode": "text",
    }
    assert executor._decode_kwargs() == {
        "parallel_vae": False,
        "vae_parallel_halo": 4,
    }


def test_llada_image_edit_request_profile_marks_state_size_unknown() -> None:
    executor = make_executor()
    request = LLaDAImageRequest(
        prompt="edit",
        generation_mode="editing",
        image=Image.new("RGB", (32, 32)),
        width=512,
        height=512,
    )

    profile = executor.request_profile(request, completed_steps=0)

    assert profile.image_tokens == 2048
    assert profile.attributes["state_bytes"] == 0
    assert profile.attributes["generation_mode"] == "editing"


def test_llada_image_prepared_state_profile_uses_exact_tensor_bytes() -> None:
    executor = make_executor()
    latents = torch.zeros((1, 128, 32, 32), dtype=torch.float32)
    state = SimpleNamespace(
        timesteps=torch.arange(4),
        step_index=1,
        height=512,
        width=512,
        image_tokens=1024,
        latents=latents,
        guidance_scale=4.5,
        generation_mode="text",
    )

    profile = executor.profile(state)

    assert profile.completed_steps == 1
    assert profile.attributes["state_bytes"] == (
        latents.numel() * latents.element_size()
    )


def test_llada_image_executor_rejects_reserved_extra_inputs() -> None:
    executor = make_executor()

    with pytest.raises(ValueError, match="reserved fields: generation_mode"):
        executor.prepare_request(
            LLaDAImageRequest(
                prompt="test",
                extra_inputs={"generation_mode": "editing"},
            )
        )


def test_executor_warmup_merges_terminal_costs() -> None:
    executor = make_executor()
    executor.pipeline.warmup_epe = lambda **kwargs: {
        "rows": [
            {
                "image_tokens": 1024,
                "width": 1,
                "latency_ms": 10.0,
                "samples_ms": [10.0, 10.0, 10.0],
            }
        ]
    }
    terminal_calls = []

    def warmup_vae(**kwargs):
        terminal_calls.append(kwargs)
        return [
            {
                "image_tokens": 1024,
                "width": 1,
                "state_bytes": 4096,
                "terminal_ms": 4.0,
                "vae_latency_ms": 3.0,
                "d2h_latency_ms": 1.0,
                "terminal_samples_ms": [4.0, 4.0, 4.0],
            }
        ]

    executor.pipeline.warmup_vae = warmup_vae
    initialized_rows = []
    executor.scheduling_module.initialize_cost_model = initialized_rows.extend

    report = executor.warmup(resolutions=((512, 512),), steps=3)

    assert terminal_calls == [
        {
            "resolutions": ((512, 512),),
            "steps": 3,
            "parallel_vae": False,
            "vae_parallel_halo": 4,
        }
    ]
    assert report["rows"][0]["terminal_ms"] == 4.0
    assert report["rows"][0]["state_bytes"] == 4096
    assert initialized_rows == report["rows"]
