from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from chitu_diffusion.models.llada_image.runtime import LLaDAImageRuntimeMixin
from chitu_diffusion.parallel.cp import EpeParallelContext


class _FakeTransformer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.config = SimpleNamespace(in_channels=2)
        self.epe = None
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        glm_cap_feats: list[torch.Tensor] | None,
        source_latents: list[torch.Tensor] | None,
        glm_cap_lengths: list[int] | None = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "t": t.clone(),
                "cap_feats": cap_feats,
                "glm_cap_feats": glm_cap_feats,
                "source_latents": source_latents,
                "glm_cap_lengths": glm_cap_lengths,
            }
        )
        return SimpleNamespace(
            sample=[
                torch.full_like(latent, float(features[0, 0]))
                for latent, features in zip(x, cap_feats, strict=True)
            ]
        )


class _FakeSigVQ:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.token_ids: torch.Tensor | None = None

    def __call__(self, *, token_ids: torch.Tensor) -> SimpleNamespace:
        self.token_ids = token_ids.clone()
        return SimpleNamespace(
            semantic_features=torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        )


class _FakeVAE:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._parameter = torch.nn.Parameter(torch.zeros(()))
        self.bn = SimpleNamespace(
            running_mean=torch.tensor([10.0, 20.0]),
            running_var=torch.tensor([4.0, 9.0]),
        )
        self.config = SimpleNamespace(batch_norm_eps=0.0)
        self.decode_input: torch.Tensor | None = None

    def parameters(self):
        yield self._parameter

    def decode(self, value: torch.Tensor, *, return_dict: bool) -> tuple[torch.Tensor]:
        assert return_dict is False
        self.decode_input = value.clone()
        return (value * 2.0,)


class _RuntimeHarness(LLaDAImageRuntimeMixin):
    def __init__(self) -> None:
        self.transformer = _FakeTransformer()
        self.text_encoder = SimpleNamespace(device=torch.device("cpu"))
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.sigvq = _FakeSigVQ()
        self.vae = _FakeVAE()
        self.latent_scale_factor = 2
        self.vae_scale_factor = 2
        self.image_processor = SimpleNamespace()
        self.generated_vq_tokens: torch.Tensor | None = None
        self.encoded_source_image: Any | None = None
        self.unpatchify_input: torch.Tensor | None = None

    def check_inputs(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def encode_prompt(
        self,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        do_cfg: bool,
        num_images_per_prompt: int,
        prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        max_sequence_length: int,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        del negative_prompt, num_images_per_prompt, max_sequence_length, device
        batch_size = 1 if isinstance(prompt, str) else len(prompt or [])
        if prompt_embeds is None:
            prompt_embeds = torch.full((batch_size, 3, 4), 3.0)
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.tensor([[True, False, True]] * batch_size)
        if do_cfg:
            if negative_prompt_embeds is None:
                negative_prompt_embeds = torch.ones(batch_size, 2, 4)
            if negative_prompt_attention_mask is None:
                negative_prompt_attention_mask = torch.ones(
                    batch_size, 2, dtype=torch.bool
                )
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def generate_vq_tokens(
        self, prompt: str | list[str], height: int, width: int
    ) -> torch.Tensor:
        del prompt, height, width
        self.generated_vq_tokens = torch.tensor([[11, 12]], dtype=torch.long)
        return self.generated_vq_tokens

    def _encode_source_image(
        self,
        image: Any,
        height: int,
        width: int,
        batch_size: int,
        num_images_per_prompt: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del height, width, batch_size, num_images_per_prompt
        self.encoded_source_image = image
        source_latents = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
        semantic_features = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        return source_latents, semantic_features

    def _unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self.unpatchify_input = latents.clone()
        return latents + 100.0

    def maybe_free_model_hooks(self) -> None:
        pass


def test_prepare_sigmas_uses_checkpoint_uniform_schedule() -> None:
    runtime = _RuntimeHarness()
    runtime.scheduler.register_to_config(use_uniform_sigmas=True)

    assert runtime._prepare_sigmas(4) == [1.0, 0.75, 0.5, 0.25]


class _CaptureScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self._step_index = 0
        self.model_output: torch.Tensor | None = None
        self.timestep: torch.Tensor | None = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        assert return_dict is False
        self.model_output = model_output.clone()
        self.timestep = timestep.clone()
        self._step_index += 1
        return (sample + model_output,)


def _prepare(
    runtime: _RuntimeHarness,
    *,
    generation_mode: str = "text",
    guidance_scale: float = 1.0,
    num_inference_steps: int = 2,
    image: Any | None = None,
):
    return runtime.prepare_request(
        prompt="a prompt",
        image=image,
        generation_mode=generation_mode,
        height=8,
        width=8,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        latents=torch.zeros(1, 2, 4, 4),
    )


def test_random_latents_remain_fp32_for_bfloat16_transformer() -> None:
    runtime = _RuntimeHarness()
    runtime.transformer.dtype = torch.bfloat16

    latents = runtime._prepare_random_latents(
        (1, 2, 4, 4),
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
    )

    assert latents.dtype == torch.float32


@pytest.mark.parametrize(
    ("prompt", "num_images_per_prompt"),
    [(["first", "second"], 1), ("first", 2)],
)
def test_prepare_request_rejects_effective_batch_larger_than_one(
    prompt: str | list[str], num_images_per_prompt: int
) -> None:
    runtime = _RuntimeHarness()

    with pytest.raises(ValueError, match="exactly one image per request"):
        runtime.prepare_request(
            prompt=prompt,
            height=8,
            width=8,
            num_images_per_prompt=num_images_per_prompt,
        )


def test_prepare_request_uses_official_kumaraswamy_schedule() -> None:
    runtime = _RuntimeHarness()

    state = _prepare(runtime, num_inference_steps=4)

    schedule = torch.linspace(0.001, 1.0, 5, dtype=torch.float64)[:-1]
    schedule = (1 - (1 - schedule**1.17) ** 0.8) ** 1.1
    expected_sigmas = (1 - schedule).float()
    torch.testing.assert_close(state.scheduler.sigmas[:-1], expected_sigmas)
    torch.testing.assert_close(
        state.timesteps, expected_sigmas * state.scheduler.config.num_train_timesteps
    )


def test_vq_generation_uses_request_seed_without_mutating_global_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RuntimeHarness()

    def sample_tokens(prompt: str | list[str], height: int, width: int) -> torch.Tensor:
        del prompt, height, width
        return torch.randint(0, 1024, (1, 32))

    monkeypatch.setattr(runtime, "generate_vq_tokens", sample_tokens)
    torch.manual_seed(777)
    rng_state = torch.random.get_rng_state()

    first = runtime._generate_vq_tokens_synchronized(
        "prompt",
        8,
        8,
        torch.Generator().manual_seed(42),
    )
    second = runtime._generate_vq_tokens_synchronized(
        "prompt",
        8,
        8,
        torch.Generator().manual_seed(42),
    )
    third = runtime._generate_vq_tokens_synchronized(
        "prompt",
        8,
        8,
        torch.Generator().manual_seed(43),
    )

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


@pytest.mark.parametrize("generation_mode", ["text", "vq", "editing"])
def test_prepare_and_step_pass_mode_specific_conditions(
    generation_mode: str,
) -> None:
    runtime = _RuntimeHarness()
    source_image = object() if generation_mode == "editing" else None

    state = _prepare(
        runtime,
        generation_mode=generation_mode,
        num_inference_steps=1,
        image=source_image,
    )
    runtime.denoise_step(state)

    call = runtime.transformer.calls[0]
    assert call["cap_feats"][0].shape == (2, 4)
    assert state.image_tokens == (32 if generation_mode == "editing" else 16)
    if generation_mode == "text":
        assert call["glm_cap_feats"] is None
        assert call["source_latents"] is None
    else:
        assert call["glm_cap_feats"][0].shape == (2, 4)
        if generation_mode == "vq":
            assert call["source_latents"] is None
            torch.testing.assert_close(
                runtime.sigvq.token_ids, runtime.generated_vq_tokens
            )
        else:
            assert runtime.encoded_source_image is source_image
            assert call["source_latents"][0].shape == (2, 1, 4, 4)


def test_unconditional_edit_branch_pads_empty_sigvq_without_changing_logical_length() -> (
    None
):
    runtime = _RuntimeHarness()
    state = _prepare(
        runtime,
        generation_mode="editing",
        guidance_scale=4.5,
        num_inference_steps=1,
        image=object(),
    )
    assert state.uncond_cap_feats is not None
    assert state.uncond_glm_cap_feats is not None
    assert state.cond_glm_cap_feats is not None
    assert state.uncond_glm_cap_feats[0].shape == (0, 4)

    runtime._transformer_branch(
        state,
        timestep=torch.tensor([0.5]),
        cap_feats=state.uncond_cap_feats,
        glm_cap_feats=state.uncond_glm_cap_feats,
    )

    call = runtime.transformer.calls[-1]
    assert call["glm_cap_feats"][0].shape == state.cond_glm_cap_feats[0].shape
    assert torch.count_nonzero(call["glm_cap_feats"][0]) == 0
    assert call["glm_cap_lengths"] == [0]
    assert state.uncond_glm_cap_feats[0].shape == (0, 4)


def test_denoise_step_applies_cfg_formula_and_advances_state() -> None:
    runtime = _RuntimeHarness()
    state = _prepare(runtime, guidance_scale=2.0, num_inference_steps=1)
    scheduler = _CaptureScheduler()
    state.scheduler = scheduler
    state.timesteps = torch.tensor([500.0])

    returned = runtime.denoise_step(state)

    assert returned is state
    assert len(runtime.transformer.calls) == 1
    torch.testing.assert_close(
        scheduler.model_output, torch.full_like(state.latents, -5.0)
    )
    torch.testing.assert_close(state.latents, torch.full_like(state.latents, -5.0))
    assert state.step_index == 1
    assert scheduler._step_index == 1
    assert state.complete


def test_denoise_step_advances_real_scheduler_once() -> None:
    runtime = _RuntimeHarness()
    state = _prepare(runtime, num_inference_steps=2)
    before = state.latents.clone()

    runtime.denoise_step(state)

    assert state.step_index == 1
    assert state.scheduler._step_index == 1
    assert not state.complete
    assert not torch.equal(state.latents, before)


def test_decode_request_denormalizes_and_unpatchifies_before_vae() -> None:
    runtime = _RuntimeHarness()
    state = _prepare(runtime, num_inference_steps=1)
    state.step_index = 1
    state.latents = torch.tensor([[[[1.0]], [[2.0]]]])

    images = runtime.decode_request(state)

    expected_denormalized = torch.tensor([[[[12.0]], [[26.0]]]])
    torch.testing.assert_close(runtime.unpatchify_input, expected_denormalized)
    torch.testing.assert_close(runtime.vae.decode_input, expected_denormalized + 100.0)
    torch.testing.assert_close(images, (expected_denormalized + 100.0) * 2.0)


def test_decode_and_finalize_reject_incomplete_state() -> None:
    runtime = _RuntimeHarness()
    state = _prepare(runtime, num_inference_steps=2)

    with pytest.raises(RuntimeError, match="incomplete denoise state"):
        runtime.decode_request(state)
    with pytest.raises(RuntimeError, match="incomplete denoise state"):
        runtime.finalize_request(state)


def test_warmup_vae_reports_terminal_and_state_transfer_costs() -> None:
    runtime = _RuntimeHarness()
    parallel = EpeParallelContext.from_torchrun(allowed_widths=(1,))
    runtime.transformer.epe = SimpleNamespace(parallel=parallel)
    try:
        rows = runtime.warmup_vae(
            resolutions=((8, 8),),
            steps=3,
            parallel_vae=False,
            vae_parallel_halo=0,
        )
    finally:
        parallel.close()

    assert len(rows) == 1
    assert rows[0]["image_tokens"] == 32
    assert rows[0]["state_bytes"] == 128
    assert rows[0]["terminal_ms"] >= rows[0]["vae_latency_ms"]
    assert rows[0]["d2h_latency_ms"] >= 0
    assert len(rows[0]["terminal_samples_ms"]) == 3


def test_synchronize_state_updates_runtime_and_scheduler_indices() -> None:
    runtime = _RuntimeHarness()
    state = _prepare(runtime, num_inference_steps=4)

    runtime.synchronize_state(state, step_index=3)

    assert state.step_index == 3
    assert state.scheduler._step_index == 3
