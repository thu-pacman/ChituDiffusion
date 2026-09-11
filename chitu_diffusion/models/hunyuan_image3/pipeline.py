"""Resumable Hunyuan Image 3 denoising and lane-parallel VAE decode.

The released pipeline runs its whole diffusion loop inside one call. The runtime
needs to own that loop, so it is split here into request preparation, a single
step, and a terminal decode. Prompt tokenization, rotary tables, conditioning
image encode, and input assembly stay on the released rank-synchronous path;
only the transformer steps and the VAE decode are parallel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ...epe.contracts import TerminalArtifact
from ...parallel.vae import parallel_vae_decode
from .cache import HunyuanImage3Cache
from .loader import LoadedHunyuanImage3
from .parallel import HunyuanImage3ParallelRuntime

# The released prompt is always tokenized as one conditional and one
# unconditional half, whether or not the two are split across ranks.
CFG_CONDITIONS = 2


def flow_match_sigmas(
    num_inference_steps: int,
    *,
    shift: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Reproduce the released discrete flow-match schedule as plain data.

    Holding the schedule in the request state instead of a stateful scheduler
    keeps every step a pure function of the state, which is what lets the
    runtime pause, resume, and interleave requests.
    """

    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    sigmas = torch.linspace(1, 0, num_inference_steps + 1, device=device)
    if shift != 1.0:
        sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    return sigmas.to(torch.float32)


@dataclass
class HunyuanImage3DenoiseState:
    """Everything needed to resume a Hunyuan Image 3 request after any step."""

    request_id: str
    latents: torch.Tensor
    input_ids: torch.Tensor
    model_kwargs: dict[str, Any]
    sigmas: torch.Tensor
    height: int
    width: int
    image_tokens: int
    guidance_scale: float
    cfg_conditions: int = CFG_CONDITIONS
    output_type: str = "pil"
    step_index: int = 0
    generator: list[torch.Generator] | None = field(default=None, repr=False)

    @property
    def total_steps(self) -> int:
        return int(self.sigmas.numel()) - 1

    @property
    def timesteps(self) -> torch.Tensor:
        return self.sigmas[:-1] * 1000.0

    @property
    def complete(self) -> bool:
        return self.step_index >= self.total_steps

    @property
    def actual_batch_size(self) -> int:
        return int(self.latents.shape[0])

    @property
    def conditions(self) -> int:
        return self.cfg_conditions


def _select_cfg_branch(
    value: Any, branch: int, conditions: int = CFG_CONDITIONS
) -> Any:
    """Keep one CFG batch entry while preserving the released container shape."""

    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == conditions:
            return value.narrow(0, branch, 1).contiguous()
        return value
    if isinstance(value, dict):
        return {
            key: _select_cfg_branch(item, branch, conditions)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) == conditions:
            return [_select_cfg_branch(value[branch], branch, conditions)]
        return [_select_cfg_branch(item, branch, conditions) for item in value]
    if isinstance(value, tuple):
        # Tuples carry structures such as (cos, sin), not a batch container.
        return tuple(_select_cfg_branch(item, branch, conditions) for item in value)
    return value


class HunyuanImage3Pipeline:
    """Step-addressable denoising over the configured TPxCFGxCPxEP stage."""

    def __init__(
        self,
        loaded: LoadedHunyuanImage3,
        runtime: HunyuanImage3ParallelRuntime,
        *,
        flow_shift: float | None = None,
    ) -> None:
        self.loaded = loaded
        self.runtime = runtime
        self.model = loaded.model
        self.vae_placement = runtime.vae_placement
        self.flow_shift = float(
            flow_shift
            if flow_shift is not None
            else self.model.generation_config.flow_shift
        )
        # Layer-wise spatial parallelism requires the full-image decoder semantics.
        self.model.vae.disable_tiling()
        self.model.vae.disable_slicing()
        downsample = tuple(int(value) for value in loaded.config.vae_downsample_factor)
        if len(downsample) != 2 or downsample[0] != downsample[1]:
            raise ValueError(
                "the parallel VAE decode requires a square downsample factor, got "
                f"{downsample}"
            )
        self.vae_spatial_scale = downsample[0]

    @property
    def parallel_context(self) -> Any:
        return self.runtime.context

    @property
    def vae_topology(self) -> Any:
        """Return this rank's decode topology, or None when it sits out.

        The plane lane stays active for the whole request, so a lane-scoped
        placement resolves against it and a stage-scoped one ignores it.
        """

        return self.vae_placement.resolve(self.parallel_context.active)

    @property
    def device(self) -> torch.device:
        return self.loaded.device

    @property
    def default_num_steps(self) -> int:
        return int(self.model.generation_config.diff_infer_steps)

    @property
    def default_guidance_scale(self) -> float:
        return float(self.model.generation_config.diff_guidance_scale)

    def resolve_image_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Snap a requested size onto the released resolution group."""

        info = self.model.image_processor.build_image_info(
            (int(image_size[0]), int(image_size[1]))
        )
        return int(info.image_height), int(info.image_width)

    def _build_message_list(
        self,
        *,
        prompt: str,
        image_size: tuple[int, int],
        condition_images: tuple[Any, ...],
        system_prompt: str | None,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Assemble the released single-turn message list for image generation.

        ``prepare_model_inputs`` only derives conditioning images from a message
        list, so both text-to-image and image-to-image go through this one path.
        """

        processor = self.model.image_processor
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "type": "text",
                    "content": system_prompt,
                    "context_type": "str",
                }
            )
        for image in condition_images:
            messages.append(
                {
                    "role": "user",
                    "type": "joint_image",
                    "content": processor.preprocess(image),
                    "context_type": "image_info",
                }
            )
        messages.append(
            {
                "role": "user",
                "type": "text",
                "content": prompt,
                "context_type": "str",
            }
        )
        gen_image_info = processor.build_image_info(
            (int(image_size[0]), int(image_size[1]))
        )
        messages.append(
            {
                "role": "assistant",
                "type": "gen_image",
                "content": gen_image_info,
                "context_type": "image_info",
            }
        )
        return messages, gen_image_info

    @torch.inference_mode()
    def prepare_request(
        self,
        *,
        request_id: str,
        prompt: str,
        image_size: tuple[int, int],
        num_steps: int,
        guidance_scale: float,
        seed: int,
        condition_images: tuple[Any, ...] = (),
        system_prompt: str | None = None,
        output_type: str = "pil",
    ) -> HunyuanImage3DenoiseState:
        """Run the released synchronous preprocessing and seed the latents."""

        if guidance_scale <= 1.0:
            # The released prompt is always tokenized with a conditional and an
            # unconditional half for image generation, so the guided branch is
            # the only one whose batch matches the sequence.
            raise ValueError("Hunyuan Image 3 requires guidance_scale > 1")
        messages, gen_image_info = self._build_message_list(
            prompt=prompt,
            image_size=image_size,
            condition_images=condition_images,
            system_prompt=system_prompt,
        )
        model_kwargs = dict(
            self.model.prepare_model_inputs(
                message_list=messages,
                mode="gen_image",
                bot_task="image",
                seed=seed,
                device=self.device,
            )
        )
        input_ids = model_kwargs.pop("input_ids")
        generator = model_kwargs.pop("generator", None)
        model_kwargs.pop("eos_token_id", None)
        model_kwargs.pop("max_new_tokens", None)
        attention_mask = self.model._prepare_attention_mask_for_generation(
            input_ids, self.model.generation_config, model_kwargs=model_kwargs
        )
        model_kwargs["attention_mask"] = attention_mask.to(self.device)
        if not self.runtime.runs_both_cfg_branches:
            branch = self.runtime.cfg_branch_index
            input_ids = _select_cfg_branch(input_ids, branch)
            model_kwargs = _select_cfg_branch(model_kwargs, branch)
        model_kwargs["past_key_values"] = HunyuanImage3Cache(
            num_layers=self.loaded.num_hidden_layers,
            max_cache_len=int(input_ids.shape[1]),
        )

        height = int(gen_image_info.image_height)
        width = int(gen_image_info.image_width)
        latents = torch.randn(
            (
                1,
                int(self.loaded.config.vae["latent_channels"]),
                height // self.vae_spatial_scale,
                width // self.vae_spatial_scale,
            ),
            generator=generator[0] if generator else None,
            device=self.device,
            dtype=torch.bfloat16,
        )
        return HunyuanImage3DenoiseState(
            request_id=request_id,
            latents=latents,
            input_ids=input_ids,
            model_kwargs=model_kwargs,
            sigmas=flow_match_sigmas(
                num_steps, shift=self.flow_shift, device=self.device
            ),
            height=height,
            width=width,
            image_tokens=int(gen_image_info.token_height)
            * int(gen_image_info.token_width),
            guidance_scale=float(guidance_scale),
            output_type=output_type,
            generator=generator,
        )

    @torch.inference_mode()
    def denoise_step(
        self,
        state: HunyuanImage3DenoiseState,
        *,
        lane_ranks: tuple[int, ...],
    ) -> None:
        """Advance one Euler flow-match step across the whole stage."""

        del lane_ranks
        if state.complete:
            raise RuntimeError("request has no remaining denoising steps")
        index = state.step_index
        timestep = state.timesteps[index]
        # Under CFG2 the branches occupy separate context-parallel lanes, so a
        # rank runs one batch entry and exchanges only the final prediction with
        # its shard-aligned partner. Under CFG1 both entries stay local and the
        # latents are the same for each of them.
        latent_input = state.latents
        if self.runtime.runs_both_cfg_branches:
            latent_input = latent_input.repeat(
                CFG_CONDITIONS, *([1] * (latent_input.ndim - 1))
            )
        model_inputs = self.model.prepare_inputs_for_generation(
            state.input_ids,
            images=latent_input,
            timestep=timestep.repeat(latent_input.shape[0]),
            **state.model_kwargs,
        )
        with self.runtime.context.activate(self.runtime.context_parallel_ranks):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                output = self.model(**model_inputs, first_step=index == 0)
        predicted = output["diffusion_prediction"].to(torch.float32)
        if self.runtime.runs_both_cfg_branches:
            conditional, unconditional = predicted[:1], predicted[1:]
        else:
            conditional, unconditional = self.runtime.context.all_gather_cfg(
                predicted, self.runtime.cfg
            )
        prediction = unconditional + state.guidance_scale * (
            conditional - unconditional
        )

        sigma = state.sigmas[index]
        sigma_next = state.sigmas[index + 1]
        # The released Euler step keeps the running latents in float32 after the
        # bfloat16 initial noise, and the patch embedding casts them back.
        state.latents = state.latents.to(torch.float32) + prediction * (
            sigma_next - sigma
        )
        state.step_index = index + 1
        if not state.complete:
            state.model_kwargs = self.model._update_model_kwargs_for_generation(
                output, state.model_kwargs
            )
            position_ids = state.model_kwargs["position_ids"]
            if state.input_ids.shape[1] != position_ids.shape[1]:
                state.input_ids = torch.gather(state.input_ids, 1, index=position_ids)

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            return self.model.vae.decode(latents, return_dict=False)[0]

    @torch.inference_mode()
    def decode_request(
        self,
        state: HunyuanImage3DenoiseState,
        *,
        timings: dict[str, object] | None = None,
    ) -> TerminalArtifact | None:
        """Decode the finished latents over the stage's VAE placement."""

        topology = self.vae_topology
        if topology is None:
            return None
        started = time.perf_counter()
        vae_config = self.model.vae.config
        latents = state.latents.to(torch.float32)
        scaling_factor = getattr(vae_config, "scaling_factor", None)
        if scaling_factor:
            latents = latents / scaling_factor
        shift_factor = getattr(vae_config, "shift_factor", None)
        if shift_factor:
            latents = latents + shift_factor
        # The released 3D VAE consumes [B, C, T, H, W] and returns the single
        # decoded frame for a one-frame latent.
        latents = latents.unsqueeze(2)

        image = parallel_vae_decode(
            self.model.vae,
            latents,
            decode_fn=self._decode_latents,
            topology=topology,
            enabled=self.vae_placement.sharded,
        )
        if timings is not None:
            timings["vae_decode_ms"] = (time.perf_counter() - started) * 1000.0
            timings["vae_end_unix_ns"] = time.time_ns()
        if image is None:
            return None
        if image.shape[2] != 1:
            raise RuntimeError(
                f"the image VAE returned {image.shape[2]} frames, expected one"
            )
        return TerminalArtifact(
            tensors={"image": image.squeeze(2)},
            metadata={
                "output_type": state.output_type,
                "height": state.height,
                "width": state.width,
            },
        )

    def synchronize_state(
        self, state: HunyuanImage3DenoiseState, *, step_index: int
    ) -> None:
        if int(step_index) != state.step_index:
            raise RuntimeError(
                "Hunyuan Image 3 cannot rewind a request: the key/value cache "
                f"holds step {state.step_index}, the plan asked for {step_index}"
            )

    def maybe_free_model_hooks(self) -> None:
        return None

    def close(self) -> None:
        self.runtime.context.close()


__all__ = [
    "CFG_CONDITIONS",
    "HunyuanImage3DenoiseState",
    "HunyuanImage3Pipeline",
    "flow_match_sigmas",
]
