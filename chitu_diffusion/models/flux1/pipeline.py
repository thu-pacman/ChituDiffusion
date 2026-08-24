from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from ...parallel import (
    EpeParallelContext,
    parallel_tiled_vae_decode,
    resolve_context_parallel_config,
)
from .transformer import EpeFlux1Transformer2DModel


@dataclass
class Flux1DenoiseState:
    """Request-owned Flux.1 state with a private scheduler cursor."""

    latents: torch.Tensor
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor
    latent_image_ids: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    guidance: torch.Tensor | None
    actual_batch_size: int
    width: int
    height: int
    image_tokens: int
    step_index: int = 0

    @property
    def complete(self) -> bool:
        return self.step_index >= len(self.timesteps)


class EpeFlux1Pipeline(FluxPipeline):
    """FluxPipeline split into request preparation, one step, and finalize."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs: Any):
        parallel = kwargs.pop("parallel_context", None)
        allowed_widths = kwargs.pop("allowed_lane_widths", None)
        ulysses_transport = kwargs.pop("ulysses_transport", None)
        agkv_transport = kwargs.pop("agkv_transport", None)
        attention_mode, ulysses_degree = resolve_context_parallel_config(
            kwargs.pop("attention_mode", "agkv"),
            kwargs.pop("ulysses_degree", None),
        )
        if parallel is None:
            parallel = EpeParallelContext.from_torchrun(
                allowed_widths=(
                    tuple(int(width) for width in allowed_widths)
                    if allowed_widths is not None
                    else None
                ),
                ulysses_degree=ulysses_degree,
                ulysses_transport=ulysses_transport,
                agkv_transport=agkv_transport,
            )
        transformer = kwargs.pop("transformer", None)
        if transformer is None:
            transformer_kwargs: dict[str, Any] = {
                "subfolder": "transformer",
                "parallel_context": parallel,
                "attention_mode": attention_mode,
            }
            for key in ("torch_dtype", "local_files_only", "variant", "revision"):
                if key in kwargs:
                    transformer_kwargs[key] = kwargs[key]
            transformer = EpeFlux1Transformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                **transformer_kwargs,
            )
        elif not isinstance(transformer, EpeFlux1Transformer2DModel):
            raise TypeError("transformer must be an EpeFlux1Transformer2DModel")
        else:
            transformer.configure_epe(parallel, attention_mode=attention_mode)
        pipeline = super().from_pretrained(
            pretrained_model_name_or_path,
            transformer=transformer,
            **kwargs,
        )
        pipeline._epe_parallel_context = parallel
        pipeline._epe_attention_mode = attention_mode
        pipeline._epe_ulysses_degree = ulysses_degree
        return pipeline

    @property
    def parallel_context(self) -> EpeParallelContext:
        parallel = getattr(self, "_epe_parallel_context", None)
        if parallel is None:
            raise RuntimeError("pipeline has no EPE parallel context")
        return parallel

    @property
    def vae_spatial_scale_factor(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    @torch.inference_mode()
    def warmup_epe(
        self,
        *,
        resolutions: tuple[tuple[int, int], ...],
        steps: int,
        text_tokens: int = 512,
    ) -> dict[str, Any]:
        if steps < 3:
            raise ValueError("EPE warmup steps must be >= 3")
        parameter = next(self.transformer.parameters())
        device, dtype = parameter.device, parameter.dtype
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for height, width in resolutions:
            image_tokens = (height // 16) * (width // 16)
            for lane_width in self.parallel_context.allowed_widths:
                if image_tokens % lane_width:
                    raise ValueError(
                        f"image token count {image_tokens} is not divisible by lane width {lane_width}"
                    )
                offset = (self.parallel_context.rank // lane_width) * lane_width
                lane = tuple(range(offset, offset + lane_width))
                generator = torch.Generator(device=device).manual_seed(
                    3_000_017 + height * 97 + width * 193 + lane[0] * 997
                )
                hidden_states = torch.randn(
                    (1, image_tokens, self.transformer.config.in_channels),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                prompt_embeds = torch.randn(
                    (1, text_tokens, self.transformer.config.joint_attention_dim),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                pooled = torch.randn(
                    (1, self.transformer.config.pooled_projection_dim),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                text_ids = torch.zeros((text_tokens, 3), device=device, dtype=dtype)
                image_ids = self._prepare_latent_image_ids(
                    1, height // 16, width // 16, device, dtype
                )
                timestep = torch.full((1,), 0.5, device=device, dtype=dtype)
                guidance = (
                    torch.full((1,), 3.5, device=device, dtype=torch.float32)
                    if self.transformer.config.guidance_embeds
                    else None
                )
                samples_ms: list[float] = []
                with self.parallel_context.activate(lane):
                    for _ in range(steps):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        step_started = time.perf_counter()
                        output = self.transformer(
                            hidden_states=hidden_states,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled,
                            timestep=timestep,
                            img_ids=image_ids,
                            txt_ids=text_ids,
                            guidance=guidance,
                            return_dict=False,
                        )[0]
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        if self.parallel_context.active.is_leader:
                            samples_ms.append(
                                (time.perf_counter() - step_started) * 1000
                            )
                        del output
                local = (
                    {"lane": list(lane), "samples_ms": samples_ms}
                    if self.parallel_context.rank == lane[0]
                    else None
                )
                gathered = [local]
                if self.parallel_context.world_size > 1:
                    gathered = [None] * self.parallel_context.world_size
                    dist.all_gather_object(gathered, local)
                if self.parallel_context.rank == 0:
                    lane_samples = [
                        sample
                        for item in gathered
                        if item is not None
                        for sample in item["samples_ms"]
                    ]
                    rows.append(
                        {
                            "resolution": [height, width],
                            "image_tokens": image_tokens,
                            "width": lane_width,
                            "batch_size": 1,
                            "cfg_conditions": 1,
                            "latency_ms": float(statistics.median(lane_samples)),
                            "samples_ms": lane_samples,
                            "state_bytes": hidden_states.numel()
                            * hidden_states.element_size(),
                        }
                    )
        report = [
            {
                "source": "startup_warmup",
                "steps": steps,
                "resolutions": [list(value) for value in resolutions],
                "allowed_lane_widths": list(self.parallel_context.allowed_widths),
                "attention_mode": self._epe_attention_mode,
                "ulysses_degree": self._epe_ulysses_degree,
                "rows": rows,
                "total_wall_ms": (time.perf_counter() - started) * 1000,
            }
            if self.parallel_context.rank == 0
            else None
        ]
        if self.parallel_context.world_size > 1:
            dist.broadcast_object_list(report, src=0)
        assert report[0] is not None
        return report[0]

    @torch.inference_mode()
    def prepare_request(
        self,
        *,
        prompt: str | list[str] | None = None,
        prompt_2: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        true_cfg_scale: float = 1.0,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> Flux1DenoiseState:
        if true_cfg_scale != 1.0:
            raise NotImplementedError("Flux.1 EPE does not support true CFG yet")
        if joint_attention_kwargs:
            raise NotImplementedError(
                "Flux.1 EPE does not support IP-Adapter or joint attention kwargs yet"
            )
        if num_images_per_prompt != 1:
            raise NotImplementedError(
                "Flux.1 EPE currently supports one image per prompt"
            )
        if height < 16 or width < 16 or height % 16 or width % 16:
            raise ValueError("Flux.1 height and width must be positive multiples of 16")
        if max_sequence_length > 512:
            raise ValueError("Flux.1 max_sequence_length cannot exceed 512")
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "pooled_prompt_embeds is required when prompt_embeds is provided"
            )
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("prompt or prompt_embeds must be provided")
        if batch_size != 1:
            raise NotImplementedError("Flux.1 EPE currently supports batch size one")

        device = self._execution_device
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
        latents, latent_image_ids = self.prepare_latents(
            batch_size,
            self.transformer.config.in_channels // 4,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        scheduler = self.scheduler.__class__.from_config(self.scheduler.config)
        selected_sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        if getattr(scheduler.config, "use_flow_sigmas", False):
            selected_sigmas = None
        mu = calculate_shift(
            latents.shape[1],
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
            sigmas=selected_sigmas,
            mu=mu,
        )
        scheduler.set_begin_index(0)
        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                (latents.shape[0],),
                guidance_scale,
                device=device,
                dtype=torch.float32,
            )
        return Flux1DenoiseState(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            timesteps=timesteps,
            scheduler=scheduler,
            guidance=guidance,
            actual_batch_size=batch_size,
            width=int(width),
            height=int(height),
            image_tokens=int(latents.shape[1]),
        )

    @torch.inference_mode()
    def denoise_step(
        self,
        state: Flux1DenoiseState,
        *,
        lane_ranks: tuple[int, ...],
    ) -> Flux1DenoiseState:
        if state.complete:
            raise RuntimeError("denoise state is already complete")
        with self.parallel_context.activate(lane_ranks):
            timestep_value = state.timesteps[state.step_index]
            timestep = timestep_value.expand(state.latents.shape[0]).to(
                state.latents.dtype
            )
            noise_pred = self.transformer(
                hidden_states=state.latents,
                timestep=timestep / 1000,
                guidance=state.guidance,
                pooled_projections=state.pooled_prompt_embeds,
                encoder_hidden_states=state.prompt_embeds,
                txt_ids=state.text_ids,
                img_ids=state.latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            state.latents = state.scheduler.step(
                noise_pred,
                timestep_value,
                state.latents,
                return_dict=False,
            )[0]
            state.step_index += 1
        return state

    def synchronize_state(self, state: Flux1DenoiseState, *, step_index: int) -> None:
        state.step_index = int(step_index)
        state.scheduler._step_index = int(step_index)

    @torch.inference_mode()
    def decode_request(
        self,
        state: Flux1DenoiseState,
        *,
        topology=None,
        parallel_vae: bool = True,
        vae_parallel_halo: int = 8,
        timings: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        if not state.complete:
            raise RuntimeError("cannot finalize an incomplete denoise state")
        profile = timings if timings is not None else {}
        latents = self._unpack_latents(
            state.latents,
            state.height,
            state.width,
            self.vae_scale_factor,
        )
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        device = latents.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile["vae_start_unix_ns"] = time.time_ns()
        started = time.perf_counter()
        active = topology or self.parallel_context.active
        image = parallel_tiled_vae_decode(
            latents,
            lambda value: self.vae.decode(value, return_dict=False)[0],
            topology=active,
            latent_split_dim=2,
            pixel_split_dim=2,
            scale=self.vae_spatial_scale_factor,
            halo=vae_parallel_halo,
            enabled=parallel_vae,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile["vae_end_unix_ns"] = time.time_ns()
        profile["vae_decode_ms"] = (time.perf_counter() - started) * 1000
        return image

    @torch.inference_mode()
    def finalize_request(
        self,
        state: Flux1DenoiseState,
        *,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        if output_type == "latent":
            image = state.latents
        else:
            decoded = self.decode_request(state)
            image = (
                None
                if decoded is None
                else self.image_processor.postprocess(decoded, output_type=output_type)
            )
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)

    def close(self) -> None:
        self.parallel_context.close()
