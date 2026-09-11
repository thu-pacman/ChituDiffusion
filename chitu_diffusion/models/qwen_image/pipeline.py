from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from diffusers import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
    calculate_shift,
    retrieve_timesteps,
)

from ...parallel.cp import EpeParallelContext, resolve_context_parallel_config
from ...parallel.vae import parallel_vae_decode
from .transformer import EpeQwenImageTransformer2DModel


@dataclass
class QwenImageDenoiseState:
    """Request-owned Qwen-Image state with a private scheduler cursor."""

    latents: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_embeds_mask: torch.Tensor | None
    img_shapes: list[list[tuple[int, int, int]]]
    timesteps: torch.Tensor
    scheduler: Any
    guidance: torch.Tensor | None
    true_cfg_scale: float
    actual_batch_size: int
    width: int
    height: int
    image_tokens: int
    step_index: int = 0

    @property
    def do_true_cfg(self) -> bool:
        return self.true_cfg_scale > 1 and self.negative_prompt_embeds is not None

    @property
    def complete(self) -> bool:
        return self.step_index >= len(self.timesteps)


def combine_qwen_cfg_predictions(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    true_cfg_scale: float,
) -> torch.Tensor:
    combined = negative + true_cfg_scale * (positive - negative)
    cond_norm = torch.norm(positive, dim=-1, keepdim=True)
    combined_norm = torch.norm(combined, dim=-1, keepdim=True)
    return combined * (cond_norm / combined_norm)


class EpeQwenImagePipeline(QwenImagePipeline):
    """QwenImagePipeline split into resumable EPE request stages."""

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
        cfg_parallel = bool(kwargs.pop("cfg_parallel", True))
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
            transformer = EpeQwenImageTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                **transformer_kwargs,
            )
        elif not isinstance(transformer, EpeQwenImageTransformer2DModel):
            raise TypeError("transformer must be an EpeQwenImageTransformer2DModel")
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
        pipeline._epe_cfg_parallel = cfg_parallel
        return pipeline

    @property
    def parallel_context(self) -> EpeParallelContext:
        parallel = getattr(self, "_epe_parallel_context", None)
        if parallel is None:
            raise RuntimeError("pipeline has no EPE parallel context")
        return parallel

    @property
    def cfg_parallel(self) -> bool:
        return bool(getattr(self, "_epe_cfg_parallel", False))

    @torch.inference_mode()
    def prepare_request(
        self,
        *,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = " ",
        true_cfg_scale: float = 4.0,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> QwenImageDenoiseState:
        if num_images_per_prompt != 1:
            raise NotImplementedError(
                "Qwen-Image EPE currently supports one image per prompt"
            )
        if height < 16 or width < 16 or height % 16 or width % 16:
            raise ValueError(
                "Qwen-Image height and width must be positive multiples of 16"
            )
        if max_sequence_length > 1024:
            raise ValueError("Qwen-Image max_sequence_length cannot exceed 1024")
        if attention_kwargs:
            raise NotImplementedError(
                "Qwen-Image EPE does not support attention kwargs"
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
            raise NotImplementedError(
                "Qwen-Image EPE currently supports batch size one"
            )

        device = self._execution_device
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        has_negative = negative_prompt is not None or negative_prompt_embeds is not None
        do_true_cfg = true_cfg_scale > 1 and has_negative
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                )
            ]
        ]
        scheduler = self.scheduler.__class__.from_config(self.scheduler.config)
        selected_sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
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
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model")
        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                (latents.shape[0],),
                float(guidance_scale),
                device=device,
                dtype=torch.float32,
            )
        return QwenImageDenoiseState(
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds if do_true_cfg else None,
            negative_prompt_embeds_mask=(
                negative_prompt_embeds_mask if do_true_cfg else None
            ),
            img_shapes=img_shapes,
            timesteps=timesteps,
            scheduler=scheduler,
            guidance=guidance,
            true_cfg_scale=float(true_cfg_scale),
            actual_batch_size=batch_size,
            width=int(width),
            height=int(height),
            image_tokens=int(latents.shape[1]),
        )

    def _predict_branch(
        self,
        state: QwenImageDenoiseState,
        timestep: torch.Tensor,
        *,
        negative: bool,
    ) -> torch.Tensor:
        embeds = state.negative_prompt_embeds if negative else state.prompt_embeds
        mask = (
            state.negative_prompt_embeds_mask if negative else state.prompt_embeds_mask
        )
        if embeds is None:
            raise RuntimeError("Qwen-Image CFG branch has no prompt embeddings")
        with self.transformer.cache_context("uncond" if negative else "cond"):
            return self.transformer(
                hidden_states=state.latents,
                timestep=timestep / 1000,
                guidance=state.guidance,
                encoder_hidden_states=embeds,
                encoder_hidden_states_mask=mask,
                img_shapes=state.img_shapes,
                attention_kwargs=None,
                return_dict=False,
            )[0]

    @torch.inference_mode()
    def denoise_step(
        self,
        state: QwenImageDenoiseState,
        *,
        lane_ranks: tuple[int, ...],
    ) -> QwenImageDenoiseState:
        if state.complete:
            raise RuntimeError("denoise state is already complete")
        timestep_value = state.timesteps[state.step_index]
        timestep = timestep_value.expand(state.latents.shape[0]).to(state.latents.dtype)
        with self.parallel_context.activate(lane_ranks):
            cfg_topology = (
                self.parallel_context.cfg_parallel_topology(lane_ranks)
                if state.do_true_cfg and self.cfg_parallel
                else None
            )
            if cfg_topology is not None:
                with self.parallel_context.activate(cfg_topology.cp_ranks):
                    branch = self._predict_branch(
                        state,
                        timestep,
                        negative=cfg_topology.branch_index == 1,
                    )
                positive, negative = self.parallel_context.all_gather_cfg(
                    branch, cfg_topology
                )
                noise_pred = combine_qwen_cfg_predictions(
                    positive,
                    negative,
                    true_cfg_scale=state.true_cfg_scale,
                )
            else:
                positive = self._predict_branch(state, timestep, negative=False)
                if state.do_true_cfg:
                    negative = self._predict_branch(state, timestep, negative=True)
                    noise_pred = combine_qwen_cfg_predictions(
                        positive,
                        negative,
                        true_cfg_scale=state.true_cfg_scale,
                    )
                else:
                    noise_pred = positive
            latent_dtype = state.latents.dtype
            state.latents = state.scheduler.step(
                noise_pred,
                timestep_value,
                state.latents,
                return_dict=False,
            )[0]
            if state.latents.dtype != latent_dtype:
                state.latents = state.latents.to(latent_dtype)
            state.step_index += 1
        return state

    def synchronize_state(
        self, state: QwenImageDenoiseState, *, step_index: int
    ) -> None:
        state.step_index = int(step_index)
        state.scheduler._step_index = int(step_index)

    @torch.inference_mode()
    def warmup_epe(
        self,
        *,
        resolutions: tuple[tuple[int, int], ...],
        steps: int,
        text_tokens: int = 64,
    ) -> dict[str, Any]:
        if steps < 3:
            raise ValueError("EPE warmup steps must be >= 3")
        parameter = next(self.transformer.parameters())
        device, dtype = parameter.device, parameter.dtype
        rows: list[dict[str, Any]] = []
        report_started = time.perf_counter()
        for height, width in resolutions:
            image_tokens = (height // 16) * (width // 16)
            for lane_width in self.parallel_context.allowed_widths:
                offset = (self.parallel_context.rank // lane_width) * lane_width
                lane = tuple(range(offset, offset + lane_width))
                generator = torch.Generator(device=device).manual_seed(
                    4_000_019 + height * 97 + width * 193 + lane[0] * 997
                )
                latents = torch.randn(
                    (1, image_tokens, self.transformer.config.in_channels),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                positive = torch.randn(
                    (1, text_tokens, self.transformer.config.joint_attention_dim),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                negative = torch.randn(
                    positive.shape,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                timestep = torch.full((1,), 0.5, device=device, dtype=dtype)
                img_shapes = [[(1, height // 16, width // 16)]]
                samples_ms: list[float] = []
                with self.parallel_context.activate(lane) as lane_topology:
                    cfg_topology = (
                        self.parallel_context.cfg_parallel_topology(lane)
                        if self.cfg_parallel
                        else None
                    )
                    for _ in range(steps):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        if cfg_topology is not None:
                            embeds = (
                                positive if cfg_topology.branch_index == 0 else negative
                            )
                            with self.parallel_context.activate(cfg_topology.cp_ranks):
                                branch = self.transformer(
                                    hidden_states=latents,
                                    encoder_hidden_states=embeds,
                                    timestep=timestep,
                                    img_shapes=img_shapes,
                                    return_dict=False,
                                )[0]
                            pos_pred, neg_pred = self.parallel_context.all_gather_cfg(
                                branch, cfg_topology
                            )
                        else:
                            pos_pred = self.transformer(
                                hidden_states=latents,
                                encoder_hidden_states=positive,
                                timestep=timestep,
                                img_shapes=img_shapes,
                                return_dict=False,
                            )[0]
                            neg_pred = self.transformer(
                                hidden_states=latents,
                                encoder_hidden_states=negative,
                                timestep=timestep,
                                img_shapes=img_shapes,
                                return_dict=False,
                            )[0]
                        prediction = combine_qwen_cfg_predictions(
                            pos_pred, neg_pred, true_cfg_scale=4.0
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        if lane_topology.is_leader:
                            samples_ms.append((time.perf_counter() - started) * 1000)
                        del prediction
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
                            "cfg_conditions": 2,
                            "latency_ms": float(statistics.median(lane_samples)),
                            "samples_ms": lane_samples,
                            "state_bytes": latents.numel() * latents.element_size(),
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
                "total_wall_ms": (time.perf_counter() - report_started) * 1000,
            }
            if self.parallel_context.rank == 0
            else None
        ]
        if self.parallel_context.world_size > 1:
            dist.broadcast_object_list(report, src=0)
        assert report[0] is not None
        return report[0]

    @torch.inference_mode()
    def decode_request(
        self,
        state: QwenImageDenoiseState,
        *,
        topology=None,
        parallel_vae: bool = True,
        vae_parallel_halo: int = 8,
        timings: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        if not state.complete:
            raise RuntimeError("cannot finalize an incomplete denoise state")
        latents = self._unpack_latents(
            state.latents, state.height, state.width, self.vae_scale_factor
        ).to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            (1.0 / torch.tensor(self.vae.config.latents_std))
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        profile = timings if timings is not None else {}
        device = latents.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile["vae_start_unix_ns"] = time.time_ns()
        started = time.perf_counter()
        image = parallel_vae_decode(
            self.vae,
            latents,
            decode_fn=lambda value: self.vae.decode(value, return_dict=False)[0][
                :, :, 0
            ],
            topology=topology or self.parallel_context.active,
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
        state: QwenImageDenoiseState,
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
        return QwenImagePipelineOutput(images=image)

    def close(self) -> None:
        self.parallel_context.close()
