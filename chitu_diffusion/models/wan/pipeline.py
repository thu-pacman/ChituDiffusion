from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler, WanPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from ...parallel import (
    EpeParallelContext,
    parallel_tiled_vae_decode,
    resolve_context_parallel_config,
)
from .loader import load_wan_diffusers_components
from .transformer import EpeWanTransformer3DModel


@dataclass
class WanDenoiseState:
    """Request-owned Wan video latent and stateless scheduler cursor."""

    latents: torch.Tensor
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    timesteps: torch.Tensor
    scheduler: Any
    guidance_scale: float
    actual_batch_size: int
    width: int
    height: int
    num_frames: int
    image_tokens: int
    step_index: int = 0

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1 and self.negative_prompt_embeds is not None

    @property
    def complete(self) -> bool:
        return self.step_index >= len(self.timesteps)


def combine_wan_cfg_predictions(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    guidance_scale: float,
) -> torch.Tensor:
    return negative + guidance_scale * (positive - negative)


class EpeWanPipeline(WanPipeline):
    """Official Diffusers WanPipeline split into resumable EPAC stages."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs: Any):
        parallel = kwargs.pop("parallel_context", None)
        allowed_widths = kwargs.pop("allowed_lane_widths", None)
        attention_mode, ulysses_degree = resolve_context_parallel_config(
            kwargs.pop("attention_mode", "agkv"),
            kwargs.pop("ulysses_degree", None),
        )
        cfg_parallel = bool(kwargs.pop("cfg_parallel", True))
        parallel_vae = bool(kwargs.pop("parallel_vae", True))
        vae_parallel_halo = int(kwargs.pop("vae_parallel_halo", 8))
        if vae_parallel_halo < 0:
            raise ValueError("vae_parallel_halo must be non-negative")
        flow_shift = float(kwargs.pop("flow_shift", 8.0))
        if parallel is None:
            parallel = EpeParallelContext.from_torchrun(
                allowed_widths=(
                    tuple(int(width) for width in allowed_widths)
                    if allowed_widths is not None
                    else None
                ),
                ulysses_degree=ulysses_degree,
            )

        model_path = Path(pretrained_model_name_or_path)
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        local_files_only = bool(kwargs.pop("local_files_only", True))
        is_original_directory = (
            model_path.is_dir()
            and (model_path / "diffusion_pytorch_model.safetensors").exists()
        )
        if not is_original_directory:
            transformer = kwargs.pop("transformer", None)
            if transformer is None:
                transformer = EpeWanTransformer3DModel.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder="transformer",
                    parallel_context=parallel,
                    attention_mode=attention_mode,
                    torch_dtype=torch_dtype,
                    local_files_only=local_files_only,
                )
            pipeline = super().from_pretrained(
                pretrained_model_name_or_path,
                transformer=transformer,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
                **kwargs,
            )
            pipeline.register_modules(
                scheduler=FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    shift=flow_shift,
                    use_dynamic_shifting=False,
                )
            )
        else:
            if kwargs:
                raise TypeError(
                    "unsupported original-checkpoint loader options: "
                    + ", ".join(sorted(kwargs))
                )
            tokenizer, text_encoder, vae, scheduler, transformer = (
                load_wan_diffusers_components(
                    model_path,
                    parallel_context=parallel,
                    attention_mode=attention_mode,
                    torch_dtype=torch_dtype,
                    flow_shift=flow_shift,
                    local_files_only=local_files_only,
                )
            )
            pipeline = cls(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                scheduler=scheduler,
                transformer=transformer,
            )
        pipeline._epac_parallel_context = parallel
        pipeline._epac_attention_mode = attention_mode
        pipeline._epac_ulysses_degree = ulysses_degree
        pipeline._epac_cfg_parallel = cfg_parallel
        pipeline._epac_parallel_vae = parallel_vae
        pipeline._epac_vae_parallel_halo = vae_parallel_halo
        return pipeline

    @property
    def parallel_context(self) -> EpeParallelContext:
        parallel = getattr(self, "_epac_parallel_context", None)
        if parallel is None:
            raise RuntimeError("pipeline has no EPAC parallel context")
        return parallel

    @property
    def cfg_parallel(self) -> bool:
        return bool(getattr(self, "_epac_cfg_parallel", False))

    @property
    def parallel_vae(self) -> bool:
        return bool(getattr(self, "_epac_parallel_vae", True))

    @property
    def vae_parallel_halo(self) -> int:
        return int(getattr(self, "_epac_vae_parallel_halo", 8))

    @torch.inference_mode()
    def prepare_request(
        self,
        *,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> WanDenoiseState:
        if num_videos_per_prompt != 1:
            raise NotImplementedError("Wan EPAC supports one video per prompt")
        if attention_kwargs:
            raise NotImplementedError("Wan EPAC does not support attention kwargs")
        if height % 16 or width % 16:
            raise ValueError("Wan height and width must be multiples of 16")
        if num_frames < 1 or (num_frames - 1) % self.vae_scale_factor_temporal:
            raise ValueError("Wan num_frames must be positive and equal to 4n+1")
        if max_sequence_length > 512:
            raise ValueError("Wan max_sequence_length cannot exceed 512")
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("prompt or prompt_embeds must be provided")
        if batch_size != 1:
            raise NotImplementedError("Wan EPAC currently supports batch size one")

        device = self._execution_device
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance_scale > 1,
            num_videos_per_prompt=1,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        scheduler = self.scheduler.__class__.from_config(self.scheduler.config)
        scheduler.set_timesteps(num_inference_steps, device=device)
        latents = self.prepare_latents(
            batch_size,
            self.transformer.config.in_channels,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        latent_frames = latents.shape[2]
        image_tokens = latent_frames * (height // 16) * (width // 16)
        return WanDenoiseState(
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
            guidance_scale=float(guidance_scale),
            actual_batch_size=batch_size,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            image_tokens=int(image_tokens),
        )

    def _predict_branch(
        self,
        state: WanDenoiseState,
        timestep: torch.Tensor,
        *,
        negative: bool,
    ) -> torch.Tensor:
        embeds = state.negative_prompt_embeds if negative else state.prompt_embeds
        if embeds is None:
            raise RuntimeError("Wan CFG branch has no prompt embeddings")
        with self.transformer.cache_context("uncond" if negative else "cond"):
            return self.transformer(
                hidden_states=state.latents.to(self.transformer.dtype),
                timestep=timestep,
                encoder_hidden_states=embeds,
                return_dict=False,
            )[0]

    @torch.inference_mode()
    def denoise_step(
        self,
        state: WanDenoiseState,
        *,
        lane_ranks: tuple[int, ...],
    ) -> WanDenoiseState:
        if state.complete:
            raise RuntimeError("denoise state is already complete")
        timestep_value = state.timesteps[state.step_index]
        timestep = timestep_value.expand(state.actual_batch_size)
        with self.parallel_context.activate(lane_ranks):
            cfg_topology = (
                self.parallel_context.cfg_parallel_topology(lane_ranks)
                if state.do_classifier_free_guidance and self.cfg_parallel
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
            else:
                positive = self._predict_branch(state, timestep, negative=False)
                negative = (
                    self._predict_branch(state, timestep, negative=True)
                    if state.do_classifier_free_guidance
                    else None
                )
            prediction = (
                combine_wan_cfg_predictions(
                    positive,
                    negative,
                    guidance_scale=state.guidance_scale,
                )
                if negative is not None
                else positive
            )
            state.latents = state.scheduler.step(
                prediction,
                timestep_value,
                state.latents,
                return_dict=False,
            )[0].to(state.latents.dtype)
            state.step_index += 1
        return state

    def synchronize_state(self, state: WanDenoiseState, *, step_index: int) -> None:
        state.step_index = int(step_index)
        state.scheduler._step_index = int(step_index)

    @torch.inference_mode()
    def warmup_epe(
        self,
        *,
        resolutions: tuple[tuple[int, int], ...],
        steps: int,
        num_frames: int = 17,
        text_tokens: int = 512,
    ) -> dict[str, Any]:
        if steps < 3:
            raise ValueError("EPE warmup steps must be >= 3")
        parameter = next(self.transformer.parameters())
        device, dtype = parameter.device, parameter.dtype
        rows: list[dict[str, Any]] = []
        report_started = time.perf_counter()
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        for height, width in resolutions:
            image_tokens = latent_frames * (height // 16) * (width // 16)
            for lane_width in self.parallel_context.allowed_widths:
                offset = (self.parallel_context.rank // lane_width) * lane_width
                lane = tuple(range(offset, offset + lane_width))
                generator = torch.Generator(device=device).manual_seed(
                    7_000_021 + height * 97 + width * 193 + lane[0] * 997
                )
                latents = torch.randn(
                    (1, 16, latent_frames, height // 8, width // 8),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                positive = torch.randn(
                    (1, text_tokens, self.transformer.config.text_dim),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                negative = torch.randn(
                    positive.shape, generator=generator, device=device, dtype=dtype
                )
                timestep = torch.full((1,), 500.0, device=device, dtype=torch.float32)
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
                                    hidden_states=latents.to(dtype),
                                    timestep=timestep,
                                    encoder_hidden_states=embeds,
                                    return_dict=False,
                                )[0]
                            pos_pred, neg_pred = self.parallel_context.all_gather_cfg(
                                branch, cfg_topology
                            )
                        else:
                            pos_pred = self.transformer(
                                hidden_states=latents.to(dtype),
                                timestep=timestep,
                                encoder_hidden_states=positive,
                                return_dict=False,
                            )[0]
                            neg_pred = self.transformer(
                                hidden_states=latents.to(dtype),
                                timestep=timestep,
                                encoder_hidden_states=negative,
                                return_dict=False,
                            )[0]
                        prediction = combine_wan_cfg_predictions(
                            pos_pred, neg_pred, guidance_scale=6.0
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        if lane_topology.is_leader:
                            samples_ms.append((time.perf_counter() - started) * 1000)
                        del prediction
                local = (
                    {"lane": list(lane), "samples_ms": samples_ms}
                    if lane_topology.is_leader
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
                            "num_frames": num_frames,
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
                "num_frames": num_frames,
                "resolutions": [list(value) for value in resolutions],
                "allowed_lane_widths": list(self.parallel_context.allowed_widths),
                "attention_mode": self._epac_attention_mode,
                "ulysses_degree": self._epac_ulysses_degree,
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

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.vae.dtype)
        mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        inverse_std = (
            (1.0 / torch.tensor(self.vae.config.latents_std))
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        return latents / inverse_std + mean

    @torch.inference_mode()
    def decode_request(
        self,
        state: WanDenoiseState,
        *,
        topology=None,
        parallel_vae: bool | None = None,
        vae_parallel_halo: int | None = None,
        timings: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        if not state.complete:
            raise RuntimeError("cannot finalize an incomplete Wan state")
        topology = topology or self.parallel_context.active
        profile = timings if timings is not None else {}
        device = state.latents.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile["vae_start_unix_ns"] = time.time_ns()
        started = time.perf_counter()
        enabled = self.parallel_vae if parallel_vae is None else bool(parallel_vae)
        halo = (
            self.vae_parallel_halo
            if vae_parallel_halo is None
            else int(vae_parallel_halo)
        )
        video = parallel_tiled_vae_decode(
            self._denormalize_latents(state.latents),
            lambda value: self.vae.decode(value, return_dict=False)[0],
            topology=topology,
            latent_split_dim=4,
            pixel_split_dim=4,
            scale=int(self.vae_scale_factor_spatial),
            halo=halo,
            enabled=enabled,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile["vae_end_unix_ns"] = time.time_ns()
        profile["vae_decode_ms"] = (time.perf_counter() - started) * 1000
        return video

    @torch.inference_mode()
    def finalize_request(
        self,
        state: WanDenoiseState,
        *,
        output_type: str = "np",
        return_dict: bool = True,
    ):
        if output_type == "latent":
            video = state.latents
        else:
            decoded = self.decode_request(state)
            video = (
                None
                if decoded is None
                else self.video_processor.postprocess_video(
                    decoded, output_type=output_type
                )
            )
        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,)
        return WanPipelineOutput(frames=video)

    def close(self) -> None:
        self.parallel_context.close()
