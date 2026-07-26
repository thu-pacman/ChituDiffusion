from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from diffusers import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput
from diffusers.pipelines.z_image.pipeline_z_image import (
    calculate_shift,
    retrieve_timesteps,
)

from .epe import ZImageEpeModule
from ...parallel import (
    ActiveLaneTopology,
    EpeParallelContext,
    parallel_tiled_vae_decode,
)
from .transformer import EpeZImageTransformer2DModel


@dataclass
class ZImageDenoiseState:
    """Request-owned state that can move between pre-created EPE lanes."""

    latents: torch.Tensor
    prompt_embeds: list[torch.Tensor]
    negative_prompt_embeds: list[torch.Tensor] | None
    timesteps: torch.Tensor
    scheduler: Any
    actual_batch_size: int
    guidance_scale: float
    cfg_normalization: bool
    cfg_truncation: float
    width: int
    height: int
    image_tokens: int
    step_index: int = 0

    @property
    def complete(self) -> bool:
        return self.step_index >= len(self.timesteps)


def combine_cfg_predictions(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    guidance_scale: float,
    normalize: bool,
) -> torch.Tensor:
    """Combine batched cond/uncond predictions with Z-Image CFG semantics."""
    predictions = []
    for positive_value, negative_value in zip(positive, negative, strict=True):
        positive_value = positive_value.float()
        negative_value = negative_value.float()
        prediction = positive_value + guidance_scale * (
            positive_value - negative_value
        )
        if normalize:
            original_norm = torch.linalg.vector_norm(positive_value)
            prediction_norm = torch.linalg.vector_norm(prediction)
            max_norm = original_norm * float(normalize)
            if prediction_norm > max_norm:
                prediction = prediction * (max_norm / prediction_norm)
        predictions.append(prediction)
    return torch.stack(predictions)


class EpeZImagePipeline(ZImagePipeline):
    """A Diffusers-compatible Z-Image pipeline independent of Chitu runtime state."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs: Any):
        parallel = kwargs.pop("parallel_context", None)
        allowed_widths = kwargs.pop("allowed_lane_widths", None)
        attention_mode = str(kwargs.pop("attention_mode", "agkv"))
        cfg_parallel = bool(kwargs.pop("cfg_parallel", True))
        ulysses_degree = kwargs.pop("ulysses_degree", None)
        if ulysses_degree is None:
            ulysses_degree = 2 if attention_mode == "usp" else 1
        epe_options = dict(kwargs.pop("epe_options", {}))
        epe_options.setdefault("attention_mode", attention_mode)
        epe_options.setdefault("cfg_parallel", cfg_parallel)
        if parallel is None:
            parallel = EpeParallelContext.from_torchrun(
                allowed_widths=(
                    tuple(int(width) for width in allowed_widths)
                    if allowed_widths is not None
                    else None
                ),
                ulysses_degree=int(ulysses_degree),
            )
        epe = ZImageEpeModule(parallel, **epe_options)

        transformer = kwargs.pop("transformer", None)
        if transformer is None:
            transformer_kwargs: dict[str, Any] = {
                "subfolder": "transformer",
                "epe_module": epe,
            }
            for key in ("torch_dtype", "local_files_only", "variant", "revision"):
                if key in kwargs:
                    transformer_kwargs[key] = kwargs[key]
            transformer = EpeZImageTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                **transformer_kwargs,
            )
        elif not isinstance(transformer, EpeZImageTransformer2DModel):
            raise TypeError("transformer must be an EpeZImageTransformer2DModel")
        else:
            transformer.configure_epe(epe)

        pipeline = super().from_pretrained(
            pretrained_model_name_or_path,
            transformer=transformer,
            **kwargs,
        )
        return pipeline

    @property
    def parallel_context(self) -> EpeParallelContext:
        epe = getattr(self.transformer, "epe", None)
        if epe is None:
            raise RuntimeError("pipeline has no EPE parallel context")
        return epe.parallel

    @torch.inference_mode()
    def warmup_epe(
        self,
        *,
        resolutions: tuple[tuple[int, int], ...],
        steps: int = 5,
        text_tokens: int = 32,
        cfg_conditions: int = 2,
    ) -> dict[str, Any]:
        epe = getattr(self.transformer, "epe", None)
        if epe is None:
            raise RuntimeError("pipeline has no EPE module")
        return epe.warmup_transformer(
            self.transformer,
            resolutions=resolutions,
            steps=steps,
            vae_scale_factor=self.vae_scale_factor,
            text_tokens=text_tokens,
            cfg_conditions=cfg_conditions,
        )

    @torch.inference_mode()
    def warmup_vae(
        self,
        *,
        resolutions: tuple[tuple[int, int], ...],
        steps: int,
        parallel_vae: bool,
        vae_parallel_halo: int,
    ) -> list[dict[str, Any]]:
        """Measure lane-critical VAE decode and leader D2H for every shape."""
        parameter = next(self.vae.parameters())
        device = parameter.device
        dtype = self.vae.dtype
        latent_channels = int(self.vae.config.latent_channels)
        rows: list[dict[str, Any]] = []
        for height, width in resolutions:
            latent_shape = (
                1,
                latent_channels,
                height // int(self.vae_scale_factor),
                width // int(self.vae_scale_factor),
            )
            image_tokens = (height // 16) * (width // 16)
            for lane_width in self.parallel_context.allowed_widths:
                offset = (self.parallel_context.rank // lane_width) * lane_width
                lane = tuple(range(offset, offset + lane_width))
                generator = torch.Generator(device=device).manual_seed(
                    7_000_001 + height * 97 + width * 193 + lane[0] * 997
                )
                latents = torch.randn(
                    latent_shape,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                vae_samples_ms = []
                d2h_samples_ms = []
                with self.parallel_context.activate(lane) as topology:
                    for _ in range(steps):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        decoded = parallel_tiled_vae_decode(
                            latents,
                            lambda value: self.vae.decode(value, return_dict=False)[0],
                            topology=topology,
                            latent_split_dim=2,
                            pixel_split_dim=2,
                            scale=int(self.vae_scale_factor),
                            halo=vae_parallel_halo,
                            enabled=parallel_vae,
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        vae_samples_ms.append((time.perf_counter() - started) * 1000.0)
                        if topology.is_leader:
                            if decoded is None:
                                raise RuntimeError("VAE warmup leader has no output")
                            d2h_started = time.perf_counter()
                            host = decoded.to(device="cpu")
                            d2h_samples_ms.append(
                                (time.perf_counter() - d2h_started) * 1000.0
                            )
                            del host
                        del decoded

                local = {
                    "rank": self.parallel_context.rank,
                    "lane": list(lane),
                    "vae_samples_ms": vae_samples_ms,
                    "d2h_samples_ms": d2h_samples_ms,
                }
                gathered = [local]
                if self.parallel_context.world_size > 1:
                    gathered = [None] * self.parallel_context.world_size
                    dist.all_gather_object(gathered, local)
                if self.parallel_context.rank == 0:
                    lane_groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
                    for item in gathered:
                        assert item is not None
                        lane_groups.setdefault(tuple(item["lane"]), []).append(item)
                    critical_samples = []
                    vae_critical_samples = []
                    d2h_samples = []
                    for members in lane_groups.values():
                        leader = min(members, key=lambda item: item["rank"])
                        for sample_index in range(steps):
                            vae_ms = max(
                                item["vae_samples_ms"][sample_index] for item in members
                            )
                            d2h_ms = leader["d2h_samples_ms"][sample_index]
                            vae_critical_samples.append(vae_ms)
                            d2h_samples.append(d2h_ms)
                            critical_samples.append(vae_ms + d2h_ms)
                    rows.append(
                        {
                            "resolution": [height, width],
                            "image_tokens": image_tokens,
                            "width": lane_width,
                            "batch_size": 1,
                            "cfg_conditions": 2,
                            # Denoise state remains float32 until VAE finalize,
                            # even when the VAE itself runs in bf16/fp16.
                            "state_bytes": latents.numel()
                            * torch.empty((), dtype=torch.float32).element_size(),
                            "terminal_ms": float(statistics.median(critical_samples)),
                            "vae_latency_ms": float(
                                statistics.median(vae_critical_samples)
                            ),
                            "d2h_latency_ms": float(statistics.median(d2h_samples)),
                            "terminal_samples_ms": critical_samples,
                        }
                    )
                del latents
        broadcast = [rows if self.parallel_context.rank == 0 else None]
        if self.parallel_context.world_size > 1:
            dist.broadcast_object_list(broadcast, src=0)
        assert broadcast[0] is not None
        return broadcast[0]

    @torch.inference_mode()
    def __call__(
        self, *args: Any, lane_ranks: tuple[int, ...] | None = None, **kwargs: Any
    ):
        ranks = lane_ranks or tuple(range(self.parallel_context.world_size))
        with self.parallel_context.activate(ranks):
            return super().__call__(*args, **kwargs)

    @torch.inference_mode()
    def prepare_request(
        self,
        *,
        prompt: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: list[torch.Tensor] | None = None,
        negative_prompt_embeds: list[torch.Tensor] | None = None,
        max_sequence_length: int = 512,
    ) -> ZImageDenoiseState:
        """Prepare one request without entering the denoise loop.

        The scheduler is cloned per request so concurrent lanes never share a
        mutable step cursor.
        """
        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale or width % vae_scale:
            raise ValueError(
                f"height and width must be divisible by {vae_scale}; "
                f"got {height}x{width}"
            )
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = len(prompt_embeds)
        else:
            raise ValueError("prompt or prompt_embeds must be provided")

        device = self._execution_device
        do_cfg = guidance_scale > 0
        if prompt_embeds is not None and prompt is None:
            if do_cfg and negative_prompt_embeds is None:
                raise ValueError(
                    "negative_prompt_embeds is required with prompt_embeds when CFG is enabled"
                )
        else:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_cfg,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )
        assert prompt_embeds is not None

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.in_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )
        if num_images_per_prompt > 1:
            prompt_embeds = [
                embedding
                for embedding in prompt_embeds
                for _ in range(num_images_per_prompt)
            ]
            if do_cfg and negative_prompt_embeds:
                negative_prompt_embeds = [
                    embedding
                    for embedding in negative_prompt_embeds
                    for _ in range(num_images_per_prompt)
                ]

        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        scheduler = self.scheduler.__class__.from_config(self.scheduler.config)
        scheduler.sigma_min = 0.0
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        return ZImageDenoiseState(
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            timesteps=timesteps,
            scheduler=scheduler,
            actual_batch_size=batch_size * num_images_per_prompt,
            guidance_scale=float(guidance_scale),
            cfg_normalization=bool(cfg_normalization),
            cfg_truncation=float(cfg_truncation),
            width=int(width),
            height=int(height),
            image_tokens=int(image_seq_len),
        )

    @torch.inference_mode()
    def denoise_step(
        self,
        state: ZImageDenoiseState,
        *,
        lane_ranks: tuple[int, ...],
    ) -> ZImageDenoiseState:
        """Run exactly one denoise step on the selected active lane."""
        if state.complete:
            raise RuntimeError("denoise state is already complete")
        with self.parallel_context.activate(lane_ranks):
            timestep_value = state.timesteps[state.step_index]
            timestep = timestep_value.expand(state.latents.shape[0])
            timestep = (1000 - timestep) / 1000
            current_guidance_scale = state.guidance_scale
            if state.guidance_scale > 0 and state.cfg_truncation <= 1:
                if timestep[0].item() > state.cfg_truncation:
                    current_guidance_scale = 0.0
            apply_cfg = state.guidance_scale > 0 and current_guidance_scale > 0

            latent_model_input = state.latents.to(self.transformer.dtype)
            prompt_model_input = state.prompt_embeds
            timestep_model_input = timestep
            epe = self.transformer.epe
            cfg_topology = (
                self.parallel_context.cfg_parallel_topology(lane_ranks)
                if apply_cfg and epe.cfg_parallel
                else None
            )
            if apply_cfg and cfg_topology is None:
                if state.negative_prompt_embeds is None:
                    raise RuntimeError("CFG state has no negative prompt embeddings")
                latent_model_input = latent_model_input.repeat(2, 1, 1, 1)
                prompt_model_input = state.prompt_embeds + state.negative_prompt_embeds
                timestep_model_input = timestep.repeat(2)

            if cfg_topology is not None:
                if state.negative_prompt_embeds is None:
                    raise RuntimeError("CFG state has no negative prompt embeddings")
                branch_prompt = (
                    state.prompt_embeds
                    if cfg_topology.branch_index == 0
                    else state.negative_prompt_embeds
                )
                with self.parallel_context.activate(cfg_topology.cp_ranks):
                    local_out = self.transformer(
                        list(latent_model_input.unsqueeze(2).unbind(dim=0)),
                        timestep_model_input,
                        branch_prompt,
                        return_dict=False,
                    )[0]
                positive, negative = self.parallel_context.all_gather_cfg(
                    torch.stack(local_out), cfg_topology
                )
                model_out = None
            else:
                model_out = self.transformer(
                    list(latent_model_input.unsqueeze(2).unbind(dim=0)),
                    timestep_model_input,
                    prompt_model_input,
                    return_dict=False,
                )[0]
            if apply_cfg:
                if cfg_topology is None:
                    assert model_out is not None
                    positive = torch.stack(model_out[: state.actual_batch_size])
                    negative = torch.stack(model_out[state.actual_batch_size :])
                noise_pred = combine_cfg_predictions(
                    positive,
                    negative,
                    guidance_scale=current_guidance_scale,
                    normalize=state.cfg_normalization,
                )
            else:
                assert model_out is not None
                noise_pred = torch.stack([value.float() for value in model_out])

            noise_pred = -noise_pred.squeeze(2)
            state.latents = state.scheduler.step(
                noise_pred.to(torch.float32),
                timestep_value,
                state.latents,
                return_dict=False,
            )[0]
            state.step_index += 1
        return state

    def synchronize_state(
        self,
        state: ZImageDenoiseState,
        *,
        step_index: int,
    ) -> None:
        """Align a replicated scheduler cursor after another lane ran a phase."""
        state.step_index = int(step_index)
        state.scheduler._step_index = int(step_index)

    @torch.inference_mode()
    def decode_request(
        self,
        state: ZImageDenoiseState,
        *,
        topology: ActiveLaneTopology | None = None,
        parallel_vae: bool = True,
        vae_parallel_halo: int = 8,
        timings: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        """Decode completed latents without performing CPU image conversion."""
        if not state.complete:
            raise RuntimeError("cannot finalize an incomplete denoise state")
        profile = timings if timings is not None else {}
        profile.update(
            {
                "latent_shape": list(state.latents.shape),
                "latent_dtype": str(state.latents.dtype),
                "latent_contiguous": bool(state.latents.is_contiguous()),
            }
        )
        device = state.latents.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            profile.update(
                {
                    "cuda_allocated_before_mb": torch.cuda.memory_allocated(device)
                    / (1024 * 1024),
                    "cuda_reserved_before_mb": torch.cuda.memory_reserved(device)
                    / (1024 * 1024),
                }
            )
        latents = state.latents.to(self.vae.dtype)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        profile["vae_start_unix_ns"] = time.time_ns()
        decode_started = time.perf_counter()
        active = topology or self.parallel_context.active

        def decode_fn(value: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(value, return_dict=False)[0]

        image = parallel_tiled_vae_decode(
            latents,
            decode_fn,
            topology=active,
            latent_split_dim=2,
            pixel_split_dim=2,
            scale=int(self.vae_scale_factor),
            halo=vae_parallel_halo,
            enabled=parallel_vae,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile["vae_end_unix_ns"] = time.time_ns()
        profile["vae_decode_ms"] = (time.perf_counter() - decode_started) * 1000.0
        if device.type == "cuda":
            profile.update(
                {
                    "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device)
                    / (1024 * 1024),
                    "cuda_allocated_after_mb": torch.cuda.memory_allocated(device)
                    / (1024 * 1024),
                    "cuda_reserved_after_mb": torch.cuda.memory_reserved(device)
                    / (1024 * 1024),
                }
            )
        hooks_started = time.perf_counter()
        self.maybe_free_model_hooks()
        profile["model_hooks_ms"] = (time.perf_counter() - hooks_started) * 1000.0
        return image

    @torch.inference_mode()
    def finalize_request(
        self,
        state: ZImageDenoiseState,
        *,
        output_type: str = "pil",
        return_dict: bool = True,
        timings: dict[str, Any] | None = None,
    ):
        """Decode and postprocess synchronously for Diffusers-style generation."""
        if not state.complete:
            raise RuntimeError("cannot finalize an incomplete denoise state")
        if output_type == "latent":
            image = state.latents
        else:
            image = self.decode_request(state, timings=timings)
            profile = timings if timings is not None else {}
            if image is not None:
                profile["postprocess_start_unix_ns"] = time.time_ns()
                postprocess_started = time.perf_counter()
                image = self.image_processor.postprocess(image, output_type=output_type)
                profile["postprocess_end_unix_ns"] = time.time_ns()
                profile["postprocess_ms"] = (
                    time.perf_counter() - postprocess_started
                ) * 1000.0
        if not return_dict:
            return (image,)
        return ZImagePipelineOutput(images=image)

    def close(self) -> None:
        epe = getattr(self.transformer, "epe", None)
        if epe is not None:
            epe.parallel.close()
