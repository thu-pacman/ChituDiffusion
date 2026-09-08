from __future__ import annotations

import statistics
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler

from ...parallel.cp import ActiveLaneTopology, EpeParallelContext
from ...parallel.vae import parallel_tiled_vae_decode


@dataclass
class LLaDAImageDenoiseState:
    """Mutable state for one batch-one denoising request."""

    latents: torch.Tensor
    cond_cap_feats: list[torch.Tensor]
    uncond_cap_feats: list[torch.Tensor] | None
    cond_glm_cap_feats: list[torch.Tensor] | None
    uncond_glm_cap_feats: list[torch.Tensor] | None
    source_latents: list[torch.Tensor] | None
    timesteps: torch.Tensor
    scheduler: FlowMatchEulerDiscreteScheduler
    guidance_scale: float
    generation_mode: str
    width: int
    height: int
    image_tokens: int
    step_index: int = 0

    @property
    def complete(self) -> bool:
        return self.step_index >= len(self.timesteps)


class LLaDAImageRuntimeMixin:
    """Chitu request lifecycle for the official LLaDA-Image pipeline."""

    def _prepare_sigmas(self, num_inference_steps: int) -> list[float]:
        if getattr(self.scheduler.config, "use_uniform_sigmas", False):
            return torch.linspace(
                1.0,
                0.0,
                num_inference_steps + 1,
                dtype=torch.float64,
            )[:-1].tolist()

        schedule = torch.linspace(
            0.001,
            1.0,
            num_inference_steps + 1,
            dtype=torch.float64,
        )[:-1]
        schedule = (1 - (1 - schedule**1.17) ** 0.8) ** 1.1
        return (1 - schedule).tolist()

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
        text_tokens: int | None = None,
        cfg_conditions: int = 2,
    ) -> dict[str, Any]:
        epe = getattr(self.transformer, "epe", None)
        if epe is None:
            raise RuntimeError("pipeline has no EPE module")
        if text_tokens is None:
            text_tokens = int(self.queryformer.config.num_queries) + 32
        return epe.warmup_transformer(
            self.transformer,
            resolutions=resolutions,
            steps=steps,
            vae_scale_factor=self.latent_scale_factor,
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
        """Measure terminal VAE decode and leader D2H for each lane width."""
        parameter = next(self.vae.parameters())
        device = parameter.device
        dtype = self.vae.dtype
        in_channels = int(self.transformer.config.in_channels)
        rows: list[dict[str, Any]] = []
        if parallel_vae and max(self.parallel_context.allowed_widths) > 1:
            warnings.warn(
                "LLaDA-Image parallel VAE warmup measures an approximate decode",
                RuntimeWarning,
                stacklevel=2,
            )

        for height, width in resolutions:
            latent_height = height // int(self.latent_scale_factor)
            latent_width = width // int(self.latent_scale_factor)
            raw_image_tokens = latent_height * latent_width
            image_tokens = ((raw_image_tokens + 31) // 32) * 32
            latent_shape = (1, in_channels, latent_height, latent_width)
            for lane_width in self.parallel_context.allowed_widths:
                offset = (self.parallel_context.rank // lane_width) * lane_width
                lane = tuple(range(offset, offset + lane_width))
                generator = torch.Generator(device=device).manual_seed(
                    7_100_003 + height * 97 + width * 193 + lane[0] * 997
                )
                patchified_latents = torch.randn(
                    latent_shape,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                vae_latents = patchified_latents.to(dtype=dtype)
                latent_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(vae_latents)
                latent_std = torch.sqrt(
                    self.vae.bn.running_var.view(1, -1, 1, 1)
                    + self.vae.config.batch_norm_eps
                ).to(vae_latents)
                vae_latents = self._unpatchify_latents(
                    vae_latents * latent_std + latent_mean
                )
                vae_samples_ms: list[float] = []
                d2h_samples_ms: list[float] = []
                with self.parallel_context.activate(lane) as topology:
                    for _ in range(steps):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        decoded = parallel_tiled_vae_decode(
                            vae_latents,
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
                    terminal_samples: list[float] = []
                    vae_critical_samples: list[float] = []
                    d2h_critical_samples: list[float] = []
                    for members in lane_groups.values():
                        leader = min(members, key=lambda item: item["rank"])
                        for sample_index in range(steps):
                            vae_ms = max(
                                item["vae_samples_ms"][sample_index] for item in members
                            )
                            d2h_ms = leader["d2h_samples_ms"][sample_index]
                            vae_critical_samples.append(vae_ms)
                            d2h_critical_samples.append(d2h_ms)
                            terminal_samples.append(vae_ms + d2h_ms)
                    rows.append(
                        {
                            "resolution": [height, width],
                            "image_tokens": image_tokens,
                            "width": lane_width,
                            "batch_size": 1,
                            "cfg_conditions": 2,
                            "state_bytes": patchified_latents.numel()
                            * torch.empty((), dtype=torch.float32).element_size(),
                            "terminal_ms": float(statistics.median(terminal_samples)),
                            "vae_latency_ms": float(
                                statistics.median(vae_critical_samples)
                            ),
                            "d2h_latency_ms": float(
                                statistics.median(d2h_critical_samples)
                            ),
                            "terminal_samples_ms": terminal_samples,
                        }
                    )
                del patchified_latents, vae_latents

        broadcast = [rows if self.parallel_context.rank == 0 else None]
        if self.parallel_context.world_size > 1:
            dist.broadcast_object_list(broadcast, src=0)
        assert broadcast[0] is not None
        return broadcast[0]

    def _lane_context(self, lane_ranks: tuple[int, ...] | None):
        epe = getattr(self.transformer, "epe", None)
        if epe is None:
            if lane_ranks not in (None, (0,)):
                raise ValueError("lane_ranks requires an EPE-enabled transformer")
            return nullcontext(None)
        ranks = lane_ranks or tuple(range(epe.parallel.world_size))
        return epe.parallel.activate(ranks)

    def _generate_vq_tokens_synchronized(
        self,
        prompt: str | list[str],
        height: int,
        width: int,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        distributed = (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )
        is_leader = not distributed or dist.get_rank() == 0
        payload: tuple[torch.Tensor | None, str | None] | None = None
        seed_generator = generator[0] if isinstance(generator, list) else generator
        seed = seed_generator.initial_seed() if seed_generator is not None else None

        if is_leader:
            try:
                text_device = self.text_encoder.device
                fork_devices = [text_device] if text_device.type == "cuda" else []
                with torch.random.fork_rng(devices=fork_devices):
                    if seed is not None:
                        torch.manual_seed(seed)
                    token_ids = self.generate_vq_tokens(prompt, height, width)
                payload = (token_ids.detach().cpu(), None)
            except Exception as exc:
                if not distributed:
                    raise
                payload = (None, f"{type(exc).__name__}: {exc}")

        if distributed:
            objects: list[Any] = [payload]
            dist.broadcast_object_list(objects, src=0)
            payload = objects[0]
        if payload is None:
            raise RuntimeError("VQ token broadcast produced no payload")
        token_ids, error = payload
        if error is not None:
            raise RuntimeError(f"VQ token generation failed on rank 0: {error}")
        if token_ids is None:
            raise RuntimeError("VQ token broadcast produced no token IDs")
        return token_ids

    @torch.inference_mode()
    def prepare_request(
        self,
        *,
        prompt: str | list[str] | None = None,
        image: Any | None = None,
        generation_mode: str = "text",
        negative_prompt: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 2048,
    ) -> LLaDAImageDenoiseState:
        self.check_inputs(
            prompt,
            image,
            generation_mode,
            height,
            width,
            num_images_per_prompt,
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            ["latents"],
            num_inference_steps,
        )

        if prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        elif isinstance(prompt, str):
            batch_size = 1
        else:
            assert prompt is not None
            batch_size = len(prompt)
        effective_batch_size = batch_size * num_images_per_prompt
        if effective_batch_size != 1:
            raise ValueError(
                "LLaDA-Image currently supports exactly one image per request; "
                f"got effective batch size {effective_batch_size}"
            )

        device = self.transformer.device
        do_cfg = guidance_scale > 1.0
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            negative_prompt,
            do_cfg,
            num_images_per_prompt,
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            max_sequence_length,
            device,
        )

        source_latents = None
        semantic_features = None
        if generation_mode == "vq":
            assert prompt is not None
            vq_token_ids = self._generate_vq_tokens_synchronized(
                prompt, height, width, generator
            )
            semantic_features = self.sigvq(
                token_ids=vq_token_ids.to(self.sigvq.device)
            ).semantic_features
        elif generation_mode == "editing":
            source_latents, semantic_features = self._encode_source_image(
                image,
                height,
                width,
                batch_size,
                num_images_per_prompt,
            )

        latent_shape = (
            1,
            self.transformer.config.in_channels,
            height // self.latent_scale_factor,
            width // self.latent_scale_factor,
        )
        if latents is None:
            latents = self._prepare_random_latents(
                latent_shape,
                generator=generator,
                device=device,
            )
        else:
            if tuple(latents.shape) != latent_shape:
                raise ValueError(
                    f"Expected latents with shape {latent_shape}, got {tuple(latents.shape)}"
                )
            latents = latents.to(device=device, dtype=torch.float32)

        sigmas = self._prepare_sigmas(num_inference_steps)
        scheduler = self.scheduler.__class__.from_config(self.scheduler.config)
        scheduler.set_timesteps(sigmas=sigmas, device=device)

        cond_cap_feats = [
            embeds[mask].to(device=device, dtype=self.transformer.dtype)
            for embeds, mask in zip(
                prompt_embeds,
                prompt_attention_mask.bool(),
                strict=True,
            )
        ]
        uncond_cap_feats = None
        if do_cfg:
            assert negative_prompt_embeds is not None
            assert negative_prompt_attention_mask is not None
            uncond_cap_feats = [
                embeds[mask].to(device=device, dtype=self.transformer.dtype)
                for embeds, mask in zip(
                    negative_prompt_embeds,
                    negative_prompt_attention_mask.bool(),
                    strict=True,
                )
            ]

        cond_glm_cap_feats = None
        uncond_glm_cap_feats = None
        source_latent_list = None
        if semantic_features is not None:
            cond_glm_cap_feats = [
                features.to(device=device, dtype=self.transformer.dtype)
                for features in semantic_features
            ]
            if do_cfg:
                empty_glm = semantic_features.new_zeros(
                    (0, semantic_features.shape[-1])
                ).to(device=device, dtype=self.transformer.dtype)
                uncond_glm_cap_feats = [empty_glm]
            if source_latents is not None:
                source_latent_list = [
                    latent.unsqueeze(1).to(
                        device=device,
                        dtype=self.transformer.dtype,
                    )
                    for latent in source_latents
                ]

        target_tokens = latent_shape[-2] * latent_shape[-1]
        image_tokens = target_tokens * (2 if generation_mode == "editing" else 1)
        return LLaDAImageDenoiseState(
            latents=latents,
            cond_cap_feats=cond_cap_feats,
            uncond_cap_feats=uncond_cap_feats,
            cond_glm_cap_feats=cond_glm_cap_feats,
            uncond_glm_cap_feats=uncond_glm_cap_feats,
            source_latents=source_latent_list,
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
            guidance_scale=float(guidance_scale),
            generation_mode=generation_mode,
            width=int(width),
            height=int(height),
            image_tokens=int(image_tokens),
        )

    def _prepare_random_latents(
        self,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
    ) -> torch.Tensor:
        from diffusers.utils.torch_utils import randn_tensor

        latents = randn_tensor(
            shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        return latents.to(self.transformer.dtype).float()

    def _transformer_branch(
        self,
        state: LLaDAImageDenoiseState,
        *,
        timestep: torch.Tensor,
        cap_feats: list[torch.Tensor],
        glm_cap_feats: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        glm_cap_lengths = None
        if (
            state.generation_mode == "editing"
            and glm_cap_feats is not None
            and any(len(features) == 0 for features in glm_cap_feats)
        ):
            if state.cond_glm_cap_feats is None or len(glm_cap_feats) != len(
                state.cond_glm_cap_feats
            ):
                raise RuntimeError("invalid unconditional editing condition state")
            glm_cap_lengths = [len(features) for features in glm_cap_feats]
            glm_cap_feats = [
                torch.zeros_like(cond_features) if len(features) == 0 else features
                for features, cond_features in zip(
                    glm_cap_feats,
                    state.cond_glm_cap_feats,
                    strict=True,
                )
            ]
        latent_list = [
            latent.unsqueeze(1).to(self.transformer.dtype) for latent in state.latents
        ]
        output = self.transformer(
            x=latent_list,
            t=timestep.to(self.transformer.dtype),
            cap_feats=cap_feats,
            glm_cap_feats=glm_cap_feats,
            source_latents=state.source_latents,
            glm_cap_lengths=glm_cap_lengths,
        ).sample
        return torch.stack(output, dim=0)

    def _transformer_cfg_batch(
        self,
        state: LLaDAImageDenoiseState,
        *,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.uncond_cap_feats is None:
            raise RuntimeError("CFG state has no unconditional caption")

        latent_model_input = torch.cat(
            [state.latents, state.latents],
            dim=0,
        )
        latent_list = [
            latent.unsqueeze(1).to(self.transformer.dtype)
            for latent in latent_model_input
        ]
        glm_cap_feats = state.cond_glm_cap_feats
        if glm_cap_feats is not None:
            assert state.uncond_glm_cap_feats is not None
            glm_cap_feats = glm_cap_feats + state.uncond_glm_cap_feats
        source_latents = state.source_latents
        if source_latents is not None:
            source_latents = source_latents + source_latents
        output = self.transformer(
            x=latent_list,
            t=timestep.expand(2).to(self.transformer.dtype),
            cap_feats=state.cond_cap_feats + state.uncond_cap_feats,
            glm_cap_feats=glm_cap_feats,
            source_latents=source_latents,
        ).sample
        conditional, unconditional = torch.stack(output, dim=0).chunk(2)
        return conditional, unconditional

    @torch.inference_mode()
    def denoise_step(
        self,
        state: LLaDAImageDenoiseState,
        *,
        lane_ranks: tuple[int, ...] | None = None,
    ) -> LLaDAImageDenoiseState:
        if state.complete:
            raise RuntimeError("denoise state is already complete")

        epe = getattr(self.transformer, "epe", None)
        ranks = (
            lane_ranks
            if lane_ranks is not None
            else tuple(range(epe.parallel.world_size))
            if epe is not None
            else None
        )
        with self._lane_context(ranks):
            timestep_value = state.timesteps[state.step_index]
            model_timestep = (
                timestep_value / state.scheduler.config.num_train_timesteps
            ).expand(1)
            apply_cfg = state.guidance_scale > 1.0
            cfg_topology = None
            if (
                apply_cfg
                and epe is not None
                and epe.cfg_parallel
                and ranks is not None
                and len(ranks) % 2 == 0
            ):
                cfg_topology = epe.parallel.cfg_parallel_topology(ranks)

            if cfg_topology is not None:
                if state.uncond_cap_feats is None:
                    raise RuntimeError("CFG state has no unconditional caption")
                is_conditional = cfg_topology.branch_index == 0
                cap_feats = (
                    state.cond_cap_feats if is_conditional else state.uncond_cap_feats
                )
                glm_cap_feats = (
                    state.cond_glm_cap_feats
                    if is_conditional
                    else state.uncond_glm_cap_feats
                )
                with epe.parallel.activate(cfg_topology.cp_ranks):
                    local_output = self._transformer_branch(
                        state,
                        timestep=model_timestep,
                        cap_feats=cap_feats,
                        glm_cap_feats=glm_cap_feats,
                    )
                conditional, unconditional = epe.parallel.all_gather_cfg(
                    local_output,
                    cfg_topology,
                )
                conditional = -conditional.squeeze(2).float()
                unconditional = -unconditional.squeeze(2).float()
                model_output = unconditional + state.guidance_scale * (
                    conditional - unconditional
                )
            elif apply_cfg:
                if epe is None or epe.parallel.active.width == 1:
                    conditional, unconditional = self._transformer_cfg_batch(
                        state,
                        timestep=model_timestep,
                    )
                else:
                    if state.uncond_cap_feats is None:
                        raise RuntimeError("CFG state has no unconditional caption")
                    conditional = self._transformer_branch(
                        state,
                        timestep=model_timestep,
                        cap_feats=state.cond_cap_feats,
                        glm_cap_feats=state.cond_glm_cap_feats,
                    )
                    unconditional = self._transformer_branch(
                        state,
                        timestep=model_timestep,
                        cap_feats=state.uncond_cap_feats,
                        glm_cap_feats=state.uncond_glm_cap_feats,
                    )
                conditional = -conditional.squeeze(2).float()
                unconditional = -unconditional.squeeze(2).float()
                model_output = unconditional + state.guidance_scale * (
                    conditional - unconditional
                )
            else:
                model_output = self._transformer_branch(
                    state,
                    timestep=model_timestep,
                    cap_feats=state.cond_cap_feats,
                    glm_cap_feats=state.cond_glm_cap_feats,
                )
                model_output = -model_output.squeeze(2).float()

            state.latents = state.scheduler.step(
                model_output,
                timestep_value,
                state.latents,
                return_dict=False,
            )[0]
            state.step_index += 1
        return state

    def synchronize_state(
        self,
        state: LLaDAImageDenoiseState,
        *,
        step_index: int,
    ) -> None:
        state.step_index = int(step_index)
        state.scheduler._step_index = int(step_index)

    @torch.inference_mode()
    def decode_request(
        self,
        state: LLaDAImageDenoiseState,
        *,
        topology: ActiveLaneTopology | None = None,
        parallel_vae: bool = False,
        vae_parallel_halo: int = 8,
        timings: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        if not state.complete:
            raise RuntimeError("cannot decode an incomplete denoise state")

        profile = timings if timings is not None else {}
        device = state.latents.device
        latents = state.latents.to(device=self.vae.device, dtype=self.vae.dtype)
        latent_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
        latent_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents)
        latents = self._unpatchify_latents(latents * latent_std + latent_mean)
        profile["vae_start_unix_ns"] = time.time_ns()
        started = time.perf_counter()

        def decode_fn(value: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(value, return_dict=False)[0]

        if topology is None:
            epe = getattr(self.transformer, "epe", None)
            topology = epe.parallel.active if epe is not None else None
        if parallel_vae and topology is not None and topology.width > 1:
            warnings.warn(
                "LLaDA-Image parallel VAE decode is approximate because "
                "AutoencoderKLFlux2 contains global normalization and attention",
                RuntimeWarning,
                stacklevel=2,
            )
        if topology is None:
            images = decode_fn(latents)
        else:
            images = parallel_tiled_vae_decode(
                latents,
                decode_fn,
                topology=topology,
                latent_split_dim=2,
                pixel_split_dim=2,
                scale=int(self.vae_scale_factor),
                halo=vae_parallel_halo,
                enabled=parallel_vae,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile["vae_end_unix_ns"] = time.time_ns()
        profile["vae_decode_ms"] = (time.perf_counter() - started) * 1000.0
        return images

    @torch.inference_mode()
    def finalize_request(
        self,
        state: LLaDAImageDenoiseState,
        *,
        output_type: str = "pil",
        return_dict: bool = True,
        timings: dict[str, Any] | None = None,
    ):
        if not state.complete:
            raise RuntimeError("cannot finalize an incomplete denoise state")
        if output_type == "latent":
            images = state.latents
        else:
            images = self.decode_request(state, timings=timings)
            if images is not None:
                images = self.image_processor.postprocess(
                    images,
                    output_type=output_type,
                )
        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        from .pipeline import LLaDAImagePipelineOutput

        return LLaDAImagePipelineOutput(images=images)

    def close(self) -> None:
        epe = getattr(self.transformer, "epe", None)
        if epe is not None:
            epe.parallel.close()
