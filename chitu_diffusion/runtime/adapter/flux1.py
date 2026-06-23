from __future__ import annotations

import os
from logging import getLogger
from typing import Any, Callable, Optional

import torch

from chitu_diffusion.modules.utils.flux import (
    calculate_flux1_shift,
    flowmatch_sigmas,
    prepare_flux1_latents,
    retrieve_timesteps as retrieve_flowmatch_timesteps,
    unpack_flux1_latents,
)
from chitu_diffusion.models.parallel import ModelParallelCapabilities
from chitu_diffusion.parallel.vae import parallel_tiled_vae_decode
from chitu_diffusion.runtime.adapter.base import (
    DiffusionRuntimeAdapter,
    as_list,
    device_scope,
    register_model_runtime,
    set_cfg_type,
)

logger = getLogger(__name__)


class FluxRuntimeAdapter(DiffusionRuntimeAdapter):
    def supports_cfg(self, args: Any) -> bool:
        return False

    def parallel_capabilities(self, args: Any) -> ModelParallelCapabilities:
        return ModelParallelCapabilities(
            dit_context_parallel=True,
            dit_cfg_parallel=False,
            vae_tile_parallel=True,
            sampler_local_latent=False,
            model_specific_context_parallel=False,
        )

    def load_vae(self, args: Any, init_device: torch.device):
        raise NotImplementedError

    def save_output(self, task, output: Optional[torch.Tensor], generator, backend) -> None:
        if torch.distributed.get_rank() == 0:
            generator._save_image(task, output)


@register_model_runtime("flux1-dev", "FLUX.1-dev")
class Flux1RuntimeAdapter(FluxRuntimeAdapter):
    @staticmethod
    def _normalize_flux_tokens(tensor: torch.Tensor, name: str) -> torch.Tensor:
        while tensor.ndim > 3 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim != 3:
            raise ValueError(f"FLUX1 {name} must be [batch, sequence, channels], got {tuple(tensor.shape)}.")
        return tensor

    def load_text_encoder(self, args: Any, init_device: torch.device):
        from chitu_diffusion.modules.encoders.clip_t5 import CLIPT5TextEncoder

        logger.info("Initializing CLIP+T5 text encoders for %s", args.models.name)
        text_encoder = CLIPT5TextEncoder(
            model_path=args.models.ckpt_dir,
            device=init_device,
            dtype=torch.bfloat16,
        )
        logger.info("Initialized CLIP+T5 text encoders for %s", args.models.name)
        return text_encoder

    def load_vae(self, args: Any, init_device: torch.device):
        from diffusers import AutoencoderKL

        logger.info("Initializing Flux.1 VAE for %s", args.models.name)
        vae = AutoencoderKL.from_pretrained(
            args.models.ckpt_dir,
            subfolder=args.models.vae.checkpoint,
            torch_dtype=torch.bfloat16,
        ).to(init_device)
        vae.eval().requires_grad_(False)
        logger.info("Initialized Flux.1 VAE for %s", args.models.name)
        return vae

    def checkpoint_paths(self, args: Any) -> list[str]:
        explicit = as_list(getattr(args.models, "checkpoints", None))
        if explicit:
            return [os.path.join(args.models.ckpt_dir, str(item)) for item in explicit]
        return [os.path.join(args.models.ckpt_dir, "transformer")]

    def encode_text(self, task, generator, backend) -> torch.Tensor:
        payload = task.req.get_prompt()
        set_cfg_type(backend, "pos")
        with device_scope(backend.text_encoder, backend):
            prompt_embeds, pooled_prompt_embeds, text_ids = backend.text_encoder.encode(
                payload,
                max_sequence_length=getattr(backend.args.models.encoder, "max_sequence_length", 512),
                device=torch.device(torch.cuda.current_device()),
            )
        task.buffer.pooled_prompt_embeds = pooled_prompt_embeds
        task.buffer.text_ids = text_ids
        logger.info(
            "[text_encode_step] FLUX1 prompt_embeds=%s pooled=%s text_ids=%s",
            tuple(prompt_embeds.shape),
            tuple(pooled_prompt_embeds.shape),
            tuple(text_ids.shape),
        )
        return prompt_embeds

    def prepare_denoise(self, task, generator, backend) -> None:
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        device = torch.cuda.current_device()
        width, height = task.req.params.size
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(task.req.params.seed)
        latents, latent_image_ids = prepare_flux1_latents(
            batch_size=1,
            num_channels_latents=backend.args.models.transformer.in_channels // 4,
            height=height,
            width=width,
            vae_scale_factor=16,
            dtype=task.buffer.text_embeddings.dtype,
            device=device,
            generator=seed_g,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            backend.args.models.ckpt_dir,
            subfolder="scheduler",
        )
        sigmas = flowmatch_sigmas(task.req.params.num_inference_steps)
        mu = calculate_flux1_shift(
            latents.shape[1],
            scheduler.config.base_image_seq_len,
            scheduler.config.max_image_seq_len,
            scheduler.config.base_shift,
            scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_flowmatch_timesteps(
            scheduler,
            task.req.params.num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        guidance = backend.args.models.sampler.guidance_scale[0]
        guidance_vec = torch.full((latents.shape[0],), guidance, device=device, dtype=torch.float32)

        task.buffer.seed_g = seed_g
        task.buffer.sampler = scheduler
        task.buffer.latents = latents
        task.buffer.latent_image_ids = latent_image_ids
        task.buffer.timesteps = timesteps
        task.buffer.guidance_vec = guidance_vec
        task.buffer.image_size = (width, height)

        backend.switch_active_model(flush=True)
        logger.info(
            "[Pre Denoise FLUX1] latents=%s img_ids=%s steps=%s guidance=%s",
            tuple(latents.shape),
            tuple(latent_image_ids.shape),
            len(timesteps),
            guidance,
        )
        generator._configure_flexcache_for_task(task)

    def denoise_step(self, task, generator, backend, run_dit_forward: Callable[..., torch.Tensor]) -> torch.Tensor:
        latents = task.buffer.latents
        timestep = task.buffer.timesteps[task.buffer.current_step]
        guidance = task.buffer.guidance_vec
        if guidance is not None:
            guidance = guidance.to(device=latents.device, dtype=torch.float32)
        timestep_vec = timestep.expand(latents.shape[0]).to(latents.dtype)
        flex_strategy = getattr(getattr(backend, "flexcache", None), "strategy", None)
        meancache_strategy = flex_strategy if getattr(flex_strategy, "type", None) == "meancache" else None
        step_index = int(task.buffer.current_step)
        sigma_pre = None
        sigma_next = None
        if meancache_strategy is not None:
            sigmas = getattr(task.buffer.sampler, "sigmas", None)
            if sigmas is not None and step_index + 1 < len(sigmas):
                sigma_pre = sigmas[step_index]
                sigma_next = sigmas[step_index + 1]
            reuse_key = meancache_strategy.get_reuse_key(step=step_index)
            if reuse_key is not None:
                noise_pred = meancache_strategy.reuse(step=step_index).to(device=latents.device, dtype=latents.dtype)
                noise_pred = self._normalize_flux_tokens(noise_pred, "meancache reuse noise_pred")
                backend.flexcache.record_compute(
                    baseline_units=1.0,
                    actual_units=0.0,
                    task_id=task.task_id,
                    scope="meancache_step",
                    unit="dit_forward",
                    extra={"decision": "reuse", "step": step_index},
                )
                latents = task.buffer.sampler.step(noise_pred, timestep, latents, return_dict=False)[0]
                return self._normalize_flux_tokens(latents, "scheduler output")

        latents_pre = latents.detach()
        noise_pred = run_dit_forward(
            task,
            "pos",
            hidden_states=latents,
            timestep=timestep_vec / 1000,
            guidance=guidance,
            pooled_projections=task.buffer.pooled_prompt_embeds,
            encoder_hidden_states=task.buffer.text_embeddings,
            txt_ids=task.buffer.text_ids,
            img_ids=task.buffer.latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        noise_pred = self._normalize_flux_tokens(noise_pred, "fresh noise_pred")
        latents = task.buffer.sampler.step(noise_pred, timestep, latents, return_dict=False)[0]
        latents = self._normalize_flux_tokens(latents, "scheduler output")
        if meancache_strategy is not None:
            if sigma_pre is None or sigma_next is None:
                raise RuntimeError("MeanCache requires scheduler sigmas for Flux1-dev.")
            meancache_strategy.store(
                step=step_index,
                latents_pre=latents_pre,
                latents=latents,
                sigma_pre=sigma_pre,
                sigma=sigma_next,
                noise_pred=noise_pred,
            )
            backend.flexcache.record_compute(
                baseline_units=1.0,
                actual_units=1.0,
                task_id=task.task_id,
                scope="meancache_step",
                unit="dit_forward",
                extra={"decision": "compute", "step": step_index},
            )
        return latents

    def decode_latents(self, task, generator, backend) -> Optional[torch.Tensor]:
        # Latents are replicated on every rank after denoise, so all ranks prepare
        # them identically and split the VAE decode spatially (parallel_vae); falls
        # back to rank-0-only decode when disabled or single-GPU.
        width, height = task.buffer.image_size or task.req.params.size
        latents = unpack_flux1_latents(
            task.buffer.latents,
            height=height,
            width=width,
            vae_scale_factor=16,
        )
        latents = (latents / backend.vae.config.scaling_factor) + backend.vae.config.shift_factor

        def _decode(z: torch.Tensor) -> torch.Tensor:
            with device_scope(backend.vae, backend):
                return backend.vae.decode(z, return_dict=False)[0]

        # latents: [B, C, H_lat, W_lat] -> split on H_lat (dim 2);
        # decoded pixels: [B, 3, H, W] -> H is dim 2; the AutoencoderKL upsamples 8x.
        return parallel_tiled_vae_decode(
            latents,
            _decode,
            latent_split_dim=2,
            pixel_split_dim=2,
            scale=8,
        )
