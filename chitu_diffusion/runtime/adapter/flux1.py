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

    def load_vae(self, args: Any, init_device: torch.device):
        raise NotImplementedError

    def save_output(self, task, output: Optional[torch.Tensor], generator, backend) -> None:
        if torch.distributed.get_rank() == 0:
            generator._save_image(task, output)


@register_model_runtime("flux1-dev", "FLUX.1-dev")
class Flux1RuntimeAdapter(FluxRuntimeAdapter):
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
        return task.buffer.sampler.step(noise_pred, timestep, latents, return_dict=False)[0]

    def decode_latents(self, task, generator, backend) -> Optional[torch.Tensor]:
        if torch.distributed.get_rank() != 0:
            return None
        width, height = task.buffer.image_size or task.req.params.size
        latents = unpack_flux1_latents(
            task.buffer.latents,
            height=height,
            width=width,
            vae_scale_factor=16,
        )
        latents = (latents / backend.vae.config.scaling_factor) + backend.vae.config.shift_factor
        with device_scope(backend.vae, backend):
            return backend.vae.decode(latents, return_dict=False)[0]
