from __future__ import annotations

import os
from logging import getLogger
from typing import Any, Callable, Optional

import torch

from chitu_diffusion.modules.utils.flux import (
    compute_flux2_empirical_mu,
    flowmatch_sigmas,
    prepare_flux2_latents,
    retrieve_timesteps as retrieve_flowmatch_timesteps,
    unpack_flux2_latents_with_ids,
    unpatchify_flux2_latents,
)
from chitu_diffusion.runtime.adapter.base import as_list, device_scope, register_model_runtime, set_cfg_type
from chitu_diffusion.runtime.adapter.flux1 import FluxRuntimeAdapter
from chitu_diffusion.runtime.parallel_vae import parallel_tiled_vae_decode

logger = getLogger(__name__)


@register_model_runtime("flux2-klein", "FLUX.2-klein-4B")
class Flux2RuntimeAdapter(FluxRuntimeAdapter):
    def load_text_encoder(self, args: Any, init_device: torch.device):
        from chitu_diffusion.modules.encoders.qwen3 import Qwen3CausalLMTextEncoder

        logger.info("Initializing Qwen3 text encoder for %s", args.models.name)
        text_encoder = Qwen3CausalLMTextEncoder(
            model_path=args.models.ckpt_dir,
            device=init_device,
            dtype=torch.bfloat16,
        )
        logger.info("Initialized Qwen3 text encoder for %s", args.models.name)
        return text_encoder

    def load_vae(self, args: Any, init_device: torch.device):
        from diffusers.models import AutoencoderKLFlux2

        logger.info("Initializing Flux2-klein VAE for %s", args.models.name)
        vae = AutoencoderKLFlux2.from_pretrained(
            args.models.ckpt_dir,
            subfolder=args.models.vae.checkpoint,
            torch_dtype=torch.bfloat16,
        ).to(init_device)
        vae.eval().requires_grad_(False)
        logger.info("Initialized Flux2-klein VAE for %s", args.models.name)
        return vae

    def checkpoint_paths(self, args: Any) -> list[str]:
        explicit = as_list(getattr(args.models, "checkpoints", None))
        if explicit:
            return [os.path.join(args.models.ckpt_dir, str(item)) for item in explicit]
        return [os.path.join(args.models.ckpt_dir, args.models.transformer_checkpoint)]

    def encode_text(self, task, generator, backend) -> torch.Tensor:
        payload = task.req.get_prompt()
        set_cfg_type(backend, "pos")
        layers = tuple(getattr(backend.args.models.encoder, "hidden_states_layers", (9, 18, 27)))
        with device_scope(backend.text_encoder, backend):
            prompt_embeds, text_ids = backend.text_encoder.encode(
                payload,
                max_sequence_length=getattr(backend.args.models.encoder, "max_sequence_length", 512),
                hidden_states_layers=layers,
                device=torch.device(torch.cuda.current_device()),
            )
        task.buffer.text_ids = text_ids
        logger.info(
            "[text_encode_step] FLUX2 prompt_embeds=%s text_ids=%s",
            tuple(prompt_embeds.shape),
            tuple(text_ids.shape),
        )
        return prompt_embeds

    def prepare_denoise(self, task, generator, backend) -> None:
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        device = torch.cuda.current_device()
        width, height = task.req.params.size
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(task.req.params.seed)
        latents, latent_ids = prepare_flux2_latents(
            batch_size=1,
            num_latents_channels=backend.args.models.transformer.in_channels // 4,
            height=height,
            width=width,
            vae_scale_factor=int(getattr(backend.args.models.vae, "scale_factor", 8)),
            dtype=task.buffer.text_embeddings.dtype,
            device=device,
            generator=seed_g,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            backend.args.models.ckpt_dir,
            subfolder="scheduler",
        )
        sigmas = flowmatch_sigmas(task.req.params.num_inference_steps)
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas = None
        mu = compute_flux2_empirical_mu(
            image_seq_len=latents.shape[1],
            num_steps=task.req.params.num_inference_steps,
        )
        timesteps, _ = retrieve_flowmatch_timesteps(
            scheduler,
            task.req.params.num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        scheduler.set_begin_index(0)

        task.buffer.seed_g = seed_g
        task.buffer.sampler = scheduler
        task.buffer.latents = latents
        task.buffer.latent_image_ids = latent_ids
        task.buffer.timesteps = timesteps
        task.buffer.image_size = (width, height)

        backend.switch_active_model(flush=True)
        logger.info(
            "[Pre Denoise FLUX2] latents=%s latent_ids=%s steps=%s mu=%.4f",
            tuple(latents.shape),
            tuple(latent_ids.shape),
            len(timesteps),
            mu,
        )
        generator._configure_flexcache_for_task(task)

    def denoise_step(self, task, generator, backend, run_dit_forward: Callable[..., torch.Tensor]) -> torch.Tensor:
        x = task.buffer.latents.to(backend.active_model.dtype)
        t_curr = task.buffer.timesteps[task.buffer.current_step]
        t_vec = t_curr.expand(x.shape[0]).to(x.dtype)

        pred = run_dit_forward(
            task,
            "pos",
            hidden_states=x,
            timestep=t_vec / 1000,
            guidance=None,
            encoder_hidden_states=task.buffer.text_embeddings,
            txt_ids=task.buffer.text_ids,
            img_ids=task.buffer.latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        pred = pred[:, : task.buffer.latents.size(1)]
        return task.buffer.sampler.step(pred, t_curr, task.buffer.latents, return_dict=False)[0]

    def decode_latents(self, task, generator, backend) -> Optional[torch.Tensor]:
        # Latents are replicated on every rank after denoise, so all ranks prepare
        # them identically and split the VAE decode spatially (parallel_vae). For a
        # 4-step distilled model the DiT is tiny, so VAE decode dominates end-to-end
        # latency -- this is where parallel decode pays off most. Falls back to
        # rank-0-only decode when disabled or single-GPU.
        latents = unpack_flux2_latents_with_ids(task.buffer.latents, task.buffer.latent_image_ids)
        latents_bn_mean = backend.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            backend.vae.bn.running_var.view(1, -1, 1, 1) + backend.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = unpatchify_flux2_latents(latents)

        def _decode(z: torch.Tensor) -> torch.Tensor:
            with device_scope(backend.vae, backend):
                return backend.vae.decode(z, return_dict=False)[0]

        # latents: [B, C, H_lat, W_lat] -> split on H_lat (dim 2);
        # decoded pixels: [B, 3, H, W] -> H is dim 2; AutoencoderKLFlux2 upsamples 8x.
        return parallel_tiled_vae_decode(
            latents,
            _decode,
            latent_split_dim=2,
            pixel_split_dim=2,
            scale=8,
        )
