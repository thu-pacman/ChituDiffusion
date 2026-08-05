from __future__ import annotations

import os
from logging import getLogger
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from chitu_diffusion.core.distributed.parallel_state import get_cp_group
from chitu_diffusion.models.parallel import ModelParallelCapabilities
from chitu_diffusion.modules.utils.flux import (
    compute_flux2_empirical_mu,
    flowmatch_sigmas,
    prepare_flux2_latents,
    retrieve_timesteps as retrieve_flowmatch_timesteps,
    unpack_flux2_latents_with_ids,
    unpatchify_flux2_latents,
)
from chitu_diffusion.parallel.state import ContextParallelLatentState, ParallelTaskState
from chitu_diffusion.parallel.vae import parallel_tiled_vae_decode
from chitu_diffusion.runtime.adapter.base import as_list, device_scope, register_model_runtime, set_cfg_type
from chitu_diffusion.runtime.adapter.flux1 import FluxRuntimeAdapter
from chitu_diffusion.runtime.parallel_utils import SequencePadder

logger = getLogger(__name__)

# Sequence-padder name used for the persistent latent shard. Must match the name
# the generic CP wrapper uses for image tokens so gather trims the same padding.
_CP_LATENT_NAME = "hidden_states"


@register_model_runtime("flux2-klein", "FLUX.2-klein-4B")
class Flux2RuntimeAdapter(FluxRuntimeAdapter):
    def parallel_capabilities(self, args: Any) -> ModelParallelCapabilities:
        # Flux2-klein uses the generic CP path (it does not own attention CP), but
        # it can keep the latent shard local across scheduler steps (parallel
        # sampler), so it advertises sampler_local_latent. With only 4 denoise
        # steps the DiT is tiny, so the relative weight of the per-step latent
        # gather is larger here than on a 50-step model.
        return ModelParallelCapabilities(
            dit_context_parallel=True,
            dit_cfg_parallel=False,
            vae_tile_parallel=True,
            sampler_local_latent=True,
            model_specific_context_parallel=False,
        )

    def _persistent_cp_latents_enabled(self, backend) -> bool:
        if os.getenv("CHITU_FLUX2_PERSISTENT_CP_LATENTS", "1") == "0":
            return False
        if int(getattr(backend.args.infer.diffusion, "cp_size", 1)) <= 1:
            return False
        plan = getattr(backend, "parallel_plan", None)
        sampler_plan = getattr(plan, "sampler", None)
        if sampler_plan is not None and not bool(getattr(sampler_plan, "enabled", False)):
            return False
        # FlexCache caches full-latent blocks/steps; keep latents replicated then.
        if getattr(getattr(backend, "flexcache", None), "strategy", None) is not None:
            return False
        return True

    @staticmethod
    def _local_cp_state(task) -> Optional[ContextParallelLatentState]:
        return getattr(getattr(task.buffer, "parallel", None), "cp_latents", None)

    @staticmethod
    def _split_local_latents(latents: torch.Tensor) -> torch.Tensor:
        group = get_cp_group()
        pieces = SequencePadder.split_sequence_padding(
            latents, group.group_size, split_dim=1, name=_CP_LATENT_NAME
        )
        return pieces[group.rank_in_group]

    @staticmethod
    def _gather_local_latents(latents: torch.Tensor) -> torch.Tensor:
        group = get_cp_group()
        pieces = [torch.empty_like(latents) for _ in range(group.group_size)]
        dist.all_gather(pieces, latents.contiguous(), group=group.gpu_group)
        return SequencePadder.remove_sequence_padding_and_concat(
            pieces, gather_dim=1, name=_CP_LATENT_NAME
        )
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
        generator._configure_flexcache_for_task(task)

        cp_local = False
        use_local = self._persistent_cp_latents_enabled(backend)
        cp_dispatcher = getattr(generator, "cp_dispatcher", None)
        if cp_dispatcher is not None:
            cp_dispatcher.local_latent_mode = use_local
        if use_local:
            full_seq_len = int(latents.shape[1])
            local_latents = self._split_local_latents(latents)
            if getattr(task.buffer, "parallel", None) is None:
                task.buffer.parallel = ParallelTaskState()
            task.buffer.parallel.cp_latents = ContextParallelLatentState(
                image_offset=0,
                image_seq_len=full_seq_len,
                is_local=True,
            )
            task.buffer.latents = local_latents
            latents = local_latents
            cp_local = True

        logger.info(
            "[Pre Denoise FLUX2] latents=%s latent_ids=%s steps=%s mu=%.4f cp_local=%s",
            tuple(latents.shape),
            tuple(latent_ids.shape),
            len(timesteps),
            mu,
            cp_local,
        )

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
        # Parallel-sampler path keeps only the rank-local latent shard during
        # denoise; gather once here before unpack / parallel VAE decode.
        cp_state = self._local_cp_state(task)
        if cp_state is not None and cp_state.is_local:
            task.buffer.latents = self._gather_local_latents(task.buffer.latents)
            cp_state.is_local = False
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
