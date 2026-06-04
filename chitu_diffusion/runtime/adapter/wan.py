from __future__ import annotations

import math
import os
from logging import getLogger
from typing import Any, Callable, Optional

import torch

from chitu_diffusion.core.distributed.parallel_state import get_cp_group
from chitu_diffusion.modules.samplers.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from chitu_diffusion.modules.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from chitu_diffusion.runtime.adapter.base import (
    DiffusionRuntimeAdapter,
    as_list,
    device_scope,
    register_model_runtime,
    set_cfg_type,
)

logger = getLogger(__name__)


@register_model_runtime("diffusion-wan", "Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B", "Wan2.2-T2V-A14B")
class WanRuntimeAdapter(DiffusionRuntimeAdapter):
    def rope_impl(self, args: Any):
        if args.infer.diffusion.cp_size > 1:
            from functools import partial

            from chitu_diffusion.modules.utils.wan import rope_apply_with_cp

            return partial(
                rope_apply_with_cp,
                cp_size=get_cp_group().group_size,
                cp_rank=get_cp_group().rank_in_group,
            )
        return None

    def load_text_encoder(self, args: Any, init_device: torch.device):
        from chitu_diffusion.modules.encoders.t5 import T5EncoderModel

        logger.info("Initializing T5 encoder for %s", args.models.name)
        text_encoder = T5EncoderModel(
            text_len=args.models.encoder.text_len,
            device=init_device,
            checkpoint_path=os.path.join(args.models.ckpt_dir, args.models.encoder.t5_checkpoint),
            tokenizer_path=os.path.join(args.models.ckpt_dir, args.models.encoder.t5_tokenizer),
        )
        logger.info("Initialized T5 encoder for %s", args.models.name)
        return text_encoder

    def load_vae(self, args: Any, init_device: torch.device):
        from chitu_diffusion.modules.vaes.wan_vae import WanVAE

        logger.info("Initializing Wan VAE for %s", args.models.name)
        vae = WanVAE(
            vae_pth=os.path.join(args.models.ckpt_dir, args.models.vae.checkpoint),
            device=init_device,
        )
        logger.info("Initialized Wan VAE for %s", args.models.name)
        return vae

    def checkpoint_paths(self, args: Any) -> list[str]:
        explicit = as_list(getattr(args.models, "checkpoints", None))
        if explicit:
            return [os.path.join(args.models.ckpt_dir, str(item)) for item in explicit]
        if hasattr(args.models, "high_noise_checkpoint") and hasattr(args.models, "low_noise_checkpoint"):
            return [
                os.path.join(args.models.ckpt_dir, args.models.high_noise_checkpoint),
                os.path.join(args.models.ckpt_dir, args.models.low_noise_checkpoint),
            ]
        return [os.path.join(args.models.ckpt_dir, "diffusion_pytorch_model.safetensors")]

    def encode_text(self, task, generator, backend) -> torch.Tensor:
        if generator.cfg_size == 1:
            if task.buffer.text_embeddings is None:
                payload = task.req.get_prompt()
                set_cfg_type(backend, "pos")
            else:
                payload = task.req.get_n_prompt()
                set_cfg_type(backend, "neg")
        elif generator.cfg_size == 2:
            if generator.cfg_dispatcher.group.rank_in_group == 0:
                payload = task.req.get_prompt()
                set_cfg_type(backend, "pos")
            else:
                payload = task.req.get_n_prompt()
                set_cfg_type(backend, "neg")
        else:
            raise ValueError(f"Unsupported cfg_size={generator.cfg_size}.")

        with device_scope(backend.text_encoder.model, backend):
            out = backend.text_encoder(payload, torch.cuda.current_device())
        logger.debug("[text_encode_step] context shape: %s", tuple(out.shape))
        return out

    def prepare_denoise(self, task, generator, backend) -> None:
        device = torch.cuda.current_device()
        if task.buffer.latents is None:
            frames = task.req.params.frame_num
            size = task.req.params.size
            vae_stride = backend.args.models.vae.stride
            target_shape = (
                backend.vae.model.z_dim,
                (frames - 1) // vae_stride[0] + 1,
                size[1] // vae_stride[1],
                size[0] // vae_stride[2],
            )
            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(task.req.params.seed)
            latents = torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=device,
                generator=seed_g,
            )
            patch_size = backend.args.models.transformer.patch_size
            seq_len = (
                math.ceil(
                    (target_shape[2] * target_shape[3])
                    / (patch_size[1] * patch_size[2])
                    * target_shape[1]
                    / generator.cp_size
                )
                * generator.cp_size
            )
        else:
            latents = task.buffer.latents
            seed_g = task.buffer.seed_g
            seq_len = task.buffer.seq_len

        if task.req.params.sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=backend.args.models.sampler.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(
                task.req.params.num_inference_steps,
                device=device,
                shift=backend.args.models.sampler.sample_shift,
            )
            timesteps = sample_scheduler.timesteps
        elif task.req.params.sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=backend.args.models.sampler.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(
                task.req.params.num_inference_steps,
                backend.args.models.sampler.sample_shift,
            )
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        task.buffer.sampler = sample_scheduler
        task.buffer.seed_g = seed_g
        task.buffer.latents = latents
        task.buffer.timesteps = timesteps
        task.buffer.seq_len = seq_len

        backend.switch_active_model(flush=True)
        generator._configure_flexcache_for_task(task)

    def denoise_step(self, task, generator, backend, run_dit_forward: Callable[..., torch.Tensor]) -> torch.Tensor:
        latent_model_input = task.buffer.latents
        timestep = task.buffer.timesteps[task.buffer.current_step]

        if backend.guidance_scale > 0:
            if generator.cfg_size == 2:
                if generator.cfg_dispatcher.group.rank_in_group == 0:
                    context = task.buffer.text_embeddings
                    set_cfg_type(backend, "pos")
                else:
                    context = task.buffer.negative_embeddings
                    set_cfg_type(backend, "neg")
                cfg_branch = "pos" if backend.cfg_type.value == "pos" else "neg"
                cfg_partial_noise_pred = run_dit_forward(
                    task,
                    cfg_branch,
                    latent_model_input,
                    t=timestep,
                    context=context,
                    seq_len=task.buffer.seq_len,
                )
                noise_pred_cond, noise_pred_uncond = generator.cfg_dispatcher.all_gather_cfg_noise_preds(
                    cfg_partial_noise_pred
                )
            else:
                set_cfg_type(backend, "pos")
                noise_pred_cond = run_dit_forward(
                    task,
                    "pos",
                    latent_model_input,
                    t=timestep,
                    context=task.buffer.text_embeddings,
                    seq_len=task.buffer.seq_len,
                )
                set_cfg_type(backend, "neg")
                noise_pred_uncond = run_dit_forward(
                    task,
                    "neg",
                    latent_model_input,
                    t=timestep,
                    context=task.buffer.negative_embeddings,
                    seq_len=task.buffer.seq_len,
                )
            noise_pred = noise_pred_uncond + backend.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            set_cfg_type(backend, "pos")
            noise_pred = run_dit_forward(
                task,
                "pos",
                latent_model_input,
                t=timestep,
                context=task.buffer.text_embeddings,
                seq_len=task.buffer.seq_len,
            )

        cache_strategy = getattr(backend.flexcache, "strategy", None)
        observe_guided_output = getattr(cache_strategy, "observe_guided_output", None)
        if observe_guided_output is not None:
            observe_guided_output(
                x=latent_model_input,
                output=noise_pred,
                t=timestep,
                step=int(task.buffer.current_step),
            )

        return task.buffer.sampler.step(
            noise_pred.unsqueeze(0),
            timestep,
            task.buffer.latents.unsqueeze(0),
            return_dict=False,
            generator=task.buffer.seed_g,
        )[0].squeeze(0)

    def decode_latents(self, task, generator, backend) -> Optional[torch.Tensor]:
        target_decode_device = 0
        if torch.distributed.get_rank() != target_decode_device:
            return None
        with device_scope(backend.vae.model, backend):
            return backend.vae.decode([task.buffer.latents])[0]

    def save_output(self, task, output: Optional[torch.Tensor], generator, backend) -> None:
        if torch.distributed.get_rank() == 0:
            generator._save_video(task, output)
