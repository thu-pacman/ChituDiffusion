from __future__ import annotations

import contextlib
import json
import os
from logging import getLogger
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from chitu_diffusion.core.logging_utils import log_result
from chitu_diffusion.flexcache.freecache_core import is_step_level_cache_strategy
from chitu_diffusion.models.parallel import ModelParallelCapabilities
from chitu_diffusion.runtime.output_layout import task_results_dir
from chitu_diffusion.runtime.adapter.base import (
    DiffusionRuntimeAdapter,
    device_scope,
    register_model_runtime,
    set_cfg_type,
)

logger = getLogger(__name__)


@register_model_runtime("z-image", "Z-Image")
class ZImageRuntimeAdapter(DiffusionRuntimeAdapter):
    def __init__(self, spec):
        super().__init__(spec)
        self.pipeline = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self._pipeline_device = None
        self._configured_attn_backend = None

    def supports_cfg(self, args: Any) -> bool:
        return float(args.models.sampler.guidance_scale[0]) > 1.0

    def parallel_capabilities(self, args: Any) -> ModelParallelCapabilities:
        return ModelParallelCapabilities(
            dit_context_parallel=False,
            dit_cfg_parallel=self.supports_cfg(args),
            vae_tile_parallel=True,
            sampler_local_latent=False,
            model_specific_context_parallel=False,
        )

    def handles_context_parallel(self, args: Any) -> bool:
        return False

    def denoise_completes_in_single_call(self) -> bool:
        return True

    def _torch_dtype(self, args: Any) -> torch.dtype:
        variant = str(getattr(args, "float_16bit_variant", "bfloat16")).lower()
        if variant in {"float16", "fp16", "half"}:
            return torch.float16
        return torch.bfloat16

    @contextlib.contextmanager
    def _float32_default_dtype(self):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            yield
        finally:
            torch.set_default_dtype(old_dtype)

    def _ensure_pipeline(self, args: Any, device: torch.device):
        if self.pipeline is not None:
            low_mem = int(getattr(getattr(args.infer, "diffusion", None), "low_mem_level", 0) or 0)
            if low_mem > 0 or self._pipeline_device != device:
                self.pipeline.to(device)
                self._pipeline_device = device
            return self.pipeline

        if self.text_encoder is None or self.vae is None or self.transformer is None:
            raise RuntimeError("Z-Image components are not initialized.")

        from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
        from transformers import Qwen2Tokenizer

        with self._float32_default_dtype():
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                args.models.ckpt_dir,
                subfolder="scheduler",
                local_files_only=True,
            )
        tokenizer = Qwen2Tokenizer.from_pretrained(
            args.models.ckpt_dir,
            subfolder="tokenizer",
            local_files_only=True,
        )
        pipe = ZImagePipeline(
            scheduler=scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=tokenizer,
            transformer=self.transformer,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe._joint_attention_kwargs = {}
        self.pipeline = pipe
        self._pipeline_device = device
        logger.info("Initialized Z-Image stage helper pipeline.")
        return pipe

    def _debug_tensor(self, name: str, tensor: torch.Tensor, *, step: Optional[int] = None, check: bool = True) -> None:
        if not isinstance(tensor, torch.Tensor):
            return
        debug_enabled = os.getenv("CHITU_Z_IMAGE_DEBUG_STATS", "").strip().lower() in {"1", "true", "yes", "on"}
        if not debug_enabled and not check:
            return
        finite = torch.isfinite(tensor)
        all_finite = bool(finite.all().item())
        if debug_enabled or not all_finite:
            stats_tensor = tensor.detach().float()
            prefix = f"[Z-Image stats] {name}"
            if step is not None:
                prefix += f" step={step}"
            logger.info(
                "%s shape=%s dtype=%s finite=%s min=%.6g max=%.6g mean=%.6g std=%.6g",
                prefix,
                tuple(tensor.shape),
                tensor.dtype,
                all_finite,
                float(torch.nan_to_num(stats_tensor, nan=0.0, posinf=0.0, neginf=0.0).min().item()),
                float(torch.nan_to_num(stats_tensor, nan=0.0, posinf=0.0, neginf=0.0).max().item()),
                float(torch.nan_to_num(stats_tensor, nan=0.0, posinf=0.0, neginf=0.0).mean().item()),
                float(torch.nan_to_num(stats_tensor, nan=0.0, posinf=0.0, neginf=0.0).std().item()),
            )
        if not all_finite:
            raise FloatingPointError(f"Z-Image tensor {name} contains non-finite values.")

    def load_text_encoder(self, args: Any, init_device: torch.device):
        from transformers import Qwen3Model

        logger.info("Initializing Z-Image Qwen3 text encoder from %s", args.models.ckpt_dir)
        with self._float32_default_dtype():
            self.text_encoder = Qwen3Model.from_pretrained(
                args.models.ckpt_dir,
                subfolder="text_encoder",
                dtype=self._torch_dtype(args),
                local_files_only=True,
            ).to(device=init_device, dtype=self._torch_dtype(args))
        self.text_encoder.eval().requires_grad_(False)
        return self.text_encoder

    def load_vae(self, args: Any, init_device: torch.device):
        from diffusers import AutoencoderKL

        logger.info("Initializing Z-Image VAE from %s", args.models.ckpt_dir)
        with self._float32_default_dtype():
            self.vae = AutoencoderKL.from_pretrained(
                args.models.ckpt_dir,
                subfolder="vae",
                torch_dtype=self._torch_dtype(args),
                local_files_only=True,
            ).to(device=init_device, dtype=self._torch_dtype(args))
        self.vae.eval().requires_grad_(False)
        return self.vae

    def checkpoint_paths(self, args: Any) -> list[str]:
        return []

    def loads_transformer_weights(self) -> bool:
        return True

    def build_transformer(self, models_config: Any, attn_backend, rope_impl) -> torch.nn.Module:
        from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

        variant = str(getattr(models_config, "float_16bit_variant", "") or torch.get_default_dtype()).lower()
        torch_dtype = torch.float16 if variant in {"float16", "fp16", "half", "torch.float16"} else torch.bfloat16
        with self._float32_default_dtype():
            transformer = ZImageTransformer2DModel.from_pretrained(
                models_config.ckpt_dir,
                subfolder="transformer",
                torch_dtype=torch_dtype,
                local_files_only=True,
            ).to(dtype=torch_dtype)
        if attn_backend is not None:
            self._install_attention_processor(transformer, attn_backend)
        transformer.eval().requires_grad_(False)
        self.transformer = transformer
        return transformer

    def _install_attention_processor(self, transformer, attn_backend) -> None:
        from chitu_diffusion.modules.attention.z_image_attention import ZImageChituAttnProcessor

        if os.getenv("CHITU_Z_IMAGE_USE_DIFFUSERS_ATTN", "").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info("Keeping diffusers native Z-Image attention processor.")
            return

        processor = ZImageChituAttnProcessor(attn_backend)
        for block in list(transformer.noise_refiner) + list(transformer.context_refiner) + list(transformer.layers):
            block.attention.set_processor(processor)
        self._configured_attn_backend = getattr(attn_backend, "impl", attn_backend.__class__.__name__)
        logger.info("Installed Z-Image Chitu attention processor: %s", self._configured_attn_backend)

    def configure_after_backend_build(self, backend) -> None:
        cp_size = int(getattr(backend.args.infer.diffusion, "cp_size", 1))
        if cp_size != 1:
            raise NotImplementedError("Z-Image adapter currently supports single-GPU/context-parallel size 1 only.")
        torch.set_default_dtype(torch.float32)
        logger.info("Z-Image runtime keeps torch default dtype at float32 for scheduler/sequence numerics.")

    def encode_text(self, task, generator, backend):
        device = torch.device(torch.cuda.current_device())
        pipe = self._ensure_pipeline(backend.args, device)
        guidance_scale = float(backend.args.models.sampler.guidance_scale[0])
        max_sequence_length = int(getattr(backend.args.models.encoder, "max_sequence_length", 512))
        negative_prompt = task.req.get_n_prompt() or ""

        if generator.cfg_size == 1:
            encode_negative = task.buffer.text_embeddings is not None and guidance_scale > 1.0
        elif generator.cfg_size == 2:
            encode_negative = generator.cfg_dispatcher.group.rank_in_group == 1
        else:
            raise ValueError(f"Unsupported cfg_size={generator.cfg_size}.")

        prompt = negative_prompt if encode_negative else task.req.get_prompt()
        set_cfg_type(backend, "neg" if encode_negative else "pos")
        embeds = pipe._encode_prompt(
            prompt=[prompt],
            device=device,
            max_sequence_length=max_sequence_length,
        )
        lengths = [int(item.shape[0]) for item in embeds]
        padded_embeds = torch.nn.utils.rnn.pad_sequence(embeds, batch_first=True, padding_value=0.0)
        if encode_negative:
            task.buffer._z_negative_embedding_lengths = lengths
        else:
            task.buffer._z_text_embedding_lengths = lengths
        logger.info(
            "[text_encode_step] Z-Image branch=%s embeds=%s",
            "neg" if encode_negative else "pos",
            [tuple(item.shape) for item in embeds],
        )
        return padded_embeds

    def prepare_denoise(self, task, generator, backend) -> None:
        from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, retrieve_timesteps

        device = torch.device(torch.cuda.current_device())
        pipe = self._ensure_pipeline(backend.args, device)
        width, height = task.req.params.size
        vae_scale = pipe.vae_scale_factor * 2
        if height % vae_scale != 0 or width % vae_scale != 0:
            raise ValueError(f"Z-Image height/width must be divisible by {vae_scale}, got {(width, height)}.")

        seed = int(task.req.params.seed if task.req.params.seed is not None else backend.args.infer.seed)
        seed_g = torch.Generator(device=device).manual_seed(seed)
        embeddings = task.buffer.text_embeddings
        if embeddings is None:
            embeddings = task.buffer.negative_embeddings
        if embeddings is None:
            raise RuntimeError("Z-Image denoise requires prompt embeddings.")
        batch_size = embeddings.shape[0] if isinstance(embeddings, torch.Tensor) else len(embeddings)
        num_inference_steps = int(task.req.params.num_inference_steps)

        latents = pipe.prepare_latents(
            batch_size,
            pipe.transformer.in_channels,
            height,
            width,
            torch.float32,
            device,
            seed_g,
            latents=None,
        )
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        pipe.scheduler.sigma_min = 0.0
        with self._float32_default_dtype():
            timesteps, _ = retrieve_timesteps(
                pipe.scheduler,
                num_inference_steps,
                device,
                mu=mu,
            )
        if getattr(pipe.scheduler, "sigmas", None) is not None:
            pipe.scheduler.sigmas = pipe.scheduler.sigmas.to(device=device, dtype=torch.float32)
        timesteps = timesteps.to(device=device, dtype=torch.float32)
        pipe._num_timesteps = len(timesteps)
        pipe._guidance_scale = float(backend.args.models.sampler.guidance_scale[0])
        pipe._cfg_normalization = bool(getattr(backend.args.models.sampler, "cfg_normalization", False))
        pipe._cfg_truncation = float(getattr(backend.args.models.sampler, "cfg_truncation", 1.0))

        task.buffer.seed_g = seed_g
        task.buffer.sampler = pipe.scheduler
        task.buffer.latents = latents
        task.buffer.timesteps = timesteps
        task.buffer.image_size = (width, height)
        self._debug_tensor("prepared_latents", latents)
        self._debug_tensor("timesteps", timesteps)

        backend.switch_active_model(flush=True)
        generator._configure_flexcache_for_task(task)
        logger.info(
            "[Pre Denoise Z-Image] latents=%s steps=%s mu=%.4f cfg=%.2f cfg_norm=%s",
            tuple(latents.shape),
            len(timesteps),
            mu,
            pipe.guidance_scale,
            pipe._cfg_normalization,
        )

    @staticmethod
    def _move_tensor_or_list_to_device(value, device: torch.device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
        return value

    def _move_task_tensors_to_device(self, task, device: torch.device) -> None:
        for name in (
            "text_embeddings",
            "negative_embeddings",
            "latents",
            "timesteps",
        ):
            setattr(task.buffer, name, self._move_tensor_or_list_to_device(getattr(task.buffer, name, None), device))

    @staticmethod
    def _embedding_tensor_to_list(tensor_or_list, lengths: Optional[list[int]]) -> list[torch.Tensor]:
        if isinstance(tensor_or_list, list):
            return tensor_or_list
        if not isinstance(tensor_or_list, torch.Tensor):
            raise TypeError(f"Z-Image prompt embeddings must be Tensor or list, got {type(tensor_or_list).__name__}.")
        if tensor_or_list.ndim == 2:
            return [tensor_or_list]
        if tensor_or_list.ndim != 3:
            raise ValueError(f"Z-Image prompt embeddings must be [B,S,D], got {tuple(tensor_or_list.shape)}.")
        if lengths is None:
            lengths = [tensor_or_list.shape[1]] * tensor_or_list.shape[0]
        return [tensor_or_list[idx, : int(length)] for idx, length in enumerate(lengths)]

    def _transformer_cache_context(self, cache_tag: str):
        if hasattr(self.transformer, "cache_context"):
            return self.transformer.cache_context(cache_tag)
        return contextlib.nullcontext()

    @staticmethod
    def _normalize_latents(latents: torch.Tensor, source: str) -> torch.Tensor:
        if latents.ndim == 4:
            return latents
        if latents.ndim == 5 and latents.shape[1] == 1:
            logger.debug("Normalizing Z-Image %s latents from %s via dim=1 squeeze.", source, tuple(latents.shape))
            return latents.squeeze(1)
        if latents.ndim == 5 and latents.shape[2] == 1:
            logger.debug("Normalizing Z-Image %s latents from %s via dim=2 squeeze.", source, tuple(latents.shape))
            return latents.squeeze(2)
        raise ValueError(f"Z-Image {source} latents must be [B,C,H,W], got {tuple(latents.shape)}.")

    def _transformer_forward(self, task, generator, backend, cache_tag: str, prompt_embeds: list[torch.Tensor]) -> torch.Tensor:
        device = torch.device(torch.cuda.current_device())
        self._move_task_tensors_to_device(task, device)
        latents = self._normalize_latents(task.buffer.latents, "input")
        task.buffer.latents = latents
        timestep = task.buffer.timesteps[task.buffer.current_step]
        timestep_vec = timestep.expand(latents.shape[0])
        timestep_vec = (1000 - timestep_vec) / 1000
        latent_model_input = latents.to(self.transformer.dtype).unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        lengths = (
            getattr(task.buffer, "_z_negative_embedding_lengths", None)
            if cache_tag == "uncond"
            else getattr(task.buffer, "_z_text_embedding_lengths", None)
        )
        prompt_embeds = self._embedding_tensor_to_list(prompt_embeds, lengths)
        self._debug_tensor("latent_model_input", latent_model_input, step=int(task.buffer.current_step), check=False)
        self._debug_tensor("timestep_model_input", timestep_vec, step=int(task.buffer.current_step), check=False)
        for idx, item in enumerate(prompt_embeds[:2]):
            self._debug_tensor(f"prompt_embeds[{idx}]", item, step=int(task.buffer.current_step), check=False)

        autocast_device = "cuda" if device.type == "cuda" else device.type
        with torch.amp.autocast(autocast_device, enabled=False), self._transformer_cache_context(cache_tag):
            output_list = self.transformer(
                latent_model_input_list,
                timestep_vec,
                prompt_embeds,
                return_dict=False,
        )[0]
        for idx, item in enumerate(output_list[:2]):
            self._debug_tensor(f"raw_model_out[{idx}]", item, step=int(task.buffer.current_step), check=False)
        noise_pred = torch.stack([item.float() for item in output_list], dim=0).squeeze(2)
        return -noise_pred

    def _transformer_forward_cfg_batch(self, task, generator, backend, guidance_scale: float, pipe) -> torch.Tensor:
        device = torch.device(torch.cuda.current_device())
        self._move_task_tensors_to_device(task, device)
        latents = self._normalize_latents(task.buffer.latents, "input")
        task.buffer.latents = latents
        timestep = task.buffer.timesteps[task.buffer.current_step]
        timestep_vec = (1000 - timestep.expand(latents.shape[0])) / 1000
        timestep_vec = timestep_vec.repeat(2)

        latent_model_input = latents.to(self.transformer.dtype).repeat(2, 1, 1, 1).unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))
        pos_embeds = self._embedding_tensor_to_list(
            task.buffer.text_embeddings,
            getattr(task.buffer, "_z_text_embedding_lengths", None),
        )
        neg_embeds = self._embedding_tensor_to_list(
            task.buffer.negative_embeddings,
            getattr(task.buffer, "_z_negative_embedding_lengths", None),
        )
        prompt_embeds = pos_embeds + neg_embeds
        self._debug_tensor("latent_model_input_cfg", latent_model_input, step=int(task.buffer.current_step), check=False)
        self._debug_tensor("timestep_model_input_cfg", timestep_vec, step=int(task.buffer.current_step), check=False)
        for idx, item in enumerate(prompt_embeds[:2]):
            self._debug_tensor(f"prompt_embeds_cfg[{idx}]", item, step=int(task.buffer.current_step), check=False)

        autocast_device = "cuda" if device.type == "cuda" else device.type
        set_cfg_type(backend, "cfg")
        with torch.amp.autocast(autocast_device, enabled=False), self._transformer_cache_context("cfg"):
            output_list = self.transformer(
                latent_model_input_list,
                timestep_vec,
                prompt_embeds,
                return_dict=False,
        )[0]
        for idx, item in enumerate(output_list[:2]):
            self._debug_tensor(f"raw_model_out_cfg[{idx}]", item, step=int(task.buffer.current_step), check=False)
        raw_noise_pred = torch.stack([item.float() for item in output_list], dim=0).squeeze(2)
        batch_size = latents.shape[0]
        pos_noise_pred = -raw_noise_pred[:batch_size]
        neg_noise_pred = -raw_noise_pred[batch_size:]
        return self._apply_cfg(pos_noise_pred, neg_noise_pred, guidance_scale, pipe._cfg_normalization)

    @staticmethod
    def _apply_cfg(
        pos_noise_pred: torch.Tensor,
        neg_noise_pred: torch.Tensor,
        guidance_scale: float,
        cfg_normalization: bool | float,
    ) -> torch.Tensor:
        combined = pos_noise_pred + guidance_scale * (pos_noise_pred - neg_noise_pred)
        if cfg_normalization and float(cfg_normalization) > 0.0:
            flat_pos = pos_noise_pred.float().flatten(1)
            flat_combined = combined.float().flatten(1)
            pos_norm = torch.linalg.vector_norm(flat_pos, dim=1).view(-1, *([1] * (combined.ndim - 1)))
            combined_norm = torch.linalg.vector_norm(flat_combined, dim=1).view(-1, *([1] * (combined.ndim - 1)))
            max_norm = pos_norm * float(cfg_normalization)
            scale = torch.where(combined_norm > max_norm, max_norm / combined_norm.clamp_min(1e-12), torch.ones_like(max_norm))
            combined = combined * scale
        return combined

    def _current_guidance_scale(self, task, pipe) -> float:
        guidance_scale = float(pipe.guidance_scale)
        cfg_truncation = getattr(pipe, "_cfg_truncation", 1.0)
        if cfg_truncation is not None and float(cfg_truncation) <= 1.0:
            timestep = task.buffer.timesteps[task.buffer.current_step]
            t_norm = float(((1000 - timestep) / 1000).item())
            if t_norm > float(cfg_truncation):
                guidance_scale = 0.0
        return guidance_scale

    def _z_guided_noise_pred(self, task, generator, backend, guidance_scale: float) -> torch.Tensor:
        pipe = self._ensure_pipeline(backend.args, torch.device(torch.cuda.current_device()))
        apply_cfg = self.supports_cfg(backend.args) and guidance_scale > 0.0
        if apply_cfg:
            if generator.cfg_size == 2:
                if generator.cfg_dispatcher.group.rank_in_group == 0:
                    set_cfg_type(backend, "pos")
                    local_noise_pred = self._transformer_forward(task, generator, backend, "cond", task.buffer.text_embeddings)
                else:
                    set_cfg_type(backend, "neg")
                    local_noise_pred = self._transformer_forward(task, generator, backend, "uncond", task.buffer.negative_embeddings)
                pos_noise_pred, neg_noise_pred = generator.cfg_dispatcher.all_gather_cfg_noise_preds(local_noise_pred)
            else:
                return self._transformer_forward_cfg_batch(task, generator, backend, guidance_scale, pipe)
            return self._apply_cfg(pos_noise_pred, neg_noise_pred, guidance_scale, pipe._cfg_normalization)

        set_cfg_type(backend, "pos")
        return self._transformer_forward(task, generator, backend, "cond", task.buffer.text_embeddings)

    def _broadcast_checkpoint_latents(self, requests: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
        if not (dist.is_available() and dist.is_initialized()):
            return [tensor.to(device=device) for tensor in requests]

        rank = dist.get_rank()
        count_tensor = torch.tensor([len(requests) if rank == 0 else 0], device=device, dtype=torch.long)
        dist.broadcast(count_tensor, src=0)
        count = int(count_tensor.item())
        if count == 0:
            return []

        if rank == 0:
            shape = tuple(requests[0].shape)
            dtype = requests[0].dtype
        else:
            shape = ()
            dtype = torch.float32
        shape_len = torch.tensor([len(shape)], device=device, dtype=torch.long)
        dist.broadcast(shape_len, src=0)
        if rank == 0:
            shape_tensor = torch.tensor(shape, device=device, dtype=torch.long)
        else:
            shape_tensor = torch.empty((int(shape_len.item()),), device=device, dtype=torch.long)
        dist.broadcast(shape_tensor, src=0)
        shape = tuple(int(value) for value in shape_tensor.tolist())

        dtype_id = torch.tensor([0 if dtype == torch.float32 else 1], device=device, dtype=torch.long) if rank == 0 else torch.empty((1,), device=device, dtype=torch.long)
        dist.broadcast(dtype_id, src=0)
        dtype = torch.float32 if int(dtype_id.item()) == 0 else torch.bfloat16

        broadcasted: list[torch.Tensor] = []
        for idx in range(count):
            if rank == 0:
                tensor = requests[idx].to(device=device, dtype=dtype).contiguous()
            else:
                tensor = torch.empty(shape, device=device, dtype=dtype)
            dist.broadcast(tensor, src=0)
            broadcasted.append(tensor)
        return broadcasted

    def _traceplanner_checkpoint_noise_preds(
        self,
        task,
        generator,
        backend,
        step_cache_strategy,
        requests: list[torch.Tensor],
        guidance_scale: float,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], int]:
        checkpoint_latents = self._broadcast_checkpoint_latents(requests, device)
        if not checkpoint_latents:
            return [], 0

        original_latents = task.buffer.latents
        original_dtype = original_latents.dtype
        predictions: list[torch.Tensor] = []
        try:
            for latent in checkpoint_latents:
                task.buffer.latents = latent.to(device=device, dtype=original_dtype)
                guided = self._z_guided_noise_pred(task, generator, backend, guidance_scale)
                if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                    continue
                predictions.append(guided.detach())
                backend.flexcache.record_compute(
                    baseline_units=0.0,
                    actual_units=1.0,
                    task_id=task.task_id,
                    scope="step_cache",
                    unit="dit_forward",
                    extra={
                        "decision": "checkpoint",
                        "step": int(task.buffer.current_step),
                        "strategy": step_cache_strategy.type,
                    },
                )
        finally:
            task.buffer.latents = original_latents
        return predictions, len(checkpoint_latents)

    def denoise_step(self, task, generator, backend, run_dit_forward: Callable[..., torch.Tensor]) -> torch.Tensor:
        if not getattr(task.buffer, "_z_image_inside_denoise_loop", False):
            device = torch.device(torch.cuda.current_device())
            autocast_device = "cuda" if device.type == "cuda" else device.type
            with torch.amp.autocast(autocast_device, enabled=False):
                task.buffer._z_image_inside_denoise_loop = True
                final_latents = task.buffer.latents
                try:
                    total_steps = int(task.req.params.num_inference_steps)
                    while int(task.buffer.current_step) < total_steps:
                        final_latents = self.denoise_step(task, generator, backend, run_dit_forward)
                        task.buffer.latents = final_latents
                        task.buffer.current_step += 1
                finally:
                    task.buffer._z_image_inside_denoise_loop = False
                return final_latents

        device = torch.device(torch.cuda.current_device())
        self._move_task_tensors_to_device(task, device)
        pipe = self._ensure_pipeline(backend.args, device)
        latents = task.buffer.latents
        timestep = task.buffer.timesteps[task.buffer.current_step]
        guidance_scale = self._current_guidance_scale(task, pipe)
        flex_strategy = getattr(getattr(backend, "flexcache", None), "strategy", None)
        step_cache_strategy = flex_strategy if is_step_level_cache_strategy(flex_strategy) else None
        step_index = int(task.buffer.current_step)
        autocast_device = "cuda" if device.type == "cuda" else device.type

        sigma_pre = None
        sigma_next = None
        if step_cache_strategy is not None:
            sigmas = getattr(pipe.scheduler, "sigmas", None)
            if sigmas is not None and step_index + 1 < len(sigmas):
                sigma_pre = sigmas[step_index]
                sigma_next = sigmas[step_index + 1]
            reuse_key = step_cache_strategy.get_reuse_key(step=step_index)
            if reuse_key is not None:
                latents_pre = latents.detach()
                noise_pred = step_cache_strategy.reuse(step=step_index).to(device=device, dtype=latents.dtype)
                self._debug_tensor("reused_noise_pred", noise_pred, step=step_index)
                backend.flexcache.record_compute(
                    baseline_units=1.0,
                    actual_units=0.0,
                    task_id=task.task_id,
                    scope="step_cache",
                    unit="dit_forward",
                    extra={"decision": "reuse", "step": step_index, "strategy": step_cache_strategy.type},
                )
                with torch.amp.autocast(autocast_device, enabled=False):
                    with self._float32_default_dtype():
                        latents = pipe.scheduler.step(noise_pred.to(torch.float32), timestep, latents, return_dict=False)[0]
                latents = self._normalize_latents(latents, "scheduler output")
                self._debug_tensor("latents_after_step", latents, step=step_index)
                if hasattr(step_cache_strategy, "record_reuse_step"):
                    if sigma_pre is None or sigma_next is None:
                        raise RuntimeError("Step-level cache requires scheduler sigmas for Z-Image.")
                    step_cache_strategy.record_reuse_step(
                        step=step_index,
                        latents_pre=latents_pre,
                        latents=latents,
                        sigma_pre=sigma_pre,
                        sigma=sigma_next,
                        noise_pred=noise_pred,
                    )
                return latents.to(torch.float32)

        latents_pre = latents.detach()
        with torch.amp.autocast(autocast_device, enabled=False):
            noise_pred = self._z_guided_noise_pred(task, generator, backend, guidance_scale)
            self._debug_tensor("noise_pred", noise_pred, step=step_index)
            with self._float32_default_dtype():
                latents = pipe.scheduler.step(noise_pred.to(torch.float32), timestep, latents, return_dict=False)[0]
        latents = self._normalize_latents(latents, "scheduler output").to(torch.float32)
        self._debug_tensor("latents_after_step", latents, step=step_index)

        if step_cache_strategy is not None:
            if sigma_pre is None or sigma_next is None:
                raise RuntimeError("Step-level cache requires scheduler sigmas for Z-Image.")
            store_kwargs = {
                "step": step_index,
                "latents_pre": latents_pre,
                "latents": latents,
                "sigma_pre": sigma_pre,
                "sigma": sigma_next,
                "noise_pred": noise_pred,
                "guided_noise_pred": noise_pred,
            }
            checkpoint_count = 0
            if hasattr(step_cache_strategy, "plan_checkpoint_requests"):
                requests = step_cache_strategy.plan_checkpoint_requests(**store_kwargs)
                checkpoint_noise_preds, checkpoint_count = self._traceplanner_checkpoint_noise_preds(
                    task,
                    generator,
                    backend,
                    step_cache_strategy,
                    requests,
                    guidance_scale,
                    device,
                )
                if checkpoint_count > 0:
                    step_cache_strategy.finish_checkpoint_requests(checkpoint_noise_preds)
            if checkpoint_count == 0:
                step_cache_strategy.store(**store_kwargs)
            backend.flexcache.record_compute(
                baseline_units=1.0,
                actual_units=1.0,
                task_id=task.task_id,
                scope="step_cache",
                unit="dit_forward",
                extra={"decision": "compute", "step": step_index, "strategy": step_cache_strategy.type},
            )
        return latents

    def decode_latents(self, task, generator, backend) -> Optional[torch.Tensor]:
        device = torch.device(torch.cuda.current_device())
        self._move_task_tensors_to_device(task, device)
        pipe = self._ensure_pipeline(backend.args, device)
        latents = task.buffer.latents.to(pipe.vae.dtype)
        self._debug_tensor("latents_before_decode", task.buffer.latents)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

        with device_scope(pipe.vae, backend):
            decoded = pipe.vae.decode(latents, return_dict=False)[0]
        self._debug_tensor("decoded_image_tensor", decoded)
        return decoded

    def save_output(self, task, output: Optional[torch.Tensor], generator, backend) -> None:
        if torch.distributed.get_rank() == 0:
            pipe = self._ensure_pipeline(backend.args, torch.device(torch.cuda.current_device()))
            run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
            if run_output_dir:
                task.req.params.save_dir = task_results_dir(run_output_dir, task.task_id)
            os.makedirs(task.req.params.save_dir, exist_ok=True)

            save_name = task.req.get_prompt()[:20].replace(" ", "_").replace(".", "") + f"_{task.task_id}.png"
            save_path = os.path.join(task.req.params.save_dir, save_name)
            image = pipe.image_processor.postprocess(output, output_type="pil")[0]
            image.save(save_path, quality=95, subsampling=0)

            sidecar_path = os.path.splitext(save_path)[0] + ".json"
            metadata = {
                "filename": os.path.basename(save_path),
                "relative_path": os.path.join(os.path.basename(task.req.params.save_dir), os.path.basename(save_path)),
                "prompt": task.req.get_prompt(),
                "seed": getattr(task.req.params, "seed", None),
                "step": getattr(task.req.params, "num_inference_steps", None),
                "task_id": task.task_id,
                "model_name": getattr(getattr(getattr(backend, "args", None), "models", None), "name", None),
            }
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            log_result(logger, task_id=task.task_id, message=f"image_saved={save_path}")
