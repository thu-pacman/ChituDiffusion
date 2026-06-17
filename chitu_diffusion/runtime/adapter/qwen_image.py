from __future__ import annotations

from logging import getLogger
from math import prod
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist

from chitu_diffusion.core.distributed.parallel_state import get_cp_group
from chitu_diffusion.runtime.adapter.base import DiffusionRuntimeAdapter, register_model_runtime, set_cfg_type
from chitu_diffusion.runtime.parallel_utils import SequencePadder

logger = getLogger(__name__)


@register_model_runtime("qwen-image", "Qwen-Image")
class QwenImageRuntimeAdapter(DiffusionRuntimeAdapter):
    def __init__(self, spec):
        super().__init__(spec)
        self.pipeline = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self._configured_attn_backend = None
        self._cp_wrapped = False

    def schedule_each_stage(self) -> bool:
        return True

    def supports_cfg(self, args: Any) -> bool:
        return float(args.models.sampler.guidance_scale[0]) > 1.0

    def _torch_dtype(self, args: Any) -> torch.dtype:
        variant = str(getattr(args, "float_16bit_variant", "bfloat16")).lower()
        if variant in {"float16", "fp16", "half"}:
            return torch.float16
        return torch.bfloat16

    def _ensure_pipeline(self, args: Any, device: torch.device):
        if self.pipeline is not None:
            self.pipeline.to(device)
            return self.pipeline

        if self.text_encoder is None or self.vae is None or self.transformer is None:
            raise RuntimeError("Qwen-Image components are not initialized.")

        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
        from transformers import Qwen2Tokenizer

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
        pipe = QwenImagePipeline(
            scheduler=scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=tokenizer,
            transformer=self.transformer,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe._attention_kwargs = {}
        self.pipeline = pipe
        logger.info("Initialized Qwen-Image stage helper pipeline.")
        return pipe

    def load_text_encoder(self, args: Any, init_device: torch.device):
        from transformers import Qwen2_5_VLForConditionalGeneration

        logger.info("Initializing Qwen-Image text encoder from %s", args.models.ckpt_dir)
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.models.ckpt_dir,
            subfolder="text_encoder",
            torch_dtype=self._torch_dtype(args),
            local_files_only=True,
        ).to(init_device)
        self.text_encoder.eval().requires_grad_(False)
        return self.text_encoder

    def load_vae(self, args: Any, init_device: torch.device):
        from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

        logger.info("Initializing Qwen-Image VAE from %s", args.models.ckpt_dir)
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            args.models.ckpt_dir,
            subfolder="vae",
            torch_dtype=self._torch_dtype(args),
            local_files_only=True,
        ).to(init_device)
        self.vae.eval().requires_grad_(False)
        return self.vae

    def checkpoint_paths(self, args: Any) -> list[str]:
        return []

    def loads_transformer_weights(self) -> bool:
        return True

    def build_transformer(self, models_config: Any, attn_backend, rope_impl) -> torch.nn.Module:
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

        variant = str(getattr(models_config, "float_16bit_variant", "") or torch.get_default_dtype()).lower()
        torch_dtype = torch.float16 if "float16" in variant or "fp16" in variant else torch.bfloat16
        transformer = QwenImageTransformer2DModel.from_pretrained(
            models_config.ckpt_dir,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        if attn_backend is not None:
            self._install_attention_processor(transformer, attn_backend)
        self._install_backbone_api(transformer)
        transformer.eval().requires_grad_(False)
        self.transformer = transformer
        return transformer

    def _install_backbone_api(self, transformer) -> None:
        """Expose Qwen-Image's block loop through the shared FlexCache backbone API."""
        import types

        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask

        from chitu_diffusion.core.models.backbone import BackboneBlockInfo, BackboneState, detach_backbone_value

        def backbone_blocks(model_self):
            return [
                BackboneBlockInfo(index=index, name=f"transformer_blocks.{index}", module=block)
                for index, block in enumerate(model_self.transformer_blocks)
            ]

        def backbone_attention_modules(model_self):
            return [
                (block_info.index, "self", block_info.module.attn)
                for block_info in model_self.backbone_blocks()
                if hasattr(block_info.module, "attn")
            ]

        def backbone_make_state(model_self, hidden_states: torch.Tensor, **kwargs) -> BackboneState:
            return BackboneState({"hidden_states": hidden_states, **kwargs})

        def block_compute(model_self, block_info: BackboneBlockInfo, state: BackboneState) -> BackboneState:
            encoder_hidden_states, hidden_states = block_info.module(
                hidden_states=state["hidden_states"],
                encoder_hidden_states=state["encoder_hidden_states"],
                encoder_hidden_states_mask=None,
                temb=state["temb"],
                image_rotary_emb=state["image_rotary_emb"],
                joint_attention_kwargs=state["block_attention_kwargs"],
                modulate_index=state["modulate_index"],
            )
            controlnet_block_samples = state.get("controlnet_block_samples")
            if controlnet_block_samples is not None:
                interval_control = len(model_self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[block_info.index // interval_control]
            state["encoder_hidden_states"] = encoder_hidden_states
            state["hidden_states"] = hidden_states
            return state

        def backbone_cache_state(model_self, state: BackboneState):
            return detach_backbone_value(
                {
                    "hidden_states": state["hidden_states"],
                    "encoder_hidden_states": state["encoder_hidden_states"],
                }
            )

        def backbone_restore_cached_state(model_self, state: BackboneState, cached_state):
            state["hidden_states"] = cached_state["hidden_states"]
            state["encoder_hidden_states"] = cached_state["encoder_hidden_states"]
            return state

        def backbone_finalize_state(model_self, state: BackboneState) -> torch.Tensor:
            temb = state["temb"]
            if model_self.zero_cond_t:
                temb = temb.chunk(2, dim=0)[0]
            hidden_states = model_self.norm_out(state["hidden_states"], temb)
            return model_self.proj_out(hidden_states)

        def model_compute(model_self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
            state = model_self.backbone_make_state(hidden_states, **kwargs)
            for block_info in model_self.backbone_blocks():
                state = model_self.block_compute(block_info, state)
            return model_self.backbone_finalize_state(state)

        def forward_with_backbone(
            model_self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            timestep: torch.Tensor = None,
            img_shapes: list[tuple[int, int, int]] | None = None,
            txt_seq_lens: list[int] | None = None,
            guidance: torch.Tensor = None,
            attention_kwargs: dict[str, Any] | None = None,
            controlnet_block_samples=None,
            additional_t_cond=None,
            return_dict: bool = True,
        ):
            hidden_states = model_self.img_in(hidden_states)
            timestep = timestep.to(hidden_states.dtype)

            if model_self.zero_cond_t:
                timestep = torch.cat([timestep, timestep * 0], dim=0)
                modulate_index = torch.tensor(
                    [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                    device=timestep.device,
                    dtype=torch.int,
                )
            else:
                modulate_index = None

            encoder_hidden_states = model_self.txt_norm(encoder_hidden_states)
            encoder_hidden_states = model_self.txt_in(encoder_hidden_states)
            text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
                encoder_hidden_states,
                encoder_hidden_states_mask,
            )

            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000
            temb = (
                model_self.time_text_embed(timestep, hidden_states, additional_t_cond)
                if guidance is None
                else model_self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
            )

            image_rotary_emb = model_self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

            block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
            if encoder_hidden_states_mask is not None:
                batch_size, image_seq_len = hidden_states.shape[:2]
                image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
                block_attention_kwargs["attention_mask"] = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)

            output = model_self.model_compute(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                block_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
                controlnet_block_samples=controlnet_block_samples,
            )
            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        transformer.backbone_blocks = types.MethodType(backbone_blocks, transformer)
        transformer.backbone_attention_modules = types.MethodType(backbone_attention_modules, transformer)
        transformer.backbone_make_state = types.MethodType(backbone_make_state, transformer)
        transformer.block_compute = types.MethodType(block_compute, transformer)
        transformer.backbone_cache_state = types.MethodType(backbone_cache_state, transformer)
        transformer.backbone_restore_cached_state = types.MethodType(backbone_restore_cached_state, transformer)
        transformer.backbone_finalize_state = types.MethodType(backbone_finalize_state, transformer)
        transformer.model_compute = types.MethodType(model_compute, transformer)
        transformer.forward = types.MethodType(forward_with_backbone, transformer)
        logger.info("Installed Qwen-Image FlexCache backbone API.")

    def _install_attention_processor(self, transformer, attn_backend) -> None:
        if attn_backend is None:
            return
        from chitu_diffusion.modules.attention.qwen_image_attention import QwenImageChituAttnProcessor

        processor = QwenImageChituAttnProcessor(attn_backend)
        for block in transformer.transformer_blocks:
            block.attn.set_processor(processor)
        self._configured_attn_backend = getattr(attn_backend, "impl", attn_backend.__class__.__name__)
        logger.info("Installed Qwen-Image Chitu attention processor: %s", self._configured_attn_backend)

    def configure_after_backend_build(self, backend) -> None:
        cp_size = int(getattr(backend.args.infer.diffusion, "cp_size", 1))
        up = int(getattr(backend.args.infer.diffusion, "up", 1))
        if cp_size <= 1:
            return
        if up < 1 or cp_size % up != 0:
            raise NotImplementedError(
                "Qwen-Image context parallel requires infer.diffusion.up to divide cp_size; "
                f"got up={up}, cp_size={cp_size}."
            )
        self._wrap_transformer_forward_with_cp(backend)

    def handles_context_parallel(self, args: Any) -> bool:
        return int(getattr(args.infer.diffusion, "cp_size", 1)) > 1

    def _split_sequence_with_offset(self, tensor: torch.Tensor, split_dim: int, name: str) -> tuple[torch.Tensor, int]:
        group = get_cp_group()
        pieces = SequencePadder.split_sequence_padding(tensor, group.group_size, split_dim=split_dim, name=name)
        local = pieces[group.rank_in_group]
        offset = sum(piece.shape[split_dim] for piece in pieces[: group.rank_in_group])
        return local, offset

    def _gather_sequence(self, tensor: torch.Tensor, gather_dim: int, name: str) -> torch.Tensor:
        group = get_cp_group()
        pieces = [torch.empty_like(tensor) for _ in range(group.group_size)]
        dist.all_gather(pieces, tensor.contiguous(), group=group.gpu_group)
        return SequencePadder.remove_sequence_padding_and_concat(pieces, gather_dim=gather_dim, name=name)

    def _wrap_transformer_forward_with_cp(self, backend) -> None:
        if self.transformer is None or self._cp_wrapped:
            return

        transformer = self.transformer
        original_forward = transformer.forward

        def cp_forward(*args, **kwargs):
            hidden_states = kwargs.get("hidden_states")
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            if hidden_states is None or encoder_hidden_states is None:
                return original_forward(*args, **kwargs)

            local_hidden, image_offset = self._split_sequence_with_offset(
                hidden_states,
                split_dim=1,
                name="qwen_image_hidden_states",
            )

            attention_kwargs = dict(kwargs.get("attention_kwargs") or {})
            attention_kwargs["qwen_cp_info"] = {
                "image_offset": image_offset,
                "image_seq_len": hidden_states.shape[1],
            }
            kwargs["attention_kwargs"] = attention_kwargs
            kwargs["hidden_states"] = local_hidden

            output = original_forward(*args, **kwargs)
            if isinstance(output, tuple):
                local_sample = output[0]
                gathered = self._gather_sequence(local_sample, gather_dim=1, name="qwen_image_hidden_states")
                return (gathered, *output[1:])
            sample = output.sample
            output.sample = self._gather_sequence(sample, gather_dim=1, name="qwen_image_hidden_states")
            return output

        transformer.forward = cp_forward
        self._cp_wrapped = True
        logger.info("Installed Qwen-Image context-parallel transformer wrapper.")

    def encode_text(self, task, generator, backend) -> torch.Tensor:
        device = torch.device(torch.cuda.current_device())
        pipe = self._ensure_pipeline(backend.args, device)
        width, height = task.req.params.size
        max_sequence_length = int(getattr(backend.args.models.encoder, "max_sequence_length", 512))
        guidance_scale = float(backend.args.models.sampler.guidance_scale[0])
        negative_prompt = task.req.get_n_prompt() or " "
        if generator.cfg_size == 1:
            encode_negative = task.buffer.text_embeddings is not None and guidance_scale > 1.0
        elif generator.cfg_size == 2:
            encode_negative = generator.cfg_dispatcher.group.rank_in_group == 1
        else:
            raise ValueError(f"Unsupported cfg_size={generator.cfg_size}.")

        prompt = negative_prompt if encode_negative else task.req.get_prompt()
        set_cfg_type(backend, "neg" if encode_negative else "pos")

        pipe.check_inputs(
            task.req.get_prompt(),
            height,
            width,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
        )

        embeds, embeds_mask = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        if encode_negative:
            task.buffer.negative_embeddings_mask = embeds_mask
        else:
            task.buffer.text_embeddings_mask = embeds_mask

        if generator.cfg_size == 2:
            other_prompt = task.req.get_prompt() if encode_negative else negative_prompt
            other_embeds, other_mask = pipe.encode_prompt(
                prompt=other_prompt,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )
            if encode_negative:
                task.buffer.text_embeddings = other_embeds
                task.buffer.text_embeddings_mask = other_mask
            else:
                task.buffer.negative_embeddings = other_embeds
                task.buffer.negative_embeddings_mask = other_mask

        logger.info(
            "[text_encode_step] Qwen-Image branch=%s embeds=%s mask=%s",
            "neg" if encode_negative else "pos",
            tuple(embeds.shape),
            None if embeds_mask is None else tuple(embeds_mask.shape),
        )
        return embeds

    def prepare_denoise(self, task, generator, backend) -> None:
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps

        device = torch.device(torch.cuda.current_device())
        pipe = self._ensure_pipeline(backend.args, device)
        width, height = task.req.params.size
        seed = int(task.req.params.seed if task.req.params.seed is not None else backend.args.infer.seed)
        seed_g = torch.Generator(device=device).manual_seed(seed)
        embeddings = task.buffer.text_embeddings if task.buffer.text_embeddings is not None else task.buffer.negative_embeddings
        if embeddings is None:
            raise RuntimeError("Qwen-Image denoise requires text or negative embeddings.")
        batch_size = embeddings.shape[0]
        num_inference_steps = int(task.req.params.num_inference_steps)
        num_channels_latents = pipe.transformer.config.in_channels // 4

        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            embeddings.dtype,
            device,
            seed_g,
            latents=None,
        )
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            latents.shape[1],
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        guidance = None
        guidance_scale = float(backend.args.models.sampler.guidance_scale[0])
        if pipe.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        elif guidance_scale is not None:
            logger.debug("Qwen-Image ignores guidance_scale input because this transformer is not guidance-distilled.")

        pipe.scheduler.set_begin_index(0)
        pipe._num_timesteps = len(timesteps)
        task.buffer.seed_g = seed_g
        task.buffer.sampler = pipe.scheduler
        task.buffer.latents = latents
        task.buffer.timesteps = timesteps
        task.buffer.guidance_vec = guidance
        task.buffer.image_size = (width, height)

        backend.switch_active_model(flush=True)
        logger.info(
            "[Pre Denoise Qwen-Image] latents=%s steps=%s mu=%.4f cfg=%.2f",
            tuple(latents.shape),
            len(timesteps),
            mu,
            guidance_scale,
        )
        generator._configure_flexcache_for_task(task)

    def _img_shapes(self, task, pipe) -> list[list[tuple[int, int, int]]]:
        width, height = task.buffer.image_size or task.req.params.size
        batch_size = task.buffer.latents.shape[0]
        return [[(1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2)]] * batch_size

    def _move_task_tensors_to_device(self, task, device: torch.device) -> None:
        for name in (
            "text_embeddings",
            "negative_embeddings",
            "text_embeddings_mask",
            "negative_embeddings_mask",
            "latents",
            "timesteps",
            "guidance_vec",
        ):
            value = getattr(task.buffer, name, None)
            if isinstance(value, torch.Tensor):
                setattr(task.buffer, name, value.to(device))

    @staticmethod
    def _normalize_latent_tokens(tensor: torch.Tensor, name: str) -> torch.Tensor:
        while tensor.ndim > 3 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim != 3:
            raise ValueError(f"Qwen-Image {name} must be [batch, sequence, channels], got {tuple(tensor.shape)}.")
        return tensor

    def _transformer_forward(
        self,
        task,
        generator,
        backend,
        cache_tag: str,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = torch.device(torch.cuda.current_device())
        self._move_task_tensors_to_device(task, device)
        pipe = self._ensure_pipeline(backend.args, device)
        latents = task.buffer.latents
        timestep = task.buffer.timesteps[task.buffer.current_step]
        timestep_vec = timestep.expand(latents.shape[0]).to(latents.dtype)
        with pipe.transformer.cache_context(cache_tag):
            output = generator._run_dit_forward(
                task,
                cache_tag,
                hidden_states=latents,
                timestep=timestep_vec / 1000,
                guidance=task.buffer.guidance_vec,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                encoder_hidden_states=encoder_hidden_states,
                img_shapes=self._img_shapes(task, pipe),
                attention_kwargs=pipe.attention_kwargs,
                return_dict=False,
            )[0]
        return self._normalize_latent_tokens(output, f"{cache_tag} transformer output")

    def denoise_step(self, task, generator, backend, run_dit_forward: Callable[..., torch.Tensor]) -> torch.Tensor:
        device = torch.device(torch.cuda.current_device())
        self._move_task_tensors_to_device(task, device)
        pipe = self._ensure_pipeline(backend.args, device)
        latents = task.buffer.latents
        timestep = task.buffer.timesteps[task.buffer.current_step]
        guidance_scale = float(backend.args.models.sampler.guidance_scale[0])
        flex_strategy = getattr(getattr(backend, "flexcache", None), "strategy", None)
        meancache_strategy = flex_strategy if getattr(flex_strategy, "type", None) == "meancache" else None
        step_index = int(task.buffer.current_step)
        sigma_pre = None
        sigma_next = None
        if meancache_strategy is not None:
            sigmas = getattr(pipe.scheduler, "sigmas", None)
            if sigmas is not None and step_index + 1 < len(sigmas):
                sigma_pre = sigmas[step_index]
                sigma_next = sigmas[step_index + 1]
            reuse_key = meancache_strategy.get_reuse_key(step=step_index)
            if reuse_key is not None:
                local_noise_pred = meancache_strategy.reuse(step=step_index).to(device=device, dtype=latents.dtype)
                backend.flexcache.record_compute(
                    baseline_units=1.0,
                    actual_units=0.0,
                    task_id=task.task_id,
                    scope="meancache_step",
                    unit="dit_forward",
                    extra={"decision": "reuse", "step": step_index},
                )
                if guidance_scale > 1.0 and generator.cfg_size == 2:
                    noise_pred, neg_noise_pred = generator.cfg_dispatcher.all_gather_cfg_noise_preds(local_noise_pred)
                    combined = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(combined, dim=-1, keepdim=True)
                    noise_pred = combined * (cond_norm / noise_norm)
                else:
                    noise_pred = local_noise_pred
                latents_dtype = latents.dtype
                latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                latents = self._normalize_latent_tokens(latents, "scheduler output")
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
                return latents

        latents_pre = latents.detach()
        if guidance_scale > 1.0:
            if generator.cfg_size == 2:
                if generator.cfg_dispatcher.group.rank_in_group == 0:
                    set_cfg_type(backend, "pos")
                    logger.debug(
                        "[denoise_step] Qwen-Image CFG rank=%s branch=cond step=%s",
                        torch.distributed.get_rank(),
                        task.buffer.current_step,
                    )
                    local_noise_pred = self._transformer_forward(
                        task,
                        generator,
                        backend,
                        "cond",
                        task.buffer.text_embeddings,
                        task.buffer.text_embeddings_mask,
                    )
                else:
                    set_cfg_type(backend, "neg")
                    logger.debug(
                        "[denoise_step] Qwen-Image CFG rank=%s branch=uncond step=%s",
                        torch.distributed.get_rank(),
                        task.buffer.current_step,
                    )
                    local_noise_pred = self._transformer_forward(
                        task,
                        generator,
                        backend,
                        "uncond",
                        task.buffer.negative_embeddings,
                        task.buffer.negative_embeddings_mask,
                    )
                logger.debug(
                    "[denoise_step] Qwen-Image CFG rank=%s entering all_gather step=%s pred=%s",
                    torch.distributed.get_rank(),
                    task.buffer.current_step,
                    tuple(local_noise_pred.shape),
                )
                noise_pred, neg_noise_pred = generator.cfg_dispatcher.all_gather_cfg_noise_preds(local_noise_pred)
                logger.debug(
                    "[denoise_step] Qwen-Image CFG rank=%s finished all_gather step=%s",
                    torch.distributed.get_rank(),
                    task.buffer.current_step,
                )
                fresh_noise_pred_for_cache = local_noise_pred
            else:
                set_cfg_type(backend, "pos")
                noise_pred = self._transformer_forward(
                    task,
                    generator,
                    backend,
                    "cond",
                    task.buffer.text_embeddings,
                    task.buffer.text_embeddings_mask,
                )
                set_cfg_type(backend, "neg")
                neg_noise_pred = self._transformer_forward(
                    task,
                    generator,
                    backend,
                    "uncond",
                    task.buffer.negative_embeddings,
                    task.buffer.negative_embeddings_mask,
                )
                fresh_noise_pred_for_cache = noise_pred
            combined = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
            logger.debug(
                "[denoise_step] Qwen-Image CFG rank=%s combined guidance step=%s",
                torch.distributed.get_rank(),
                task.buffer.current_step,
            )
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(combined, dim=-1, keepdim=True)
            noise_pred = combined * (cond_norm / noise_norm)
            logger.debug(
                "[denoise_step] Qwen-Image CFG rank=%s rescaled guidance step=%s",
                torch.distributed.get_rank(),
                task.buffer.current_step,
            )
        else:
            set_cfg_type(backend, "pos")
            noise_pred = self._transformer_forward(
                task,
                generator,
                backend,
                "cond",
                task.buffer.text_embeddings,
                task.buffer.text_embeddings_mask,
            )
            fresh_noise_pred_for_cache = noise_pred

        if meancache_strategy is not None and not (guidance_scale > 1.0 and generator.cfg_size == 2):
            fresh_noise_pred_for_cache = noise_pred

        latents_dtype = latents.dtype
        logger.debug(
            "[denoise_step] Qwen-Image rank=%s entering scheduler step=%s",
            torch.distributed.get_rank(),
            task.buffer.current_step,
        )
        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        latents = self._normalize_latent_tokens(latents, "scheduler output")
        logger.debug(
            "[denoise_step] Qwen-Image rank=%s finished scheduler step=%s",
            torch.distributed.get_rank(),
            task.buffer.current_step,
        )
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)
        if meancache_strategy is not None:
            if sigma_pre is None or sigma_next is None:
                raise RuntimeError("MeanCache requires scheduler sigmas for Qwen-Image.")
            meancache_strategy.store(
                step=step_index,
                latents_pre=latents_pre,
                latents=latents,
                sigma_pre=sigma_pre,
                sigma=sigma_next,
                noise_pred=fresh_noise_pred_for_cache,
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
        if torch.distributed.get_rank() != 0:
            return None
        device = torch.device(torch.cuda.current_device())
        self._move_task_tensors_to_device(task, device)
        pipe = self._ensure_pipeline(backend.args, device)
        width, height = task.buffer.image_size or task.req.params.size
        latents = pipe._unpack_latents(task.buffer.latents, height, width, pipe.vae_scale_factor)
        latents = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        return pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]

    def save_output(self, task, output: Optional[torch.Tensor], generator, backend) -> None:
        if torch.distributed.get_rank() == 0:
            generator._save_image(task, output)
