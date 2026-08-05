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
from chitu_diffusion.core.distributed.parallel_state import (
    dynamic_sp_degrees,
    get_cp_group,
)
from chitu_diffusion.models.parallel import ModelParallelCapabilities
from chitu_diffusion.parallel.vae import parallel_tiled_vae_decode
from chitu_diffusion.runtime.output_layout import task_results_dir
from chitu_diffusion.runtime.cfg_execution import (
    CfgBranchBatch,
    CfgBranchPredictions,
    CfgExecutionSpec,
)
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
        self._z_attn_processor = None
        self._cp_wrapped = False

    def supports_cfg(self, args: Any) -> bool:
        return float(args.models.sampler.guidance_scale[0]) > 1.0

    def supports_physical_cfg_batch(
        self, task, generator, backend, spec: CfgExecutionSpec
    ) -> bool:
        # The Z transformer accepts cond+uncond as one local list batch. Its CP
        # wrapper intentionally handles one sample branch at a time.
        return spec.cfp_degree == 1 and spec.cp_degree == 1

    def supports_rank0_text_encode(self) -> bool:
        # Z-Image's encoded prompt state is the single padded token tensor plus the
        # ``_z_*_embedding_lengths`` metadata the generator broadcasts, so the
        # rank0_gpu encode-once-and-broadcast fast path reproduces it exactly.
        return True

    def parallel_capabilities(self, args: Any) -> ModelParallelCapabilities:
        return ModelParallelCapabilities(
            dit_context_parallel=self.handles_context_parallel(args),
            dit_cfg_parallel=self.supports_cfg(args),
            vae_tile_parallel=True,
            sampler_local_latent=False,
            model_specific_context_parallel=self.handles_context_parallel(args),
        )

    def handles_context_parallel(self, args: Any) -> bool:
        diffusion = args.infer.diffusion
        return (
            int(getattr(diffusion, "cp_size", 1)) > 1
            or str(getattr(diffusion, "topology_mode", "") or "") == "elastic"
            or bool(getattr(diffusion, "dynamic_sp", False))
        )

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
        self._z_attn_processor = processor
        self._configured_attn_backend = getattr(attn_backend, "impl", attn_backend.__class__.__name__)
        logger.info("Installed Z-Image Chitu attention processor: %s", self._configured_attn_backend)

    def configure_after_backend_build(self, backend) -> None:
        cp_size = max(dynamic_sp_degrees() or [get_cp_group().group_size])
        torch.set_default_dtype(torch.float32)
        logger.info("Z-Image runtime keeps torch default dtype at float32 for scheduler/sequence numerics.")
        # Install once when any registered lane can use CP. The wrapper reads the
        # live lane group and is a no-op for CP-1.
        if cp_size > 1:
            self._wrap_transformer_forward_with_cp(max(cp_size, 1))

    def _wrap_transformer_forward_with_cp(self, cp_size: int) -> None:
        """Wrap the diffusers Z-Image transformer.forward with a context-parallel
        variant. CP is fully encapsulated inside forward: inputs and the returned
        noise prediction stay full-resolution, so the adapter's denoise loop, the
        scheduler and the CFG aggregation remain unchanged.

        Strategy (mirrors qwen_image's split-Q / all-gather-KV, adapted to the
        Z-Image single-stream packing [image, text]):
          * patchify + refiner stages + unified-sequence build run replicated on
            every rank (cheap: 2+2 refiner layers);
          * the image-token portion of the unified sequence is split across CP
            ranks right before the 30 main `layers`, so attention AND FFN are
            parallelized; attention all-gathers image K/V to stay exact;
          * after `layers`, the local image tokens are all-gathered back and the
            full sequence is reassembled for final_layer + unpatchify.
        Only the basic (non-omni), single-sample (B==1), no-controlnet/siglip,
        unpadded-mask path is parallelized; anything else transparently falls
        back to the original forward.
        """
        transformer = self.transformer
        if transformer is None:
            logger.warning("Z-Image CP requested (cp_size=%d) but transformer is not built yet; skip.", cp_size)
            return
        if self._cp_wrapped:
            return
        if self._z_attn_processor is None:
            raise NotImplementedError(
                "Z-Image context parallel requires the Chitu attention processor; "
                "unset CHITU_Z_IMAGE_USE_DIFFUSERS_ATTN to enable CP."
            )

        adapter = self
        original_forward = transformer.forward

        def cp_forward(
            x,
            t,
            cap_feats,
            return_dict: bool = True,
            controlnet_block_samples=None,
            siglip_feats=None,
            image_noise_mask=None,
            patch_size: int = 2,
            f_patch_size: int = 1,
        ):
            group = get_cp_group()
            cp = group.group_size if group is not None else 1
            omni_mode = isinstance(x[0], list)

            if (
                cp <= 1
                or omni_mode
                or siglip_feats is not None
                or controlnet_block_samples is not None
                or len(x) != 1
            ):
                return original_forward(
                    x,
                    t,
                    cap_feats,
                    return_dict=return_dict,
                    controlnet_block_samples=controlnet_block_samples,
                    siglip_feats=siglip_feats,
                    image_noise_mask=image_noise_mask,
                    patch_size=patch_size,
                    f_patch_size=f_patch_size,
                )

            device = x[0].device
            adaln_input = transformer.t_embedder(t * transformer.t_scale).type_as(x[0])

            (
                x_tok,
                cap_tok,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                x_pad_mask,
                cap_pad_mask,
            ) = transformer.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

            # Image embed (replicated; cheap). Image tokens are sharded right
            # after at the noise_refiner and stay sharded all the way through the
            # main layers and final_layer, so the image sequence is never
            # re-gathered mid-forward (see the end-to-end sharding below).
            x_seqlens = [len(xi) for xi in x_tok]
            x_emb = transformer.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x_tok, dim=0))
            x_seq, x_freqs, x_mask, _, x_noise_tensor = transformer._prepare_sequence(
                list(x_emb.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, transformer.x_pad_token, None, device
            )

            # Caption embed + refine: pure text, kept replicated on every rank
            # (only a handful of tokens). Run it BEFORE enabling CP so its
            # text-only attention is not misread as the [image, text] CP layout.
            cap_seqlens = [len(ci) for ci in cap_tok]
            cap_emb = transformer.cap_embedder(torch.cat(cap_tok, dim=0))
            cap_seq, cap_freqs, cap_mask, _, _ = transformer._prepare_sequence(
                list(cap_emb.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, transformer.cap_pad_token, None, device
            )
            for layer in transformer.context_refiner:
                cap_seq = layer(cap_seq, cap_mask, cap_freqs)

            # Z-Image's _prepare_sequence / _build_unified_sequence unconditionally
            # return an attention-mask tensor (all-ones for a single unpadded
            # sequence), never None. An all-valid mask is semantically equivalent to
            # "no mask" (cf. _mask_allows_fast_path), so normalize it to None here.
            # Without this, the CP sharding gate below -- which requires x_mask/cap_mask
            # to be None -- can NEVER fire, and cp>1 silently degrades to full-sequence
            # compute replicated on every rank (i.e. zero speedup from CP).
            if x_mask is not None and bool(x_mask.all()):
                x_mask = None
            if cap_mask is not None and bool(cap_mask.all()):
                cap_mask = None

            # End-to-end sequence-sharded image path: split the image tokens once,
            # keep them sharded through noise_refiner -> unified build -> main
            # layers -> final_layer, and gather only ONCE after final_layer, where
            # each token is just pF*pH*pW*out_channels wide (~60x smaller than the
            # hidden dim) -> that gather is almost free. The 2 noise_refiner layers
            # and the final_layer (previously replicated on the full sequence) now
            # run on the local shard too. Disable with CHITU_Z_IMAGE_CP_SHARDED=0.
            img_len = int(x_seqlens[0])
            do_cp = (
                cp > 1
                and not omni_mode
                and len(x_seqlens) == 1
                and img_len % cp == 0
                and x_mask is None
                and cap_mask is None
                and os.environ.get("CHITU_Z_IMAGE_CP_SHARDED", "1") == "1"
            )
            if os.environ.get("CHITU_Z_IMAGE_CP_PROFILE", "0") == "1" and not getattr(adapter, "_cp_diag_logged", False):
                logger.info(
                    "[Z-Image CP diag] cp=%d img_len=%d x_mask_is_none=%s do_cp=%s",
                    cp, img_len, x_mask is None, do_cp,
                )
                adapter._cp_diag_logged = True
            if not do_cp and cp > 1 and img_len % cp != 0:
                logger.warning(
                    "Z-Image CP skipped this step: image tokens=%d not divisible by cp_size=%d.", img_len, cp
                )

            if do_cp:
                chunk = img_len // cp
                r = group.rank_in_group
                lo, hi = r * chunk, (r + 1) * chunk
                x_seq = x_seq[:, lo:hi].contiguous()
                x_freqs = x_freqs[:, lo:hi].contiguous()
                x_seqlens = [chunk]
                adapter._z_attn_processor.enable_cp(group, chunk)
            for layer in transformer.noise_refiner:
                x_seq = layer(x_seq, x_mask, x_freqs, adaln_input, x_noise_tensor, None, None)

            # With a sharded image + full text, build_unified yields exactly
            # [local_image_shard, text] -> the main layers consume it directly.
            unified, unified_freqs, unified_mask, unified_noise_tensor = transformer._build_unified_sequence(
                x_seq,
                x_freqs,
                x_seqlens,
                None,
                cap_seq,
                cap_freqs,
                cap_seqlens,
                None,
                None,
                None,
                None,
                None,
                omni_mode,
                device,
            )

            try:
                for layer in transformer.layers:
                    unified = layer(
                        unified, unified_mask, unified_freqs, adaln_input, unified_noise_tensor, None, None
                    )
            finally:
                if do_cp:
                    adapter._z_attn_processor.disable_cp()

            # final_layer is per-token, so it runs on the local shard. Project to
            # the small patch-output dim first, then gather the (now tiny) image
            # tokens once and unpatchify the full image.
            unified = transformer.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
            if do_cp:
                chunk = img_len // cp
                local_img_out = unified[:, :chunk].contiguous()
                full_img_out = adapter._z_attn_processor._cp_all_gather_seq(local_img_out, group, kind="out")
            else:
                full_img_out = unified[:, :img_len]
            out = transformer.unpatchify(list(full_img_out.unbind(dim=0)), x_size, patch_size, f_patch_size, None)

            if not return_dict:
                return (out,)
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=out)

        transformer.forward = cp_forward
        self._cp_wrapped = True
        logger.info("Installed Z-Image context-parallel transformer forward wrapper (cp_size=%d).", cp_size)

    def encode_text(self, task, generator, backend):
        device = torch.device(torch.cuda.current_device())
        pipe = self._ensure_pipeline(backend.args, device)
        guidance_scale = float(backend.args.models.sampler.guidance_scale[0])
        max_sequence_length = int(getattr(backend.args.models.encoder, "max_sequence_length", 512))
        negative_prompt = task.req.get_n_prompt() or ""

        # CFG branch state is prepared FULLY and identically on every rank at
        # admission: a first pass encodes the positive prompt, a second the
        # negative. The lane's cfp1/cfp2 split happens later at denoise time in the
        # CFG executor (which forwards only the local branch on a cfp2 lane), so the
        # replicated cond+neg buffers keep a task reassignable to any live topology.
        encode_negative = task.buffer.text_embeddings is not None and guidance_scale > 1.0

        prompt = negative_prompt if encode_negative else task.req.get_prompt()
        set_cfg_type(backend, "neg" if encode_negative else "pos")
        cache = getattr(self, "_text_embed_cache", None)
        if cache is None:
            cache = {}
            self._text_embed_cache = cache
        cache_key = (prompt, int(max_sequence_length))
        cached = cache.get(cache_key)
        if cached is not None:
            padded_embeds = cached[0].to(device=device)
            lengths = list(cached[1])
        else:
            embeds = pipe._encode_prompt(
                prompt=[prompt],
                device=device,
                max_sequence_length=max_sequence_length,
            )
            lengths = [int(item.shape[0]) for item in embeds]
            padded_embeds = torch.nn.utils.rnn.pad_sequence(embeds, batch_first=True, padding_value=0.0)
            if len(cache) < 512:
                cache[cache_key] = (padded_embeds.detach().to("cpu"), list(lengths))
            logger.info(
                "[text_encode_step] Z-Image branch=%s embeds=%s",
                "neg" if encode_negative else "pos",
                [tuple(item.shape) for item in embeds],
            )
        if encode_negative:
            task.buffer._z_negative_embedding_lengths = lengths
        else:
            task.buffer._z_text_embedding_lengths = lengths
        return padded_embeds

    def prepare_denoise(self, task, generator, backend) -> None:
        from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, retrieve_timesteps

        device = torch.device(torch.cuda.current_device())
        pipe = self._ensure_pipeline(backend.args, device)
        width, height = task.req.params.size
        vae_scale = pipe.vae_scale_factor * 2
        if height % vae_scale != 0 or width % vae_scale != 0:
            raise ValueError(f"Z-Image height/width must be divisible by {vae_scale}, got {(width, height)}.")

        embeddings = task.buffer.text_embeddings
        if embeddings is None:
            embeddings = task.buffer.negative_embeddings
        if embeddings is None:
            raise RuntimeError("Z-Image denoise requires prompt embeddings.")
        num_inference_steps = int(task.req.params.num_inference_steps)

        seed = task.req.params.base_seed(fallback=int(backend.args.infer.seed))
        seed_g = torch.Generator(device=device).manual_seed(seed)
        latents = pipe.prepare_latents(
            1,
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

        # Announce the denoise step/cfg branch to the CP core so the AGKV
        # cross-step KV cache (if enabled) can pick fresh vs stale per layer.
        cp_core = getattr(getattr(self, "_z_attn_processor", None), "_cp", None)
        if cp_core is not None and hasattr(cp_core, "begin_forward"):
            cp_core.begin_forward(
                int(task.buffer.current_step), len(task.buffer.timesteps), cache_tag
            )

        autocast_device = "cuda" if device.type == "cuda" else device.type
        with torch.amp.autocast(autocast_device, enabled=False), self._transformer_cache_context(cache_tag):
            output_list = self.transformer(
                latent_model_input_list,
                timestep_vec,
                prompt_embeds,
                return_dict=False,
        )[0]
        noise_pred = torch.stack([item.float() for item in output_list], dim=0).squeeze(2)
        return -noise_pred

    def _transformer_forward_cfg_batch(
        self, task, generator, backend, branches: CfgBranchBatch
    ) -> CfgBranchPredictions[torch.Tensor]:
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

        autocast_device = "cuda" if device.type == "cuda" else device.type
        set_cfg_type(backend, "cfg")
        with torch.amp.autocast(autocast_device, enabled=False), self._transformer_cache_context("cfg"):
            output_list = self.transformer(
                latent_model_input_list,
                timestep_vec,
                prompt_embeds,
                return_dict=False,
        )[0]
        raw_noise_pred = torch.stack([item.float() for item in output_list], dim=0).squeeze(2)
        batch_size = latents.shape[0]
        return CfgBranchPredictions(
            cond=-raw_noise_pred[:batch_size],
            uncond=-raw_noise_pred[batch_size:],
        )

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
        spec = self.cfg_execution_spec(
            task, generator, backend, guidance_enabled=apply_cfg
        )
        branches = self.cfg_branch_batch(task)

        def forward_branch(name: str, state) -> torch.Tensor:
            set_cfg_type(backend, "pos" if name == "cond" else "neg")
            return self._transformer_forward(
                task,
                generator,
                backend,
                name,
                state,
            )

        return self.cfg_executor.execute(
            spec=spec,
            branches=branches,
            forward_branch=forward_branch,
            forward_physical_batch=lambda batch: self._transformer_forward_cfg_batch(
                task, generator, backend, batch
            ),
            combine=lambda cond, uncond: self._apply_cfg(
                cond,
                uncond,
                guidance_scale,
                pipe._cfg_normalization,
            ),
        )

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
        # One denoise step. The engine (single-request driver or the pool-engine
        # group contract) owns the ``while current_step < total_steps`` loop and
        # increments ``current_step`` after this returns, exactly like the other
        # DiT adapters -- Z-Image no longer self-loops here.
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
                        latents = self.scheduler_step(
                            pipe.scheduler, noise_pred.to(torch.float32), timestep, latents, step_index=step_index
                        )
                latents = self._normalize_latents(latents, "scheduler output")
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
            with self._float32_default_dtype():
                latents = self.scheduler_step(
                    pipe.scheduler, noise_pred.to(torch.float32), timestep, latents, step_index=step_index
                )
        latents = self._normalize_latents(latents, "scheduler output").to(torch.float32)

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
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

        with device_scope(pipe.vae, backend):
            def _decode(z: torch.Tensor) -> torch.Tensor:
                return pipe.vae.decode(z, return_dict=False)[0]

            # The active CP group is the task's current lane, not the world.
            decoded = parallel_tiled_vae_decode(
                latents,
                _decode,
                latent_split_dim=2,
                pixel_split_dim=2,
                scale=int(pipe.vae_scale_factor),
                group=get_cp_group(),
            )
        return decoded

    def save_output(self, task, output: Optional[torch.Tensor], generator, backend) -> None:
        if torch.distributed.get_rank() != 0 or output is None:
            return
        pipe = self._ensure_pipeline(backend.args, torch.device(torch.cuda.current_device()))
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            task.req.params.save_dir = task_results_dir(run_output_dir, task.req.request_id)
        os.makedirs(task.req.params.save_dir, exist_ok=True)

        images = pipe.image_processor.postprocess(output[:1], output_type="pil")
        prompt_slug = task.req.get_prompt()[:20].replace(" ", "_").replace(".", "")
        model_name = getattr(getattr(getattr(backend, "args", None), "models", None), "name", None)
        image = images[0]
        save_name = f"{prompt_slug}_{task.task_id}.png"
        save_path = os.path.join(task.req.params.save_dir, save_name)
        image.save(save_path, quality=95, subsampling=0)

        sidecar_path = os.path.splitext(save_path)[0] + ".json"
        metadata = {
            "filename": os.path.basename(save_path),
            "relative_path": os.path.join(os.path.basename(task.req.params.save_dir), os.path.basename(save_path)),
            "prompt": task.req.get_prompt(),
            "seed": task.req.params.base_seed(fallback=int(backend.args.infer.seed)),
            "sample_index": task.sample_index,
            "n_sample": task.sample_count,
            "step": getattr(task.req.params, "num_inference_steps", None),
            "task_id": task.task_id,
            "request_id": task.req.request_id,
            "model_name": model_name,
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        log_result(logger, task_id=task.task_id, message=f"image_saved={save_path}")
