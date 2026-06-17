from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

from chitu_diffusion.modules.attention.qwen_image_attention import QwenImageChituAttnProcessor


logger = logging.getLogger(__name__)


class QwenImageCubicAttnProcessor:
    """Qwen-Image attention processor that refreshes active image-token K/V only."""

    def __init__(self, engine: "QwenImageCubicSelectiveForwardEngine", layer_idx: int, fallback_backend):
        self.engine = engine
        self.layer_idx = layer_idx
        self.fallback_processor = QwenImageChituAttnProcessor(fallback_backend)
        self.attn_backend = fallback_backend

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        qwen_cp_info: Optional[dict[str, int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        runtime = self.engine._runtime
        if runtime is None:
            return self.fallback_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                qwen_cp_info=qwen_cp_info,
            )
        if qwen_cp_info is not None:
            raise NotImplementedError("Qwen-Image Cubic selective forward requires cp_size = 1.")
        if encoder_hidden_states is None:
            raise ValueError("Qwen-Image Cubic attention requires encoder_hidden_states.")
        if attention_mask is not None:
            raise NotImplementedError("Qwen-Image Cubic attention does not support attention_mask yet.")

        batch_size = hidden_states.shape[0]
        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        img_query = QwenImageChituAttnProcessor._ensure_sequence_major(
            img_query, hidden_states.shape[1], attn.heads, "cubic_img_query"
        )
        img_key = QwenImageChituAttnProcessor._ensure_sequence_major(
            img_key, hidden_states.shape[1], attn.heads, "cubic_img_key"
        )
        img_value = QwenImageChituAttnProcessor._ensure_sequence_major(
            img_value, hidden_states.shape[1], attn.heads, "cubic_img_value"
        )
        txt_query = QwenImageChituAttnProcessor._ensure_sequence_major(
            txt_query, encoder_hidden_states.shape[1], attn.heads, "cubic_txt_query"
        )
        txt_key = QwenImageChituAttnProcessor._ensure_sequence_major(
            txt_key, encoder_hidden_states.shape[1], attn.heads, "cubic_txt_key"
        )
        txt_value = QwenImageChituAttnProcessor._ensure_sequence_major(
            txt_value, encoder_hidden_states.shape[1], attn.heads, "cubic_txt_value"
        )

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            active_idx = runtime["active_idx"]
            img_freqs = img_freqs.index_select(0, active_idx.to(img_freqs.device))
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
            img_query = QwenImageChituAttnProcessor._ensure_sequence_major(
                img_query, hidden_states.shape[1], attn.heads, "cubic_img_query_rope"
            )
            img_key = QwenImageChituAttnProcessor._ensure_sequence_major(
                img_key, hidden_states.shape[1], attn.heads, "cubic_img_key_rope"
            )
            txt_query = QwenImageChituAttnProcessor._ensure_sequence_major(
                txt_query, encoder_hidden_states.shape[1], attn.heads, "cubic_txt_query_rope"
            )
            txt_key = QwenImageChituAttnProcessor._ensure_sequence_major(
                txt_key, encoder_hidden_states.shape[1], attn.heads, "cubic_txt_key_rope"
            )

        full_img_key, full_img_value = self.engine._update_and_get_image_kv_cache(
            cache_key=runtime["cache_key"],
            layer_idx=self.layer_idx,
            key_active=img_key,
            value_active=img_value,
            active_idx=runtime["active_idx"],
            total_tokens=runtime["total_tokens"],
        )

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, full_img_key], dim=1)
        joint_value = torch.cat([txt_value, full_img_value], dim=1)
        target_dtype = joint_value.dtype
        joint_hidden_states = self.attn_backend(
            joint_query.to(target_dtype),
            joint_key.to(target_dtype),
            joint_value,
            causal=False,
        )[0]
        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt]
        img_attn_output = joint_hidden_states[:, seq_txt:]

        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
        return img_attn_output, txt_attn_output


class QwenImageCubicSelectiveForwardEngine:
    """Qwen-Image Cubic selective forward with full image-token K/V caches."""

    def __init__(self, config=None):
        self.config = config
        self.output_cache: Dict[str, torch.Tensor] = {}
        self.k_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.v_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.residual_cache = self.output_cache
        self.last_forward_mode = "full_refresh"
        self._runtime: Optional[dict] = None
        self._patched_transformer_id: Optional[int] = None
        self._original_processors: Optional[dict] = None
        self._expected_layer_count = 0

    def reset(self) -> None:
        self.output_cache.clear()
        self.k_cache.clear()
        self.v_cache.clear()
        self.last_forward_mode = "reset"
        self._runtime = None

    def restore_attn_processors(self, transformer) -> None:
        if self._original_processors is not None and hasattr(transformer, "set_attn_processor"):
            transformer.set_attn_processor(dict(self._original_processors))
        self._original_processors = None
        self._patched_transformer_id = None
        self._expected_layer_count = 0
        self._runtime = None

    def forward(
        self,
        transformer,
        original_forward,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        img_shapes,
        attention_kwargs,
        controlnet_block_samples=None,
        additional_t_cond=None,
        step_plan=None,
        cache_key: str = "qwen",
    ) -> torch.Tensor:
        if controlnet_block_samples is not None:
            self.last_forward_mode = "original_forward"
            return original_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                img_shapes=img_shapes,
                attention_kwargs=attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                additional_t_cond=additional_t_cond,
                return_dict=False,
            )[0]

        if step_plan is None or step_plan.is_full_compute:
            self.last_forward_mode = "full_refresh"
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                original_forward=original_forward,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                img_shapes=img_shapes,
                attention_kwargs=attention_kwargs,
                additional_t_cond=additional_t_cond,
                cache_key=cache_key,
            )

        batch_size, seq_len, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError("Qwen-Image Cubic selective forward currently supports batch size 1 only.")
        active_idx = step_plan.active_token_indices.long().to(hidden_states.device)
        out_cache = self.output_cache.get(cache_key)
        if active_idx.numel() == 0:
            if out_cache is not None and out_cache.shape == hidden_states.shape:
                self.last_forward_mode = "cached_residual"
                return out_cache.to(device=hidden_states.device, dtype=hidden_states.dtype).clone()
            raise RuntimeError("Qwen-Image Cubic selected zero active tokens without a valid output cache.")
        if int(active_idx.numel()) >= seq_len or int(active_idx.max().item()) >= seq_len:
            self.last_forward_mode = "full_refresh"
            return self._full_forward_and_refresh_cache(
                transformer,
                original_forward,
                hidden_states,
                timestep,
                guidance,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                img_shapes,
                attention_kwargs,
                additional_t_cond,
                cache_key,
            )
        if out_cache is None or out_cache.shape != hidden_states.shape or not self._has_kv_cache(transformer, cache_key, seq_len):
            self.last_forward_mode = "full_refresh"
            return self._full_forward_and_refresh_cache(
                transformer,
                original_forward,
                hidden_states,
                timestep,
                guidance,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                img_shapes,
                attention_kwargs,
                additional_t_cond,
                cache_key,
            )

        hidden_active = hidden_states.index_select(1, active_idx)
        self.last_forward_mode = "selective"
        out_active = self._run_transformer_with_kv_cache(
            transformer=transformer,
            original_forward=original_forward,
            hidden_states=hidden_active,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            additional_t_cond=additional_t_cond,
            cache_key=cache_key,
            active_idx=active_idx,
            total_tokens=seq_len,
        )
        out_full = out_cache.to(device=hidden_states.device, dtype=out_active.dtype).clone()
        out_full[:, active_idx, :] = out_active
        self.output_cache[cache_key] = out_full.detach()
        return out_full

    def _full_forward_and_refresh_cache(
        self,
        transformer,
        original_forward,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        img_shapes,
        attention_kwargs,
        additional_t_cond,
        cache_key: str,
    ) -> torch.Tensor:
        seq_len = int(hidden_states.shape[1])
        all_idx = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
        out = self._run_transformer_with_kv_cache(
            transformer=transformer,
            original_forward=original_forward,
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            additional_t_cond=additional_t_cond,
            cache_key=cache_key,
            active_idx=all_idx,
            total_tokens=seq_len,
        )
        self.output_cache[cache_key] = out.detach()
        return out

    def _ensure_cubic_attn_processors(self, transformer) -> None:
        transformer_id = id(transformer)
        if self._patched_transformer_id == transformer_id:
            return
        if self._original_processors is None and hasattr(transformer, "attn_processors"):
            self._original_processors = dict(transformer.attn_processors)
        fallback_backend = self._infer_fallback_backend(transformer)
        if fallback_backend is None:
            raise RuntimeError("Qwen-Image Cubic requires the Chitu attention processor to be installed first.")
        for idx, block in enumerate(transformer.transformer_blocks):
            block.attn.set_processor(QwenImageCubicAttnProcessor(self, layer_idx=idx, fallback_backend=fallback_backend))
        self._expected_layer_count = len(transformer.transformer_blocks)
        self._patched_transformer_id = transformer_id

    def _run_transformer_with_kv_cache(
        self,
        transformer,
        original_forward,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        img_shapes,
        attention_kwargs,
        additional_t_cond,
        cache_key: str,
        active_idx: torch.Tensor,
        total_tokens: int,
    ) -> torch.Tensor:
        self._ensure_cubic_attn_processors(transformer)
        self._runtime = {
            "cache_key": cache_key,
            "active_idx": active_idx,
            "total_tokens": int(total_tokens),
        }
        try:
            return original_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                img_shapes=img_shapes,
                attention_kwargs=attention_kwargs,
                additional_t_cond=additional_t_cond,
                return_dict=False,
            )[0]
        finally:
            self._runtime = None

    def _update_and_get_image_kv_cache(
        self,
        cache_key: str,
        layer_idx: int,
        key_active: torch.Tensor,
        value_active: torch.Tensor,
        active_idx: torch.Tensor,
        total_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key_tuple = (cache_key, layer_idx)
        k_cache = self.k_cache.get(cache_key_tuple)
        v_cache = self.v_cache.get(cache_key_tuple)
        invalid = (
            k_cache is None
            or v_cache is None
            or k_cache.shape[0] != key_active.shape[0]
            or k_cache.shape[1] != total_tokens
            or k_cache.shape[2] != key_active.shape[2]
            or k_cache.shape[3] != key_active.shape[3]
            or v_cache.shape[0] != value_active.shape[0]
            or v_cache.shape[1] != total_tokens
            or v_cache.shape[2] != value_active.shape[2]
            or v_cache.shape[3] != value_active.shape[3]
        )
        if invalid:
            k_cache = torch.zeros(
                key_active.shape[0],
                total_tokens,
                key_active.shape[2],
                key_active.shape[3],
                dtype=key_active.dtype,
                device=key_active.device,
            )
            v_cache = torch.zeros(
                value_active.shape[0],
                total_tokens,
                value_active.shape[2],
                value_active.shape[3],
                dtype=value_active.dtype,
                device=value_active.device,
            )
        else:
            k_cache = k_cache.to(device=key_active.device, dtype=key_active.dtype)
            v_cache = v_cache.to(device=value_active.device, dtype=value_active.dtype)
        k_cache[:, active_idx, :, :] = key_active
        v_cache[:, active_idx, :, :] = value_active
        self.k_cache[cache_key_tuple] = k_cache.detach()
        self.v_cache[cache_key_tuple] = v_cache.detach()
        return k_cache, v_cache

    def _has_kv_cache(self, transformer, cache_key: str, total_tokens: int) -> bool:
        self._ensure_cubic_attn_processors(transformer)
        for layer_idx in range(self._expected_layer_count):
            k = self.k_cache.get((cache_key, layer_idx))
            v = self.v_cache.get((cache_key, layer_idx))
            if k is None or v is None:
                return False
            if int(k.shape[1]) != int(total_tokens) or int(v.shape[1]) != int(total_tokens):
                return False
        return True

    @staticmethod
    def _infer_fallback_backend(transformer):
        for block in transformer.transformer_blocks:
            processor = getattr(block.attn, "processor", None)
            backend = getattr(processor, "attn_backend", None)
            if backend is not None:
                return backend
        return None

    @staticmethod
    def unpack_qwen_sequence(sequence: torch.Tensor, img_shapes) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[0] != 1:
            raise ValueError(f"Qwen-Image Cubic expects [1, seq, channels], got {tuple(sequence.shape)}.")
        if img_shapes is None or len(img_shapes) != 1 or len(img_shapes[0]) != 1:
            seq_len = int(sequence.shape[1])
            side = int(seq_len**0.5)
            if side * side != seq_len:
                raise ValueError(f"Qwen-Image Cubic cannot infer square token grid for seq_len={seq_len}.")
            frame, height, width = 1, side, side
        else:
            frame, height, width = [int(v) for v in img_shapes[0][0]]
        return sequence[0].transpose(0, 1).reshape(sequence.shape[-1], frame, height, width)
