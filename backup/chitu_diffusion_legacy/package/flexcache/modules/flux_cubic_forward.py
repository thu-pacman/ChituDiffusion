from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb


def _slice_rotary_emb(image_rotary_emb, start: int, end: Optional[int] = None):
    if image_rotary_emb is None:
        return None
    if isinstance(image_rotary_emb, tuple):
        cos, sin = image_rotary_emb
        return cos[start:end], sin[start:end]
    return image_rotary_emb[start:end]


class FluxCubicKVCacheAttnProcessor:
    """Flux attention processor that updates only active image-token KV."""

    def __init__(self, engine: "FluxCubicSelectiveForwardEngine", layer_kind: str, layer_idx: int):
        self.engine = engine
        self.layer_kind = layer_kind
        self.layer_idx = layer_idx

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        runtime = self.engine._runtime
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            context_query = attn.add_q_proj(encoder_hidden_states)
            context_key = attn.add_k_proj(encoder_hidden_states)
            context_value = attn.add_v_proj(encoder_hidden_states)

            context_query = context_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            context_key = context_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            context_value = context_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                context_query = attn.norm_added_q(context_query)
            if attn.norm_added_k is not None:
                context_key = attn.norm_added_k(context_key)

            if image_rotary_emb is not None:
                text_len = encoder_hidden_states.shape[1]
                query = apply_rotary_emb(query, _slice_rotary_emb(image_rotary_emb, text_len, None))
                key = apply_rotary_emb(key, _slice_rotary_emb(image_rotary_emb, text_len, None))
                context_query = apply_rotary_emb(context_query, _slice_rotary_emb(image_rotary_emb, 0, text_len))
                context_key = apply_rotary_emb(context_key, _slice_rotary_emb(image_rotary_emb, 0, text_len))

            if runtime is not None:
                key, value = self.engine._update_and_get_image_kv_cache(
                    cache_key=runtime["cache_key"],
                    layer_key=(self.layer_kind, self.layer_idx),
                    key_active=key,
                    value_active=value,
                    active_idx=runtime["active_idx"],
                    total_tokens=runtime["total_tokens"],
                )

            query = torch.cat([context_query, query], dim=2)
            key = torch.cat([context_key, key], dim=2)
            value = torch.cat([context_value, value], dim=2)

            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            context_out, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            context_out = attn.to_add_out(context_out)
            return hidden_states, context_out

        if image_rotary_emb is not None:
            if runtime is not None:
                encoder_len = int(runtime["encoder_tokens"])
                if encoder_len < 0 or encoder_len > int(query.shape[2]):
                    raise ValueError(
                        f"Invalid Flux Cubic encoder token length: encoder_len={encoder_len}, seq={int(query.shape[2])}"
                    )

                q_text = query[:, :, :encoder_len, :]
                q_image = query[:, :, encoder_len:, :]
                k_text = key[:, :, :encoder_len, :]
                k_image = key[:, :, encoder_len:, :]
                v_text = value[:, :, :encoder_len, :]
                v_image = value[:, :, encoder_len:, :]

                q_text = apply_rotary_emb(q_text, _slice_rotary_emb(image_rotary_emb, 0, encoder_len))
                k_text = apply_rotary_emb(k_text, _slice_rotary_emb(image_rotary_emb, 0, encoder_len))
                q_image = apply_rotary_emb(q_image, _slice_rotary_emb(image_rotary_emb, encoder_len, None))
                k_image = apply_rotary_emb(k_image, _slice_rotary_emb(image_rotary_emb, encoder_len, None))

                k_image, v_image = self.engine._update_and_get_image_kv_cache(
                    cache_key=runtime["cache_key"],
                    layer_key=(self.layer_kind, self.layer_idx),
                    key_active=k_image,
                    value_active=v_image,
                    active_idx=runtime["active_idx"],
                    total_tokens=runtime["total_tokens"],
                )
                query = torch.cat([q_text, q_image], dim=2)
                key = torch.cat([k_text, k_image], dim=2)
                value = torch.cat([v_text, v_image], dim=2)
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        return hidden_states.to(query.dtype)


class FluxCubicSelectiveForwardEngine:
    """Flux-specific Cubic selective forward with full image-token KV caches."""

    def __init__(self, config=None):
        self.config = config
        self.output_cache: dict[str, torch.Tensor] = {}
        self.k_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}
        self.v_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}
        self.residual_cache: dict[str, torch.Tensor] = self.output_cache
        self.last_forward_mode = "full_refresh"
        self._runtime: Optional[dict] = None
        self._patched_transformer_id: Optional[int] = None
        self._expected_layer_keys: list[Tuple[str, int]] = []
        self._original_processors: Optional[dict] = None

    def reset(self) -> None:
        self.output_cache.clear()
        self.k_cache.clear()
        self.v_cache.clear()
        self.last_forward_mode = "full_refresh"
        self._runtime = None

    def restore_attn_processors(self, transformer) -> None:
        if self._original_processors is not None and hasattr(transformer, "set_attn_processor"):
            transformer.set_attn_processor(dict(self._original_processors))
        self._original_processors = None
        self._patched_transformer_id = None
        self._expected_layer_keys = []
        self._runtime = None

    def forward(
        self,
        transformer,
        original_forward,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: Optional[torch.Tensor],
        joint_attention_kwargs,
        step_plan=None,
        cache_key: str = "flux",
    ) -> torch.Tensor:
        if step_plan is None or step_plan.is_full_compute:
            self.last_forward_mode = "full_refresh"
            return self._full_forward_and_refresh_cache(
                transformer=transformer,
                original_forward=original_forward,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                cache_key=cache_key,
            )

        batch_size, seq_len, _ = hidden_states.shape
        active_idx = step_plan.active_token_indices.long().to(hidden_states.device)
        out_cache = self.output_cache.get(cache_key)

        if active_idx.numel() == 0:
            if out_cache is not None and out_cache.shape == hidden_states.shape:
                self.last_forward_mode = "cached_residual"
                return out_cache.to(device=hidden_states.device, dtype=hidden_states.dtype).clone()
            raise RuntimeError("Flux Cubic selected zero active tokens without a valid output cache.")

        if int(active_idx.numel()) >= int(seq_len) or int(active_idx.max().item()) >= int(seq_len):
            raise ValueError(
                f"Flux Cubic selective step has invalid active tokens: active={int(active_idx.numel())}, seq={seq_len}"
            )

        if out_cache is None or out_cache.shape != hidden_states.shape:
            cached_shape = None if out_cache is None else tuple(out_cache.shape)
            raise RuntimeError(
                f"Flux Cubic selective step requires a valid output cache; got {cached_shape}, "
                f"expected {tuple(hidden_states.shape)}."
            )

        if not self._has_kv_cache_for_all_layers(transformer, cache_key=cache_key, total_tokens=seq_len):
            raise RuntimeError("Flux Cubic selective step requires valid KV caches for all Flux layers.")

        if img_ids is not None and int(img_ids.shape[0]) != int(seq_len):
            raise ValueError(f"Flux Cubic img_ids/token mismatch: img_ids={int(img_ids.shape[0])}, seq={seq_len}")

        hidden_active = hidden_states.index_select(1, active_idx)
        img_ids_active = img_ids.index_select(0, active_idx) if img_ids is not None else None
        self.last_forward_mode = "selective"
        out_active = self._run_transformer_with_kv_cache(
            transformer=transformer,
            original_forward=original_forward,
            hidden_states=hidden_active,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids_active,
            joint_attention_kwargs=joint_attention_kwargs,
            cache_key=cache_key,
            active_idx=active_idx,
            total_tokens=seq_len,
            encoder_tokens=int(txt_ids.shape[0]),
        )

        out_full = out_cache.to(device=hidden_states.device, dtype=out_active.dtype).clone()
        out_full[:, active_idx, :] = out_active
        self.output_cache[cache_key] = out_full.detach()
        logging.debug(
            "[flux-cubic] active_tokens=%d frozen_tokens=%d batch=%d",
            int(active_idx.numel()),
            int(seq_len - active_idx.numel()),
            int(batch_size),
        )
        return out_full

    def _full_forward_and_refresh_cache(
        self,
        transformer,
        original_forward,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: Optional[torch.Tensor],
        joint_attention_kwargs,
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
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            cache_key=cache_key,
            active_idx=all_idx,
            total_tokens=seq_len,
            encoder_tokens=int(txt_ids.shape[0]),
        )
        self.output_cache[cache_key] = out.detach()
        return out

    def _ensure_cubic_attn_processors(self, transformer) -> None:
        transformer_id = id(transformer)
        if self._patched_transformer_id == transformer_id:
            return
        if self._original_processors is None and hasattr(transformer, "attn_processors"):
            self._original_processors = dict(transformer.attn_processors)

        self._expected_layer_keys = []
        for idx, block in enumerate(transformer.transformer_blocks):
            block.attn.set_processor(FluxCubicKVCacheAttnProcessor(self, layer_kind="joint", layer_idx=idx))
            self._expected_layer_keys.append(("joint", idx))
        for idx, block in enumerate(transformer.single_transformer_blocks):
            block.attn.set_processor(FluxCubicKVCacheAttnProcessor(self, layer_kind="single", layer_idx=idx))
            self._expected_layer_keys.append(("single", idx))
        self._patched_transformer_id = transformer_id

    def _run_transformer_with_kv_cache(
        self,
        transformer,
        original_forward,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        pooled_projections: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: Optional[torch.Tensor],
        joint_attention_kwargs,
        cache_key: str,
        active_idx: torch.Tensor,
        total_tokens: int,
        encoder_tokens: int,
    ) -> torch.Tensor:
        self._ensure_cubic_attn_processors(transformer)
        self._runtime = {
            "cache_key": cache_key,
            "active_idx": active_idx,
            "total_tokens": int(total_tokens),
            "encoder_tokens": int(encoder_tokens),
        }
        try:
            return original_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]
        finally:
            self._runtime = None

    def _update_and_get_image_kv_cache(
        self,
        cache_key: str,
        layer_key: Tuple[str, int],
        key_active: torch.Tensor,
        value_active: torch.Tensor,
        active_idx: torch.Tensor,
        total_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_k_key = (cache_key, layer_key[0], layer_key[1])
        cache_v_key = (cache_key, layer_key[0], layer_key[1])
        k_cache = self.k_cache.get(cache_k_key)
        v_cache = self.v_cache.get(cache_v_key)

        invalid = (
            k_cache is None
            or v_cache is None
            or k_cache.shape[0] != key_active.shape[0]
            or k_cache.shape[1] != key_active.shape[1]
            or k_cache.shape[2] != total_tokens
            or k_cache.shape[3] != key_active.shape[3]
            or v_cache.shape[0] != value_active.shape[0]
            or v_cache.shape[1] != value_active.shape[1]
            or v_cache.shape[2] != total_tokens
            or v_cache.shape[3] != value_active.shape[3]
        )
        if invalid:
            k_cache = torch.zeros(
                key_active.shape[0],
                key_active.shape[1],
                total_tokens,
                key_active.shape[3],
                dtype=key_active.dtype,
                device=key_active.device,
            )
            v_cache = torch.zeros(
                value_active.shape[0],
                value_active.shape[1],
                total_tokens,
                value_active.shape[3],
                dtype=value_active.dtype,
                device=value_active.device,
            )
        else:
            k_cache = k_cache.to(device=key_active.device, dtype=key_active.dtype)
            v_cache = v_cache.to(device=value_active.device, dtype=value_active.dtype)

        k_cache[:, :, active_idx, :] = key_active
        v_cache[:, :, active_idx, :] = value_active
        self.k_cache[cache_k_key] = k_cache.detach()
        self.v_cache[cache_v_key] = v_cache.detach()
        return k_cache, v_cache

    def _has_kv_cache_for_all_layers(self, transformer, cache_key: str, total_tokens: int) -> bool:
        self._ensure_cubic_attn_processors(transformer)
        for layer_kind, layer_idx in self._expected_layer_keys:
            k = self.k_cache.get((cache_key, layer_kind, layer_idx))
            v = self.v_cache.get((cache_key, layer_kind, layer_idx))
            if k is None or v is None:
                return False
            if int(k.shape[2]) != int(total_tokens) or int(v.shape[2]) != int(total_tokens):
                return False
        return True
