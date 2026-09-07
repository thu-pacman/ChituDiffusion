from __future__ import annotations

import torch
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    apply_rotary_emb_qwen,
)

from ...parallel.cp import EpeParallelContext, ImageContextParallelAttention


class QwenImageCpAttnProcessor:
    """Qwen joint attention with replicated text and sharded image tokens."""

    def __init__(self, parallel: EpeParallelContext, *, mode: str = "agkv") -> None:
        self.parallel = parallel
        self.attention = ImageContextParallelAttention(mode=mode)
        self.native = QwenDoubleStreamAttnProcessor2_0()
        self._enabled = False

    def enable(self) -> None:
        self._enabled = self.parallel.active.width > 1

    def disable(self) -> None:
        self._enabled = False

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ):
        if not self._enabled:
            return self.native(
                attn,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                attention_mask,
                image_rotary_emb,
            )
        if encoder_hidden_states is None:
            raise ValueError("Qwen-Image CP requires encoder_hidden_states")
        if attention_mask is not None and not bool(attention_mask.to(torch.bool).all()):
            raise NotImplementedError(
                "Qwen-Image EPE currently requires an all-valid text mask"
            )

        img_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        img_key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        img_value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        txt_query = attn.add_q_proj(encoder_hidden_states).unflatten(
            -1, (attn.heads, -1)
        )
        txt_key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_value = attn.add_v_proj(encoder_hidden_states).unflatten(
            -1, (attn.heads, -1)
        )

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        output_dtype = img_query.dtype
        target_dtype = img_value.dtype
        text_output, image_output = self.attention(
            img_query.to(target_dtype).contiguous(),
            img_key.to(target_dtype).contiguous(),
            img_value.contiguous(),
            txt_query.to(target_dtype).contiguous(),
            txt_key.to(target_dtype).contiguous(),
            txt_value.contiguous(),
            lane_process_group=self.parallel.active.process_group,
            ulysses_topology=(
                self.parallel.active_ulysses
                if self.attention.mode == "ulysses"
                else None
            ),
            agkv_transport=(
                self.parallel.active_agkv_transport
                if self.attention.mode == "agkv"
                else None
            ),
            joint_first=True,
        )
        image_output = image_output.flatten(2, 3).to(output_dtype)
        text_output = text_output.flatten(2, 3).to(output_dtype)
        image_output = attn.to_out[0](image_output.contiguous())
        if len(attn.to_out) > 1:
            image_output = attn.to_out[1](image_output)
        text_output = attn.to_add_out(text_output.contiguous())
        return image_output, text_output
