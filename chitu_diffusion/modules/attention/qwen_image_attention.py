from __future__ import annotations

from typing import Optional

import torch

from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen


class QwenImageChituAttnProcessor:
    """Qwen-Image joint text/image attention processor backed by Chitu attention."""

    def __init__(self, attn_backend):
        self.attn_backend = attn_backend

    @staticmethod
    def _slice_rotary_freqs(freqs: torch.Tensor, offset: int, length: int) -> torch.Tensor:
        available = max(0, freqs.shape[0] - offset)
        sliced = freqs.narrow(0, offset, min(length, available))
        if sliced.shape[0] == length:
            return sliced
        pad_shape = list(freqs.shape)
        pad_shape[0] = length - sliced.shape[0]
        padding = torch.zeros(pad_shape, dtype=freqs.dtype, device=freqs.device)
        return torch.cat([sliced, padding], dim=0)

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
        if encoder_hidden_states is None:
            raise ValueError("QwenImageChituAttnProcessor requires encoder_hidden_states.")
        if attention_mask is not None:
            raise NotImplementedError("Qwen-Image Chitu attention backend does not support attention_mask yet.")

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

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            if qwen_cp_info is not None:
                img_offset = int(qwen_cp_info.get("image_offset", 0))
                txt_offset = int(qwen_cp_info.get("text_offset", 0))
                img_freqs = self._slice_rotary_freqs(img_freqs, img_offset, img_query.shape[1])
                txt_freqs = self._slice_rotary_freqs(txt_freqs, txt_offset, txt_query.shape[1])
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_hidden_states = self.attn_backend(
            joint_query,
            joint_key,
            joint_value,
            causal=False,
        )[0]
        joint_hidden_states = joint_hidden_states.reshape(batch_size, -1, attn.heads * joint_query.shape[-1])
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt]
        img_attn_output = joint_hidden_states[:, seq_txt:]

        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output
