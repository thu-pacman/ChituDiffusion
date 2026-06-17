from __future__ import annotations

from logging import getLogger
from typing import Optional

import torch
import torch.distributed as dist

from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

from chitu_diffusion.core.distributed.parallel_state import get_cp_group

logger = getLogger(__name__)


class QwenImageChituAttnProcessor:
    """Qwen-Image joint text/image attention processor backed by Chitu attention."""

    def __init__(self, attn_backend):
        self.attn_backend = attn_backend
        self.base_attn_backend = getattr(attn_backend, "attn", attn_backend)

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

    @staticmethod
    def _all_gather_sequence(tensor: torch.Tensor) -> torch.Tensor:
        group = get_cp_group()
        pieces = [torch.empty_like(tensor) for _ in range(group.group_size)]
        dist.all_gather(pieces, tensor.contiguous(), group=group.gpu_group)
        return torch.cat(pieces, dim=1)

    @staticmethod
    def _ensure_sequence_major(tensor: torch.Tensor, seq_len: int, heads: int, name: str) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(f"{name} must be a 4D attention tensor, got shape={tuple(tensor.shape)}.")
        if tensor.shape[1] == seq_len and tensor.shape[2] == heads:
            return tensor
        if tensor.shape[1] == heads and tensor.shape[2] == seq_len:
            return tensor.transpose(1, 2).contiguous()
        raise ValueError(
            f"{name} must be [batch, seq, heads, dim] or [batch, heads, seq, dim], "
            f"got shape={tuple(tensor.shape)}, seq_len={seq_len}, heads={heads}."
        )

    @staticmethod
    def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
        return tuple(tensor.shape)

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

        img_query = self._ensure_sequence_major(img_query, hidden_states.shape[1], attn.heads, "img_query")
        img_key = self._ensure_sequence_major(img_key, hidden_states.shape[1], attn.heads, "img_key")
        img_value = self._ensure_sequence_major(img_value, hidden_states.shape[1], attn.heads, "img_value")
        txt_query = self._ensure_sequence_major(txt_query, encoder_hidden_states.shape[1], attn.heads, "txt_query")
        txt_key = self._ensure_sequence_major(txt_key, encoder_hidden_states.shape[1], attn.heads, "txt_key")
        txt_value = self._ensure_sequence_major(txt_value, encoder_hidden_states.shape[1], attn.heads, "txt_value")

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            if qwen_cp_info is not None:
                img_offset = int(qwen_cp_info.get("image_offset", 0))
                img_freqs = self._slice_rotary_freqs(img_freqs, img_offset, img_query.shape[1])
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
            img_query = self._ensure_sequence_major(img_query, hidden_states.shape[1], attn.heads, "img_query_rope")
            img_key = self._ensure_sequence_major(img_key, hidden_states.shape[1], attn.heads, "img_key_rope")
            txt_query = self._ensure_sequence_major(
                txt_query, encoder_hidden_states.shape[1], attn.heads, "txt_query_rope"
            )
            txt_key = self._ensure_sequence_major(txt_key, encoder_hidden_states.shape[1], attn.heads, "txt_key_rope")

        if qwen_cp_info is not None:
            full_img_key = self._all_gather_sequence(img_key)
            full_img_value = self._all_gather_sequence(img_value)
            image_seq_len = int(qwen_cp_info.get("image_seq_len", full_img_key.shape[1]))
            full_img_key = full_img_key[:, :image_seq_len]
            full_img_value = full_img_value[:, :image_seq_len]
            try:
                joint_query = torch.cat([txt_query, img_query], dim=1)
                joint_key = torch.cat([txt_key, full_img_key], dim=1)
                joint_value = torch.cat([txt_value, full_img_value], dim=1)
            except RuntimeError:
                logger.exception(
                    "Qwen attention shape mismatch rank=%s qwen_cp_info=%s txt_q=%s img_q=%s txt_k=%s full_img_k=%s "
                    "txt_v=%s full_img_v=%s",
                    dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
                    qwen_cp_info,
                    self._shape(txt_query),
                    self._shape(img_query),
                    self._shape(txt_key),
                    self._shape(full_img_key),
                    self._shape(txt_value),
                    self._shape(full_img_value),
                )
                raise
        else:
            try:
                joint_query = torch.cat([txt_query, img_query], dim=1)
                joint_key = torch.cat([txt_key, img_key], dim=1)
                joint_value = torch.cat([txt_value, img_value], dim=1)
            except RuntimeError:
                logger.exception(
                    "Qwen attention shape mismatch rank=%s txt_q=%s img_q=%s txt_k=%s img_k=%s txt_v=%s img_v=%s",
                    dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
                    self._shape(txt_query),
                    self._shape(img_query),
                    self._shape(txt_key),
                    self._shape(img_key),
                    self._shape(txt_value),
                    self._shape(img_value),
                )
                raise
        target_dtype = joint_value.dtype
        joint_query = joint_query.to(target_dtype)
        joint_key = joint_key.to(target_dtype)

        attn_backend = self.base_attn_backend if qwen_cp_info is not None else self.attn_backend
        joint_hidden_states = attn_backend(
            joint_query,
            joint_key,
            joint_value,
            causal=False,
        )[0]
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt]
        img_attn_output = joint_hidden_states[:, seq_txt:]

        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output
