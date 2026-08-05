from __future__ import annotations

import os
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

        seq_txt = encoder_hidden_states.shape[1]

        def _img_q():
            x = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
            if attn.norm_q is not None:
                x = attn.norm_q(x)
            x = self._ensure_sequence_major(x, hidden_states.shape[1], attn.heads, "img_query")
            if image_rotary_emb is not None:
                img_freqs, _ = image_rotary_emb
                if qwen_cp_info is not None:
                    img_offset = int(qwen_cp_info.get("image_offset", 0))
                    img_freqs = self._slice_rotary_freqs(img_freqs, img_offset, x.shape[1])
                x = apply_rotary_emb_qwen(x, img_freqs, use_real=False)
                x = self._ensure_sequence_major(x, hidden_states.shape[1], attn.heads, "img_query_rope")
            return x

        def _img_k():
            x = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
            if attn.norm_k is not None:
                x = attn.norm_k(x)
            x = self._ensure_sequence_major(x, hidden_states.shape[1], attn.heads, "img_key")
            if image_rotary_emb is not None:
                img_freqs, _ = image_rotary_emb
                if qwen_cp_info is not None:
                    img_offset = int(qwen_cp_info.get("image_offset", 0))
                    img_freqs = self._slice_rotary_freqs(img_freqs, img_offset, x.shape[1])
                x = apply_rotary_emb_qwen(x, img_freqs, use_real=False)
                x = self._ensure_sequence_major(x, hidden_states.shape[1], attn.heads, "img_key_rope")
            return x

        def _img_v():
            x = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
            return self._ensure_sequence_major(x, hidden_states.shape[1], attn.heads, "img_value")

        def _txt_q():
            x = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            if attn.norm_added_q is not None:
                x = attn.norm_added_q(x)
            x = self._ensure_sequence_major(x, encoder_hidden_states.shape[1], attn.heads, "txt_query")
            if image_rotary_emb is not None:
                _, txt_freqs = image_rotary_emb
                x = apply_rotary_emb_qwen(x, txt_freqs, use_real=False)
                x = self._ensure_sequence_major(x, encoder_hidden_states.shape[1], attn.heads, "txt_query_rope")
            return x

        def _txt_k():
            x = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            if attn.norm_added_k is not None:
                x = attn.norm_added_k(x)
            x = self._ensure_sequence_major(x, encoder_hidden_states.shape[1], attn.heads, "txt_key")
            if image_rotary_emb is not None:
                _, txt_freqs = image_rotary_emb
                x = apply_rotary_emb_qwen(x, txt_freqs, use_real=False)
                x = self._ensure_sequence_major(x, encoder_hidden_states.shape[1], attn.heads, "txt_key_rope")
            return x

        def _txt_v():
            x = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            return self._ensure_sequence_major(x, encoder_hidden_states.shape[1], attn.heads, "txt_value")

        use_overlap_qkv = (
            qwen_cp_info is not None
            and hidden_states.is_cuda
            and hasattr(self.attn_backend, "cp_attn_with_full_txt_fused")
            and os.environ.get("CHITU_CP_OVERLAP_QKV", "0") == "1"
            and (
                not hasattr(self.attn_backend, "supports_fused_overlap")
                or self.attn_backend.supports_fused_overlap(num_heads=attn.heads)
            )
        )
        if use_overlap_qkv:
            image_seq_len = int(qwen_cp_info.get("image_seq_len", hidden_states.shape[1] * get_cp_group().group_size))
            target_dtype = hidden_states.dtype

            def produce_img_q():
                return _img_q().to(target_dtype).contiguous()

            def produce_img_k():
                return _img_k().to(target_dtype).contiguous()

            def produce_img_v():
                return _img_v().contiguous()

            def produce_txt_q():
                return _txt_q().to(target_dtype).contiguous()

            def produce_txt_k():
                return _txt_k().to(target_dtype).contiguous()

            def produce_txt_v():
                return _txt_v().contiguous()

            txt_attn_output, img_attn_output = self.attn_backend.cp_attn_with_full_txt_fused(
                produce_txt_q=produce_txt_q,
                produce_txt_k=produce_txt_k,
                produce_txt_v=produce_txt_v,
                produce_img_q=produce_img_q,
                produce_img_k=produce_img_k,
                produce_img_v=produce_img_v,
                image_seq_len=image_seq_len,
                causal=False,
                num_heads=attn.heads,
            )
            img_attn_output = img_attn_output.flatten(2, 3).to(hidden_states.dtype)
            txt_attn_output = txt_attn_output.flatten(2, 3).to(hidden_states.dtype)
            img_attn_output = attn.to_out[0](img_attn_output.contiguous())
            if len(attn.to_out) > 1:
                img_attn_output = attn.to_out[1](img_attn_output)
            txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
            return img_attn_output, txt_attn_output

        img_query = _img_q()
        img_key = _img_k()
        img_value = _img_v()

        txt_query = _txt_q()
        txt_key = _txt_k()
        txt_value = _txt_v()

        if qwen_cp_info is not None:
            image_seq_len = int(qwen_cp_info.get("image_seq_len", img_key.shape[1] * get_cp_group().group_size))
            if hasattr(self.attn_backend, "cp_attn_with_full_txt"):
                target_dtype = img_value.dtype
                txt_query = txt_query.to(target_dtype)
                txt_key = txt_key.to(target_dtype)
                img_query = img_query.to(target_dtype)
                img_key = img_key.to(target_dtype)
                txt_attn_output, img_attn_output = self.attn_backend.cp_attn_with_full_txt(
                    txt_query,
                    txt_key,
                    txt_value,
                    img_query,
                    img_key,
                    img_value,
                    image_seq_len=image_seq_len,
                    causal=False,
                )
                joint_query_dtype = txt_query.dtype
                txt_attn_output = txt_attn_output.flatten(2, 3).to(joint_query_dtype)
                img_attn_output = img_attn_output.flatten(2, 3).to(joint_query_dtype)

                img_attn_output = attn.to_out[0](img_attn_output.contiguous())
                if len(attn.to_out) > 1:
                    img_attn_output = attn.to_out[1](img_attn_output)
                txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
                return img_attn_output, txt_attn_output
            else:
                full_img_key = self._all_gather_sequence(img_key)
                full_img_value = self._all_gather_sequence(img_value)
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
