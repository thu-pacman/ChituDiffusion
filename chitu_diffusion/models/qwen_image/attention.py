from __future__ import annotations

from weakref import WeakKeyDictionary

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    apply_rotary_emb_qwen,
)

from ...parallel.cp import EpeParallelContext, ImageContextParallelAttention


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    return (
        tensor.data_ptr(),
        version,
        tensor.shape,
        tensor.stride(),
        tensor.device,
        tensor.dtype,
    )


class QwenImageCpAttnProcessor:
    """Qwen joint attention over local image and optional local text shards."""

    def __init__(
        self,
        parallel: EpeParallelContext,
        *,
        mode: str = "agkv",
        planned_attention: bool = False,
        sharded_text: bool = False,
    ) -> None:
        self.parallel = parallel
        self.attention = ImageContextParallelAttention(
            mode=mode,
            planned=planned_attention,
        )
        self.native = QwenDoubleStreamAttnProcessor2_0()
        self.sharded_text = bool(sharded_text)
        self._packed_qkv = WeakKeyDictionary()
        self._enabled = False
        self._active_sharded_text = False

    @staticmethod
    def _pack_projections(projections):
        weight = torch.cat(
            [projection.weight for projection in projections]
        ).contiguous()
        biases = [projection.bias for projection in projections]
        if any(bias is None for bias in biases) != all(
            bias is None for bias in biases
        ):
            raise ValueError("Qwen Q/K/V projections must agree on bias presence")
        bias = None if biases[0] is None else torch.cat(biases).contiguous()
        return weight, bias

    @staticmethod
    def _projection_signature(attn) -> tuple:
        projections = (
            attn.to_q,
            attn.to_k,
            attn.to_v,
            attn.add_q_proj,
            attn.add_k_proj,
            attn.add_v_proj,
        )
        return tuple(
            (
                _tensor_signature(projection.weight),
                None
                if projection.bias is None
                else _tensor_signature(projection.bias),
            )
            for projection in projections
        )

    def _packed_qkv_for(self, attn):
        signature = self._projection_signature(attn)
        cached = self._packed_qkv.get(attn)
        if cached is not None and cached[0] == signature:
            return cached[1:]
        image_weight, image_bias = self._pack_projections(
            (attn.to_q, attn.to_k, attn.to_v)
        )
        text_weight, text_bias = self._pack_projections(
            (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)
        )
        packed = (signature, image_weight, image_bias, text_weight, text_bias)
        self._packed_qkv[attn] = packed
        return packed[1:]

    def enable(self, *, sharded_text: bool | None = None) -> None:
        self._active_sharded_text = (
            self.sharded_text if sharded_text is None else bool(sharded_text)
        )
        self._enabled = self._active_sharded_text or self.parallel.active.width > 1

    def disable(self) -> None:
        self._enabled = False
        self._active_sharded_text = False

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

        joint_qkv = None
        if self._active_sharded_text:
            from ...parallel.cp.fast.projection.qk_epilogue import joint_qkv_norm_rope

            image_weight, image_bias, text_weight, text_bias = self._packed_qkv_for(attn)
            joint_qkv = joint_qkv_norm_rope(
                F.linear(hidden_states, image_weight, image_bias),
                F.linear(encoder_hidden_states, text_weight, text_bias),
                attn,
                image_rotary_emb,
            )
            text_tokens = encoder_hidden_states.shape[1]
            txt_query, txt_key, txt_value = (t[:, :text_tokens] for t in joint_qkv)
            img_query, img_key, img_value = (t[:, text_tokens:] for t in joint_qkv)
        else:
            img_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
            img_key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
            img_value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
            txt_query = attn.add_q_proj(encoder_hidden_states).unflatten(
                -1, (attn.heads, -1)
            )
            txt_key = attn.add_k_proj(encoder_hidden_states).unflatten(
                -1, (attn.heads, -1)
            )
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
                img_query = apply_rotary_emb_qwen(
                    img_query, img_freqs, use_real=False
                )
                img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
                txt_query = apply_rotary_emb_qwen(
                    txt_query, txt_freqs, use_real=False
                )
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
            joint_sharded=self._active_sharded_text,
            joint_qkv=joint_qkv,
        )
        image_output = image_output.flatten(2, 3).to(output_dtype)
        text_output = text_output.flatten(2, 3).to(output_dtype)
        image_output = attn.to_out[0](image_output.contiguous())
        if len(attn.to_out) > 1:
            image_output = attn.to_out[1](image_output)
        text_output = attn.to_add_out(text_output.contiguous())
        return image_output, text_output
