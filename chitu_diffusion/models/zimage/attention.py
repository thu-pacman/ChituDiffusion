from __future__ import annotations

from typing import Optional
from weakref import WeakKeyDictionary

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

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


class ZImageCpAttnProcessor:
    """Diffusers projection/RoPE glue around shared CP attention primitives."""

    def __init__(
        self,
        parallel: EpeParallelContext,
        *,
        mode: str = "agkv",
        planned_attention: bool = False,
        sharded_text: bool = False,
    ) -> None:
        self.parallel = parallel
        self.context_parallel = ImageContextParallelAttention(
            mode=mode,
            planned=planned_attention,
        )
        self._image_tokens = 0
        self._enabled = False
        self.sharded_text = bool(sharded_text)
        self._active_sharded_text = False
        self._packed_qkv = WeakKeyDictionary()

    def _packed_qkv_for(self, attn: Attention):
        projections = (attn.to_q, attn.to_k, attn.to_v)
        signature = tuple(
            (
                _tensor_signature(projection.weight),
                None
                if projection.bias is None
                else _tensor_signature(projection.bias),
            )
            for projection in projections
        )
        cached = self._packed_qkv.get(attn)
        if cached is not None and cached[0] == signature:
            return cached[1:]
        weight = torch.cat(
            [projection.weight for projection in projections]
        ).contiguous()
        biases = [projection.bias for projection in projections]
        if any(bias is None for bias in biases) != all(
            bias is None for bias in biases
        ):
            raise ValueError("Z-Image Q/K/V projections must agree on bias presence")
        bias = (
            None if biases[0] is None else torch.cat(biases).contiguous()
        )
        self._packed_qkv[attn] = (signature, weight, bias)
        return weight, bias

    @property
    def mode(self) -> str:
        return self.context_parallel.mode

    def enable(
        self,
        image_tokens: int,
        *,
        sharded_text: bool | None = None,
    ) -> None:
        self._image_tokens = int(image_tokens)
        self._enabled = self._image_tokens > 0
        self._active_sharded_text = (
            self.sharded_text if sharded_text is None else bool(sharded_text)
        )

    def disable(self) -> None:
        self._enabled = False
        self._image_tokens = 0
        self._active_sharded_text = False

    @staticmethod
    def _apply_rotary(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            x_out = torch.view_as_real(x * freqs_cis.unsqueeze(2)).flatten(3)
            return x_out.type_as(x_in)

    @staticmethod
    def _local_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mask = attention_mask
        if mask is not None and mask.ndim == 2:
            mask = mask[:, None, None, :].to(torch.bool)
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return output.transpose(1, 2).contiguous()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del encoder_hidden_states
        if self._enabled:
            from ...parallel.cp.fast.projection.qk_epilogue import (
                qk_norm_rope_inplace,
            )

            packed_weight, packed_bias = self._packed_qkv_for(attn)
            query, key, value = F.linear(
                hidden_states, packed_weight, packed_bias
            ).chunk(3, dim=-1)
            query = qk_norm_rope_inplace(
                query, attn.norm_q, freqs_cis, heads=attn.heads
            )
            key = qk_norm_rope_inplace(
                key, attn.norm_k, freqs_cis, heads=attn.heads
            )
            value = value.unflatten(-1, (attn.heads, -1))
        else:
            query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
            key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
            value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            if freqs_cis is not None:
                query = self._apply_rotary(query, freqs_cis)
                key = self._apply_rotary(key, freqs_cis)

        output_dtype = query.dtype
        query = query.to(value.dtype).contiguous()
        key = key.to(value.dtype).contiguous()
        value = value.contiguous()
        if self._enabled:
            if attention_mask is not None and not bool(
                attention_mask.to(torch.bool).all()
            ):
                raise NotImplementedError(
                    f"Z-Image {self.mode} CP requires an all-valid attention mask"
                )
            image_tokens = self._image_tokens
            joint_output, image_output = self.context_parallel(
                query[:, :image_tokens],
                key[:, :image_tokens],
                value[:, :image_tokens],
                query[:, image_tokens:],
                key[:, image_tokens:],
                value[:, image_tokens:],
                lane_process_group=self.parallel.active.process_group,
                ulysses_topology=(
                    self.parallel.active_ulysses if self.mode == "ulysses" else None
                ),
                agkv_transport=(
                    self.parallel.active_agkv_transport if self.mode == "agkv" else None
                ),
                joint_sharded=self._active_sharded_text,
            )
            output = torch.cat([image_output, joint_output], dim=1)
        else:
            output = self._local_attention(query, key, value, attention_mask)

        output = output.flatten(2, 3).to(output_dtype)
        output = attn.to_out[0](output)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output
