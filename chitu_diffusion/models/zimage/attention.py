from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from ...parallel import EpeParallelContext, ImageContextParallelAttention


class ZImageCpAttnProcessor:
    """Diffusers projection/RoPE glue around shared CP attention primitives."""

    def __init__(self, parallel: EpeParallelContext, *, mode: str = "agkv") -> None:
        self.parallel = parallel
        self.context_parallel = ImageContextParallelAttention(mode=mode)
        self._image_tokens = 0
        self._enabled = False

    @property
    def mode(self) -> str:
        return self.context_parallel.mode

    def enable(self, image_tokens: int) -> None:
        self._image_tokens = int(image_tokens)
        self._enabled = self.parallel.active.width > 1

    def disable(self) -> None:
        self._enabled = False
        self._image_tokens = 0

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
                usp_topology=(self.parallel.active_usp if self.mode == "usp" else None),
            )
            output = torch.cat([image_output, joint_output], dim=1)
        else:
            output = self._local_attention(query, key, value, attention_mask)

        output = output.flatten(2, 3).to(output_dtype)
        output = attn.to_out[0](output)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


# Compatibility name retained for callers that explicitly imported the old processor.
ZImageAgkvAttnProcessor = ZImageCpAttnProcessor
