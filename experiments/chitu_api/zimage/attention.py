from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from .parallel import EpeParallelContext


class ZImageAgkvAttnProcessor:
    """Z-Image attention with local Q and all-gathered image K/V."""

    def __init__(self, parallel: EpeParallelContext) -> None:
        self.parallel = parallel
        self._image_tokens = 0
        self._enabled = False

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
    def _attention(
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

    def _all_gather_image(self, tensor: torch.Tensor) -> torch.Tensor:
        topology = self.parallel.active
        pieces = [torch.empty_like(tensor) for _ in range(topology.width)]
        dist.all_gather(pieces, tensor.contiguous(), group=topology.process_group)
        return torch.cat(pieces, dim=1).contiguous()

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

        if self._enabled:
            if attention_mask is not None and not bool(attention_mask.to(torch.bool).all()):
                raise NotImplementedError("AGKV CP currently requires an all-valid attention mask")
            image_tokens = self._image_tokens
            full_image_key = self._all_gather_image(key[:, :image_tokens])
            full_image_value = self._all_gather_image(value[:, :image_tokens])
            full_key = torch.cat([full_image_key, key[:, image_tokens:]], dim=1)
            full_value = torch.cat([full_image_value, value[:, image_tokens:]], dim=1)
            hidden_states = self._attention(query, full_key, full_value, None)
        else:
            hidden_states = self._attention(query, key, value, attention_mask)

        hidden_states = hidden_states.flatten(2, 3).to(output_dtype)
        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
