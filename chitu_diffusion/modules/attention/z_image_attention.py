from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention


class ZImageChituAttnProcessor:
    """Z-Image single-stream attention processor backed by Chitu attention."""

    def __init__(self, attn_backend):
        self.attn_backend = attn_backend

    @staticmethod
    def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    @staticmethod
    def _mask_allows_fast_path(attention_mask: Optional[torch.Tensor]) -> bool:
        if attention_mask is None:
            return True
        # Z-Image uses ragged prompt/image sequences internally. Flash attention is
        # safe only when the packed sequence is already fully valid; otherwise keep
        # the explicit mask and fall back to SDPA for correctness.
        return bool(attention_mask.to(torch.bool).all().item())

    @staticmethod
    def _sdpa_with_mask(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attn_mask = attention_mask[:, None, None, :].to(torch.bool)
            elif attention_mask.ndim == 4:
                attn_mask = attention_mask.to(torch.bool)
            else:
                raise ValueError(f"Unsupported Z-Image attention mask shape: {tuple(attention_mask.shape)}")
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return out.transpose(1, 2).contiguous()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if freqs_cis is not None:
            query = self._apply_rotary_emb(query, freqs_cis)
            key = self._apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query = query.to(value.dtype).contiguous()
        key = key.to(value.dtype).contiguous()
        value = value.contiguous()

        if self._mask_allows_fast_path(attention_mask):
            hidden_states = self.attn_backend(query, key, value, causal=False)[0]
        else:
            hidden_states = self._sdpa_with_mask(query, key, value, attention_mask)

        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output
