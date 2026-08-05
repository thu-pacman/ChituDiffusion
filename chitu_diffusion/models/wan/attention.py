from __future__ import annotations

import torch
from diffusers.models.transformers.transformer_wan import WanAttnProcessor

from ...parallel import EpeParallelContext, ImageSelfAttention


def _apply_rotary(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = x1 * cos - x2 * sin
    output[..., 1::2] = x1 * sin + x2 * cos
    return output.type_as(hidden_states)


class WanCpAttnProcessor:
    """Wan self-attention with local queries and lane-gathered K/V."""

    def __init__(self, parallel: EpeParallelContext, *, mode: str = "agkv") -> None:
        self.parallel = parallel
        self.attention = ImageSelfAttention(mode=mode)
        self.native = WanAttnProcessor()

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        topology = self.parallel.active
        if topology.width == 1 or encoder_hidden_states is not None:
            return self.native(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
            )
        if attention_mask is not None:
            raise NotImplementedError("Wan EPAC self-attention masks are unsupported")

        if getattr(attn, "fused_projections", False):
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary(query, *rotary_emb)
            key = _apply_rotary(key, *rotary_emb)

        output = self.attention(
            query,
            key,
            value,
            lane_process_group=topology.process_group,
            usp_topology=(
                self.parallel.active_usp if self.attention.mode == "usp" else None
            ),
        )
        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output.contiguous())
        return attn.to_out[1](output)
