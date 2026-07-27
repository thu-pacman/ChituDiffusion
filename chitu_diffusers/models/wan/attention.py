from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.transformers.transformer_wan import WanAttnProcessor

from ...parallel import EpeParallelContext


def _all_gather_sequence(
    tensor: torch.Tensor, process_group: object | None
) -> torch.Tensor:
    width = dist.get_world_size(process_group) if process_group is not None else 1
    if width == 1:
        return tensor
    pieces = [torch.empty_like(tensor) for _ in range(width)]
    dist.all_gather(pieces, tensor.contiguous(), group=process_group)
    return torch.cat(pieces, dim=1).contiguous()


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
        if mode != "agkv":
            raise NotImplementedError("Wan EPAC currently supports AGKV only")
        self.parallel = parallel
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

        full_key = _all_gather_sequence(key, topology.process_group)
        full_value = _all_gather_sequence(value, topology.process_group)
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            full_key.transpose(1, 2),
            full_value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output.contiguous())
        return attn.to_out[1](output)
