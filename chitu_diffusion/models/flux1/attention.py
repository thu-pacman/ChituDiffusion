from __future__ import annotations

import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

from ...parallel.cp import EpeParallelContext, ImageContextParallelAttention


class Flux1CpAttnProcessor:
    """Flux joint attention for local image tokens and replicated text tokens."""

    def __init__(self, parallel: EpeParallelContext, *, mode: str = "agkv") -> None:
        self.parallel = parallel
        self.attention = ImageContextParallelAttention(mode=mode)
        self.native = FluxAttnProcessor()
        self._text_tokens: int | None = None

    def enable(self, text_tokens: int) -> None:
        self._text_tokens = int(text_tokens)

    def disable(self) -> None:
        self._text_tokens = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ):
        if attention_mask is not None:
            raise NotImplementedError("Flux.1 EPE does not support attention masks")
        if getattr(attn, "fused_projections", False):
            raise NotImplementedError(
                "Flux.1 EPE does not support fused QKV projections"
            )
        text_tokens = self._text_tokens
        if text_tokens is None:
            return self.native(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            image_query, image_key, image_value = query, key, value
            joint_query = attn.add_q_proj(encoder_hidden_states).unflatten(
                -1, (attn.heads, -1)
            )
            joint_key = attn.add_k_proj(encoder_hidden_states).unflatten(
                -1, (attn.heads, -1)
            )
            joint_value = attn.add_v_proj(encoder_hidden_states).unflatten(
                -1, (attn.heads, -1)
            )
            joint_query = attn.norm_added_q(joint_query)
            joint_key = attn.norm_added_k(joint_key)
        else:
            joint_query, image_query = query[:, :text_tokens], query[:, text_tokens:]
            joint_key, image_key = key[:, :text_tokens], key[:, text_tokens:]
            joint_value, image_value = value[:, :text_tokens], value[:, text_tokens:]

        if image_rotary_emb is not None:
            combined_query = torch.cat([joint_query, image_query], dim=1)
            combined_key = torch.cat([joint_key, image_key], dim=1)
            combined_query = apply_rotary_emb(
                combined_query, image_rotary_emb, sequence_dim=1
            )
            combined_key = apply_rotary_emb(
                combined_key, image_rotary_emb, sequence_dim=1
            )
            joint_query, image_query = combined_query.split(
                [joint_query.shape[1], image_query.shape[1]], dim=1
            )
            joint_key, image_key = combined_key.split(
                [joint_key.shape[1], image_key.shape[1]], dim=1
            )

        joint_output, image_output = self.attention(
            image_query,
            image_key,
            image_value,
            joint_query,
            joint_key,
            joint_value,
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
        )
        image_output = image_output.flatten(2, 3).to(query.dtype)
        joint_output = joint_output.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is None:
            return torch.cat([joint_output, image_output], dim=1)

        image_output = attn.to_out[0](image_output.contiguous())
        image_output = attn.to_out[1](image_output)
        joint_output = attn.to_add_out(joint_output.contiguous())
        return image_output, joint_output
