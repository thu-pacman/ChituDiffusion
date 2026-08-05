from __future__ import annotations

import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import (
    Flux2AttnProcessor,
    Flux2ParallelSelfAttnProcessor,
)

from ...parallel import EpeParallelContext, ImageContextParallelAttention


class Flux2KleinCpAttnProcessor:
    """Flux2 double-stream attention with sequence-sharded image tokens."""

    def __init__(self, parallel: EpeParallelContext, *, mode: str = "agkv") -> None:
        self.parallel = parallel
        self.attention = ImageContextParallelAttention(mode=mode)
        self.native = Flux2AttnProcessor()
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
            raise NotImplementedError(
                "FLUX.2-klein CP does not support attention masks"
            )
        if self._text_tokens is None:
            return self.native(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
            )
        if encoder_hidden_states is None:
            raise ValueError("double-stream attention requires text hidden states")
        if getattr(attn, "fused_projections", False):
            raise NotImplementedError(
                "FLUX.2-klein CP does not support fused double-stream QKV"
            )

        image_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        image_key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        image_value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        image_query = attn.norm_q(image_query)
        image_key = attn.norm_k(image_key)

        text_query = attn.add_q_proj(encoder_hidden_states).unflatten(
            -1, (attn.heads, -1)
        )
        text_key = attn.add_k_proj(encoder_hidden_states).unflatten(
            -1, (attn.heads, -1)
        )
        text_value = attn.add_v_proj(encoder_hidden_states).unflatten(
            -1, (attn.heads, -1)
        )
        text_query = attn.norm_added_q(text_query)
        text_key = attn.norm_added_k(text_key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(
                torch.cat([text_query, image_query], dim=1),
                image_rotary_emb,
                sequence_dim=1,
            )
            key = apply_rotary_emb(
                torch.cat([text_key, image_key], dim=1),
                image_rotary_emb,
                sequence_dim=1,
            )
            text_query, image_query = query.split(
                [text_query.shape[1], image_query.shape[1]], dim=1
            )
            text_key, image_key = key.split(
                [text_key.shape[1], image_key.shape[1]], dim=1
            )

        text_output, image_output = self.attention(
            image_query,
            image_key,
            image_value,
            text_query,
            text_key,
            text_value,
            lane_process_group=self.parallel.active.process_group,
            usp_topology=(
                self.parallel.active_usp if self.attention.mode == "usp" else None
            ),
            joint_first=True,
        )
        image_output = image_output.flatten(2, 3).to(image_query.dtype)
        text_output = text_output.flatten(2, 3).to(text_query.dtype)
        image_output = attn.to_out[1](attn.to_out[0](image_output.contiguous()))
        text_output = attn.to_add_out(text_output.contiguous())
        return image_output, text_output


class Flux2KleinCpSingleAttnProcessor:
    """Flux2 fused attention/MLP processor for replicated text and local image."""

    def __init__(self, parallel: EpeParallelContext, *, mode: str = "agkv") -> None:
        self.parallel = parallel
        self.attention = ImageContextParallelAttention(mode=mode)
        self.native = Flux2ParallelSelfAttnProcessor()
        self._text_tokens: int | None = None

    def enable(self, text_tokens: int) -> None:
        self._text_tokens = int(text_tokens)

    def disable(self) -> None:
        self._text_tokens = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError(
                "FLUX.2-klein CP does not support attention masks"
            )
        text_tokens = self._text_tokens
        if text_tokens is None:
            return self.native(attn, hidden_states, attention_mask, image_rotary_emb)

        projected = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            projected,
            [
                3 * attn.inner_dim,
                attn.mlp_hidden_dim * attn.mlp_mult_factor,
            ],
            dim=-1,
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        text_query, image_query = query.split(
            [text_tokens, query.shape[1] - text_tokens], dim=1
        )
        text_key, image_key = key.split(
            [text_tokens, key.shape[1] - text_tokens], dim=1
        )
        text_value, image_value = value.split(
            [text_tokens, value.shape[1] - text_tokens], dim=1
        )
        text_output, image_output = self.attention(
            image_query,
            image_key,
            image_value,
            text_query,
            text_key,
            text_value,
            lane_process_group=self.parallel.active.process_group,
            usp_topology=(
                self.parallel.active_usp if self.attention.mode == "usp" else None
            ),
            joint_first=True,
        )
        attention_output = torch.cat([text_output, image_output], dim=1)
        attention_output = attention_output.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        return attn.to_out(torch.cat([attention_output, mlp_hidden_states], dim=-1))
