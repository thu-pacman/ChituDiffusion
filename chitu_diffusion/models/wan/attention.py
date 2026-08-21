from __future__ import annotations

import torch
from diffusers.models.transformers.transformer_wan import WanAttnProcessor

from ...parallel import (
    AsyncAgkvTransport,
    AsyncTaggedUlyssesTransport,
    EpeParallelContext,
    ImageSelfAttention,
)


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

    def _can_use_fast_async(self, attn) -> bool:
        if self.attention.mode != "usp" or getattr(attn, "fused_projections", False):
            return False
        topology = self.parallel.active_usp_for_heads(attn.heads)
        transport = topology.ulysses_transport
        return (
            topology.ring_degree == 1
            and attn.heads % topology.ulysses_degree == 0
            and isinstance(transport, AsyncTaggedUlyssesTransport)
            and transport.async_ce_enabled
        )

    def _fast_async_attention(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        topology = self.parallel.active_usp_for_heads(attn.heads)
        transport = topology.ulysses_transport
        if not isinstance(transport, AsyncTaggedUlyssesTransport):
            raise TypeError("async tagged Ulysses transport is required")
        transport.reset()

        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(
            2, (attn.heads, -1)
        )
        if rotary_emb is not None:
            query = _apply_rotary(query, *rotary_emb)
        query_handle = transport.begin_all_to_all(
            query,
            2,
            1,
            tag="chitu_cp_0",
            barrier=False,
        )

        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            key = _apply_rotary(key, *rotary_emb)
        key_handle = transport.begin_all_to_all(
            key,
            2,
            1,
            tag="chitu_cp_1",
            barrier=True,
        )

        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        # The K handle carries the grouped handshake and publishes Q/K while
        # the caller stream computes V. V has no following projection to hide
        # its transfer, so use the faster synchronous kernel path for it.
        key = key_handle.wait()
        query = query_handle.wait()
        value = transport.all_to_all_tagged(
            value,
            2,
            1,
            tag="chitu_cp_2",
        )
        return self.attention.finish_usp(
            query,
            key,
            value,
            topology=topology,
            output_tag="chitu_cp_3",
        )

    def _can_use_fast_agkv_async(self, attn) -> bool:
        if self.attention.mode != "agkv" or getattr(
            attn, "fused_projections", False
        ):
            return False
        topology = getattr(self.parallel, "active_usp", None)
        transport = getattr(topology, "agkv_transport", None)
        return (
            isinstance(transport, AsyncAgkvTransport)
            and transport.async_enabled
        )

    def _fast_agkv_attention(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.dtype]:
        transport = getattr(
                    getattr(self.parallel, "active_usp", None),
                    "agkv_transport",
                    None,
                )
        if not isinstance(transport, AsyncAgkvTransport):
            raise TypeError("asynchronous AGKV transport is required")

        # Compute K/V first, then overlap their all-gather with the independent
        # Q projection, normalization, and RoPE on the caller stream.
        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            key = _apply_rotary(key, *rotary_emb)
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        gather = transport.begin_all_gather_kv(key, value)

        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(
            2, (attn.heads, -1)
        )
        if rotary_emb is not None:
            query = _apply_rotary(query, *rotary_emb)
        full_key, full_value = gather.wait()
        return (
            self.attention.finish_agkv(query, full_key, full_value),
            query.dtype,
        )

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

        use_fast_ulysses = self._can_use_fast_async(attn)
        use_fast_agkv = self._can_use_fast_agkv_async(attn)
        if use_fast_ulysses:
            output = self._fast_async_attention(attn, hidden_states, rotary_emb)
            query_dtype = output.dtype
        elif use_fast_agkv:
            output, query_dtype = self._fast_agkv_attention(
                attn,
                hidden_states,
                rotary_emb,
            )
        elif getattr(attn, "fused_projections", False):
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        if not use_fast_ulysses and not use_fast_agkv:
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
                    self.parallel.active_usp_for_heads(attn.heads)
                    if self.attention.mode == "usp"
                    else None
                ),
                agkv_transport=(
                    getattr(
                        getattr(self.parallel, "active_usp", None),
                        "agkv_transport",
                        None,
                    )
                    if self.attention.mode == "agkv"
                    else None
                ),
            )
            query_dtype = query.dtype
        output = output.flatten(2, 3).to(query_dtype)
        output = attn.to_out[0](output.contiguous())
        return attn.to_out[1](output)
