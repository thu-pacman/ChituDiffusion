"""Tensor-parallel, variable-length AGKV attention for Hunyuan Image 3."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from ...parallel.cp import (
    heads_to_sequence,
    prefix_masked_attention,
    sequence_to_heads,
    shard_query_mask,
)
from ...parallel.tp import PlainColumnParallelLinear, PlainRowParallelLinear
from .sequence import ContextParallelSequenceState


class ParallelSelfAttention(nn.Module):
    """Replacement for the released SDPA attention module.

    Projections are tensor-parallel and queries remain sequence-sharded. Only
    K/V are gathered across the CP lane, which is especially useful for the
    released grouped-query attention shape. Rotary embeddings are applied before
    the gather on each rank's contiguous shard. The gathered K/V update a
    lane-replicated cache, while the attention output remains sequence-sharded
    and therefore needs no inverse exchange.
    """

    def __init__(
        self,
        *,
        config: Any,
        layer_idx: int,
        sequence_state: ContextParallelSequenceState,
        apply_rotary_pos_emb: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        repeat_kv: Callable[[torch.Tensor, int], torch.Tensor],
        rms_norm_factory: Callable[[int, float], nn.Module],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.attention_type = "self"
        self.sequence_state = sequence_state
        self._apply_rotary_pos_emb = apply_rotary_pos_emb
        self._repeat_kv = repeat_kv

        tp_degree = sequence_state.runtime.plan.tensor_parallel_degree
        self.head_dim = int(config.attention_head_dim)
        self.num_key_value_groups = int(config.num_attention_heads) // int(
            config.num_key_value_heads
        )
        self.local_num_heads = int(config.num_attention_heads) // tp_degree
        self.local_num_key_value_heads = int(config.num_key_value_heads) // tp_degree
        self.use_qk_norm = bool(config.use_qk_norm)
        self.use_rotary_pos_emb = bool(config.use_rotary_pos_emb)

        query_width = self.local_num_heads * self.head_dim
        key_value_width = self.local_num_key_value_heads * self.head_dim
        # The released checkpoint interleaves the fused projection by key/value
        # head, so a contiguous column shard keeps whole query/key/value groups
        # together and needs no reordering.
        self.qkv_proj = PlainColumnParallelLinear(
            int(config.hidden_size),
            (query_width + 2 * key_value_width) * tp_degree,
            bias=bool(config.attention_bias),
            params_dtype=dtype,
            device="meta",
        )
        self.o_proj = PlainRowParallelLinear(
            int(config.num_attention_heads) * self.head_dim,
            int(config.hidden_size),
            bias=bool(config.attention_bias),
            input_is_parallel=True,
            reduce_results=True,
            params_dtype=dtype,
            device="meta",
        )
        if self.use_qk_norm:
            self.query_layernorm = rms_norm_factory(
                self.head_dim, float(config.rms_norm_eps)
            )
            self.key_layernorm = rms_norm_factory(
                self.head_dim, float(config.rms_norm_eps)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool | None = False,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, Any]:
        del use_cache, kwargs
        if output_attentions:
            raise NotImplementedError(
                "Hunyuan Image 3 parallel attention does not return attention weights"
            )
        partition = self.sequence_state.require_partition()
        batch, local_length, _ = hidden_states.shape

        fused = self.qkv_proj(hidden_states).reshape(
            batch,
            local_length,
            self.local_num_key_value_heads,
            self.num_key_value_groups + 2,
            self.head_dim,
        )
        query, key, value = torch.split(
            fused, [self.num_key_value_groups, 1, 1], dim=3
        )
        query = query.reshape(
            batch, local_length, self.local_num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.reshape(
            batch, local_length, self.local_num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            batch, local_length, self.local_num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if self.use_rotary_pos_emb:
            if custom_pos_emb is None:
                raise ValueError("Hunyuan Image 3 requires precomputed rotary tables")
            cosine, sine = self.sequence_state.shard_rotary(custom_pos_emb)
            query, key = self._apply_rotary_pos_emb(query, key, cosine, sine)
        if self.use_qk_norm:
            query = self.query_layernorm(query)
            key = self.key_layernorm(key)
        query = query.to(value.dtype)
        key = key.to(value.dtype)

        group = self.sequence_state.process_group
        use_agkv = self.sequence_state.runtime.plan.attention_mode == "agkv"
        ulysses_transport = (
            None
            if use_agkv
            else self.sequence_state.runtime.context.active_ulysses.transport
        )
        if use_agkv:
            key, value = self.sequence_state.gather_kv(key, value)
        else:
            query = heads_to_sequence(
                query,
                partition=partition,
                process_group=group,
                transport=ulysses_transport,
            )
            key = heads_to_sequence(
                key,
                partition=partition,
                process_group=group,
                transport=ulysses_transport,
            )
            value = heads_to_sequence(
                value,
                partition=partition,
                process_group=group,
                transport=ulysses_transport,
            )

        if past_key_value is not None:
            key, value = past_key_value.update(
                key, value, self.layer_idx, {"cache_position": position_ids}
            )
            query = query.to(key.dtype)

        key = self._repeat_kv(key, self.num_key_value_groups)
        value = self._repeat_kv(value, self.num_key_value_groups)
        plan = self.sequence_state.mask_plan
        local_mask = attention_mask
        if use_agkv:
            local_mask = shard_query_mask(
                attention_mask,
                offset=partition.offset,
                length=partition.local_length,
                total_length=partition.total_length,
            )
            if plan is not None:
                plan = plan.shard_queries(
                    offset=partition.offset, length=partition.local_length
                )
        attention = prefix_masked_attention(
            query,
            key,
            value,
            mask=local_mask,
            plan=plan,
        )

        if not use_agkv:
            attention = sequence_to_heads(
                attention,
                partition=partition,
                process_group=group,
                transport=ulysses_transport,
            )
        attention = attention.transpose(1, 2).reshape(batch, local_length, -1)
        return self.o_proj(attention), None, past_key_value


__all__ = ["ParallelSelfAttention"]
