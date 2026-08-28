"""Sequence-parallel decoder layers that replace the released blocks in place.

The released backbone forward is left untouched. Instead, every decoder layer is
swapped for a wrapper that keeps the released parameter names, and the two
boundary layers own the sequence transition: the first shards the incoming
sequence across the context-parallel lane, the last gathers it again so the
ragged final layer, the language-model head, and the caller all still see the
full sequence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .attention import ParallelSelfAttention
from .moe import ParallelMoE, ParallelSwiGLU
from .parallel import HunyuanImage3ParallelRuntime
from .sequence import ContextParallelSequenceState


def _is_mixture_layer(config: Any, layer_idx: int) -> bool:
    experts = config.num_experts
    count = max(experts) if isinstance(experts, (list, tuple)) else experts
    return int(count) > 1 and layer_idx >= int(config.moe_layer_num_skipped)


class ParallelDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        config: Any,
        layer_idx: int,
        runtime: HunyuanImage3ParallelRuntime,
        sequence_state: ContextParallelSequenceState,
        apply_rotary_pos_emb: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        repeat_kv: Callable[[torch.Tensor, int], torch.Tensor],
        rms_norm_factory: Callable[[int, float], nn.Module],
        gate_factory: Callable[[Any, int], nn.Module],
        dtype: torch.dtype,
        shards_sequence: bool,
        gathers_sequence: bool,
    ) -> None:
        super().__init__()
        if config.norm_type not in ("rms", "hf_rms"):
            raise ValueError(
                f"Hunyuan Image 3 adaptation supports RMS norms, got {config.norm_type}"
            )
        self.layer_idx = int(layer_idx)
        self.sequence_state = sequence_state
        self.shards_sequence = bool(shards_sequence)
        self.gathers_sequence = bool(gathers_sequence)
        self.self_attn = ParallelSelfAttention(
            config=config,
            layer_idx=layer_idx,
            sequence_state=sequence_state,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
            repeat_kv=repeat_kv,
            rms_norm_factory=rms_norm_factory,
            dtype=dtype,
        )
        if _is_mixture_layer(config, layer_idx):
            self.mlp: nn.Module = ParallelMoE(
                config=config,
                layer_idx=layer_idx,
                runtime=runtime,
                gate_factory=gate_factory,
                dtype=dtype,
            )
        else:
            self.mlp = ParallelSwiGLU(
                hidden_size=int(config.hidden_size),
                intermediate_size=int(config.intermediate_size),
                bias=bool(config.mlp_bias),
                dtype=dtype,
                reduce_results=True,
            )
        self.input_layernorm = rms_norm_factory(
            int(config.hidden_size), float(config.rms_norm_eps)
        )
        self.post_attention_layernorm = rms_norm_factory(
            int(config.hidden_size), float(config.rms_norm_eps)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        custom_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        del kwargs
        if self.shards_sequence:
            self.sequence_state.begin(hidden_states.shape[1], attention_mask)
            hidden_states = self.sequence_state.shard_hidden(hidden_states)

        residual = hidden_states
        attention, attention_weights, present = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=bool(output_attentions),
            use_cache=use_cache,
            custom_pos_emb=custom_pos_emb,
        )
        hidden_states = residual + attention
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )

        if self.gathers_sequence:
            hidden_states = self.sequence_state.gather_hidden(hidden_states)
            self.sequence_state.end()

        outputs: tuple[Any, ...] = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (present,)
        return outputs


__all__ = ["ParallelDecoderLayer"]
