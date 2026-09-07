"""Tensor- and expert-parallel feed-forward blocks for Hunyuan Image 3."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ...parallel.ep import expert_parallel_moe
from ...parallel.tp import (
    PlainMergedColumnParallelLinear,
    PlainRowParallelLinear,
    get_tp_group,
    get_tp_world_size,
)
from .parallel import HunyuanImage3ParallelRuntime


class ParallelSwiGLU(nn.Module):
    """A released SwiGLU block with both projections sharded over tensor ranks.

    ``reduce_results`` is left to the caller so a mixture-of-experts block can
    keep every partial sum local and pay for a single reduction after the shared
    and routed contributions have been added together.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        bias: bool,
        dtype: torch.dtype,
        reduce_results: bool,
    ) -> None:
        super().__init__()
        self.gate_and_up_proj = PlainMergedColumnParallelLinear(
            hidden_size,
            (intermediate_size, intermediate_size),
            bias=bias,
            params_dtype=dtype,
            device="meta",
        )
        self.down_proj = PlainRowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            input_is_parallel=True,
            reduce_results=reduce_results,
            params_dtype=dtype,
            device="meta",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        up, gate = self.gate_and_up_proj(inputs).chunk(2, dim=-1)
        return self.down_proj(up * F.silu(gate))


def swiglu_intermediate_size(config: Any, layer_idx: int, *, shared: bool) -> int:
    """Return the per-half SwiGLU width the released config implies."""

    size = config.moe_intermediate_size
    if isinstance(size, (list, tuple)):
        size = size[layer_idx]
    if shared:
        shared_experts = config.num_shared_expert
        if isinstance(shared_experts, (list, tuple)):
            shared_experts = shared_experts[layer_idx]
        size *= int(shared_experts)
    return int(size)


class ParallelMoE(nn.Module):
    """Replacement for the released mixture-of-experts block.

    Routed experts are owned by contiguous slices of the expert-parallel group,
    so each rank materializes only its own experts and exchanges the tokens that
    belong to them. Parameter names match the release, which keeps the checkpoint
    contract intact.
    """

    def __init__(
        self,
        *,
        config: Any,
        layer_idx: int,
        runtime: HunyuanImage3ParallelRuntime,
        gate_factory: Callable[[Any, int], nn.Module],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.use_shared_mlp = bool(config.use_mixed_mlp_moe)
        hidden_size = int(config.hidden_size)
        bias = bool(config.mlp_bias)
        if self.use_shared_mlp:
            self.shared_mlp = ParallelSwiGLU(
                hidden_size=hidden_size,
                intermediate_size=swiglu_intermediate_size(
                    config, layer_idx, shared=True
                ),
                bias=bias,
                dtype=dtype,
                reduce_results=False,
            )
        self.gate = gate_factory(config, layer_idx)
        routed_size = swiglu_intermediate_size(config, layer_idx, shared=False)
        topology = runtime.experts
        # A ModuleDict keyed by the global expert index produces exactly the
        # parameter names a ModuleList would, so only the owned slice is built.
        self.experts = nn.ModuleDict(
            {
                str(index): ParallelSwiGLU(
                    hidden_size=hidden_size,
                    intermediate_size=routed_size,
                    bias=bias,
                    dtype=dtype,
                    reduce_results=False,
                )
                for index in range(
                    topology.local_expert_start, topology.local_expert_end
                )
            }
        )

    def _run_expert(self, expert_index: int, rows: torch.Tensor) -> torch.Tensor:
        return self.experts[str(expert_index)](rows)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden_size = hidden_states.shape
        weights, indices = self.gate(hidden_states, topk_impl="easy")
        routed = expert_parallel_moe(
            hidden_states.reshape(-1, hidden_size),
            weights,
            indices,
            topology=self.runtime.experts,
            run_expert=self._run_expert,
        ).reshape(batch, sequence, hidden_size)
        if self.use_shared_mlp:
            routed = routed + self.shared_mlp(hidden_states)
        if get_tp_world_size() > 1:
            dist.all_reduce(routed, group=get_tp_group())
        return routed


__all__ = ["ParallelMoE", "ParallelSwiGLU", "swiglu_intermediate_size"]
