"""Per-step sequence sharding shared by every wrapped decoder layer."""

from __future__ import annotations

import torch

from ...parallel.cp import (
    PrefixMaskPlan,
    SequencePartition,
    gather_uneven_kv,
    plan_prefix_mask,
)
from .parallel import HunyuanImage3ParallelRuntime


class ContextParallelSequenceState:
    """The sequence partition in force for the step currently being computed.

    A denoising step changes length: the first step carries the whole prompt
    while later steps carry only the timestep and image tokens. The boundary
    decoder layers publish the partition here so that attention layers, rotary
    tables, and the final gather all agree without threading an extra argument
    through the released forward signatures. The attention mask is read here too,
    once per step rather than once per layer.
    """

    def __init__(self, runtime: HunyuanImage3ParallelRuntime) -> None:
        self.runtime = runtime
        self._partition: SequencePartition | None = None
        self._mask_plan: PrefixMaskPlan | None = None

    @property
    def process_group(self) -> object | None:
        return self.runtime.context_parallel_group

    @property
    def degree(self) -> int:
        return self.runtime.context_parallel_degree

    def begin(
        self, total_length: int, attention_mask: torch.Tensor | None = None
    ) -> SequencePartition:
        self._partition = self.runtime.sequence_partition(total_length)
        self._mask_plan = plan_prefix_mask(attention_mask)
        return self._partition

    def end(self) -> None:
        self._partition = None
        self._mask_plan = None

    @property
    def mask_plan(self) -> PrefixMaskPlan | None:
        return self._mask_plan

    def require_partition(self) -> SequencePartition:
        if self._partition is None:
            raise RuntimeError(
                "no sequence partition is active; the first decoder layer must "
                "shard the sequence before attention runs"
            )
        return self._partition

    def shard_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        partition = self.require_partition()
        if hidden_states.shape[1] == partition.local_length:
            return hidden_states
        return partition.shard(hidden_states, 1).contiguous()

    def gather_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        partition = self.require_partition()
        return partition.gather(hidden_states, 1, process_group=self.process_group)

    def gather_kv(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather uneven ``[B, H, S, D]`` K/V through the AGKV transport."""

        partition = self.require_partition()
        return gather_uneven_kv(
            key,
            value,
            partition=partition,
            transport=self.runtime.context.active_agkv_transport,
        )

    def shard_rotary(
        self, rotary: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        partition = self.require_partition()
        cosine, sine = rotary
        if cosine.shape[1] != partition.total_length:
            return cosine, sine
        return partition.shard(cosine, 1), partition.shard(sine, 1)


__all__ = ["ContextParallelSequenceState"]
