"""Expert-parallel ownership, independent of tensor and context parallelism."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ExpertParallelTopology:
    """Contiguous routed-expert ownership over one expert-parallel group."""

    ranks: tuple[int, ...]
    rank: int
    rank_in_group: int
    degree: int
    num_experts: int
    process_group: object | None

    def __post_init__(self) -> None:
        if not self.ranks:
            raise ValueError("expert-parallel group must contain at least one rank")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("expert-parallel ranks must be unique")
        if self.degree != len(self.ranks):
            raise ValueError("degree must match the number of group ranks")
        if not 0 <= self.rank_in_group < self.degree:
            raise ValueError("rank_in_group must be within the group")
        if self.ranks[self.rank_in_group] != self.rank:
            raise ValueError("rank_in_group must index this rank inside the group")
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive")
        if self.num_experts % self.degree:
            raise ValueError(
                f"{self.num_experts} experts are not divisible by "
                f"expert-parallel degree {self.degree}"
            )
        if self.degree > 1 and self.process_group is None:
            raise ValueError("multi-rank expert parallelism requires a process group")

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.degree

    @property
    def local_expert_start(self) -> int:
        return self.rank_in_group * self.experts_per_rank

    @property
    def local_expert_end(self) -> int:
        return self.local_expert_start + self.experts_per_rank

    def owns(self, expert_index: int) -> bool:
        return self.local_expert_start <= expert_index < self.local_expert_end

    def owner_of(self, expert_index: int) -> int:
        if not 0 <= expert_index < self.num_experts:
            raise ValueError(f"expert {expert_index} is outside the routed range")
        return expert_index // self.experts_per_rank

    def owner_of_tensor(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """Map routed expert ids to their owning rank inside the group."""

        return torch.div(expert_indices, self.experts_per_rank, rounding_mode="floor")


def expert_parallel_ranks(
    group_ranks: tuple[int, ...],
    *,
    rank: int,
    degree: int,
) -> tuple[int, ...]:
    """Return the contiguous slice of ``group_ranks`` that holds one replica.

    A degree equal to the group width shards the routing table once across the
    whole group. A smaller degree splits the group into several slices, each
    holding its own copy of every expert, which trades memory for a narrower
    dispatch collective.
    """

    ranks = tuple(int(value) for value in group_ranks)
    if degree < 1 or len(ranks) % degree:
        raise ValueError(
            f"expert-parallel degree {degree} must divide the group width "
            f"{len(ranks)}"
        )
    if int(rank) not in ranks:
        raise ValueError("expert-parallel ranks must contain the current rank")
    start = (ranks.index(int(rank)) // degree) * degree
    return ranks[start : start + degree]


def build_expert_parallel_topology(
    *,
    num_experts: int,
    ranks: tuple[int, ...],
    rank: int,
    process_group: object | None,
) -> ExpertParallelTopology:
    """Bind routed-expert ownership to a process group owned by the caller.

    The group is never created or destroyed here. Callers reuse an existing
    group, such as a context-parallel lane, without coupling the expert axis to
    that lane's semantics.
    """

    normalized = tuple(int(value) for value in ranks)
    global_rank = int(rank)
    if global_rank not in normalized:
        raise ValueError("expert-parallel ranks must contain the current rank")
    return ExpertParallelTopology(
        ranks=normalized,
        rank=global_rank,
        rank_in_group=normalized.index(global_rank),
        degree=len(normalized),
        num_experts=int(num_experts),
        process_group=process_group,
    )
