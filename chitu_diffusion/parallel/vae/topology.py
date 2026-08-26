"""Minimal topology contract required by VAE-parallel operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch.distributed as dist


@runtime_checkable
class VaeParallelTopology(Protocol):
    """Structural view of an active decode lane."""

    rank_in_lane: int
    width: int
    process_group: object | None

    @property
    def is_leader(self) -> bool: ...


@dataclass
class VaeParallelGroup:
    """Independent VAE decode group, orthogonal to DiT TP/CP axes."""

    ranks: tuple[int, ...]
    global_rank: int
    process_group: object | None
    owns_group: bool = False

    @property
    def width(self) -> int:
        return len(self.ranks)

    @property
    def rank_in_lane(self) -> int:
        return self.ranks.index(self.global_rank) if self.is_member else -1

    @property
    def is_member(self) -> bool:
        return self.global_rank in self.ranks

    @property
    def is_leader(self) -> bool:
        return self.global_rank == self.ranks[0]

    def close(self) -> None:
        if self.owns_group and self.is_member and dist.is_initialized():
            assert self.process_group is not None
            dist.destroy_process_group(self.process_group)
            self.owns_group = False


def create_vae_parallel_group(
    degree: int,
    *,
    world_size: int | None = None,
    rank: int | None = None,
) -> VaeParallelGroup:
    """Create one decode group on the first ``degree`` stage ranks."""

    if world_size is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    if degree < 1 or degree > world_size:
        raise ValueError("vae_parallel_degree must be in [1, world_size]")
    ranks = tuple(range(degree))
    if degree == 1:
        process_group = None
        owns_group = False
    elif degree == world_size:
        if not dist.is_initialized():
            raise RuntimeError("distributed must be initialized for multi-rank VAEP")
        process_group = dist.group.WORLD
        owns_group = False
    else:
        if not dist.is_initialized():
            raise RuntimeError("distributed must be initialized for multi-rank VAEP")
        created = dist.new_group(ranks=list(ranks))
        process_group = created if rank in ranks else None
        owns_group = rank in ranks
    return VaeParallelGroup(
        ranks=ranks,
        global_rank=rank,
        process_group=process_group,
        owns_group=owns_group,
    )


__all__ = [
    "VaeParallelGroup",
    "VaeParallelTopology",
    "create_vae_parallel_group",
]
