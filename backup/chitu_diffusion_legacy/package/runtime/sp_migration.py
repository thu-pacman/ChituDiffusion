# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Latent re-shard (migration) primitives for dynamic sequence-parallel (SP/CP)
degree switching (M6).

A logical sequence of length ``L`` is sharded contiguously across ``d`` ranks
following the same convention the Z-Image ``cp_forward`` uses::

    chunk = L // d
    rank r owns global positions [r*chunk, (r+1)*chunk)        (L % d == 0)

When ``L`` is not divisible by ``d`` we fall back to the ``torch.tensor_split``
convention (the first ``L % d`` shards are one element larger) so the index math
stays well-defined for the CPU unit tests; the Z-Image GPU path always uses the
divisible case (it skips CP for a step whose ``img_len`` is not divisible).

Two migration algorithms are provided:

* **gather + scatter** (correctness-first): all-gather the full sequence on the
  old layout, then slice each rank's new shard. Simple and obviously correct;
  moves ``(d-1)/d`` of the full sequence to every rank.
* **slice-intersection** (bandwidth-optimal): every rank keeps the overlap
  between its old and new shard ranges and exchanges only the *intersections*
  with the peers whose ranges it now overlaps, via P2P. This module owns the
  pure index math (``slice_intersections`` / ``rank_migration_plan``); the
  collective is a thin wrapper around it.

Note for Z-Image specifically: the task-level latent is kept **full and
replicated** on every rank (the sharding lives entirely inside ``cp_forward``),
so a degree switch needs *no* task-level data movement -- ``migrate_task_latents``
is a no-op there. These primitives exist for the general sharded-latent case
(and are exercised by the CPU unit tests + the migration micro-benchmark).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


def shard_bounds(seq_len: int, degree: int, rank: int) -> Tuple[int, int]:
    """Global ``[lo, hi)`` positions owned by ``rank`` when ``seq_len`` is split
    contiguously across ``degree`` ranks.

    Uses equal ``seq_len // degree`` chunks when divisible; otherwise the
    ``torch.tensor_split`` convention (first ``seq_len % degree`` shards get one
    extra element)."""
    if degree <= 0:
        raise ValueError(f"degree must be positive, got {degree}")
    if not (0 <= rank < degree):
        raise ValueError(f"rank {rank} out of range for degree {degree}")
    base = seq_len // degree
    rem = seq_len % degree
    if rem == 0:
        lo = rank * base
        return lo, lo + base
    # tensor_split: first `rem` shards have size base+1, the rest base.
    if rank < rem:
        lo = rank * (base + 1)
        return lo, lo + (base + 1)
    lo = rem * (base + 1) + (rank - rem) * base
    return lo, lo + base


@dataclass(frozen=True)
class Transfer:
    """A contiguous global slice ``[lo, hi)`` that moves from ``src`` (its owner
    in the OLD layout) to ``dst`` (its owner in the NEW layout)."""

    src: int
    dst: int
    lo: int
    hi: int

    @property
    def count(self) -> int:
        return self.hi - self.lo

    @property
    def is_local(self) -> bool:
        """True when the slice already lives on the right rank (no comm)."""
        return self.src == self.dst


def slice_intersections(seq_len: int, old_degree: int, new_degree: int) -> List[Transfer]:
    """Every non-empty overlap between an OLD shard and a NEW shard.

    The union of these transfers exactly tiles ``[0, seq_len)`` with no gaps or
    overlaps, so applying them reconstructs the new sharding losslessly. This is
    the core index math for the slice-intersection (P2P) migration.
    """
    olds = [shard_bounds(seq_len, old_degree, r) for r in range(old_degree)]
    news = [shard_bounds(seq_len, new_degree, r) for r in range(new_degree)]
    transfers: List[Transfer] = []
    for s, (olo, ohi) in enumerate(olds):
        for d, (nlo, nhi) in enumerate(news):
            lo = max(olo, nlo)
            hi = min(ohi, nhi)
            if hi > lo:
                transfers.append(Transfer(src=s, dst=d, lo=lo, hi=hi))
    return transfers


@dataclass(frozen=True)
class LocalTransfer:
    """A transfer expressed in a rank's LOCAL shard coordinates."""

    peer: int          # the other rank (src for recv, dst for send)
    local_offset: int  # offset into the local (old for send / new for recv) shard
    count: int
    global_lo: int
    global_hi: int


@dataclass
class MigrationPlan:
    """What a single rank must do to migrate from ``old_degree`` -> ``new_degree``."""

    rank: int
    seq_len: int
    old_degree: int
    new_degree: int
    # Slices this rank must SEND (in OLD-local coordinates) to a new owner.
    sends: List[LocalTransfer]
    # Slices this rank must RECEIVE (in NEW-local coordinates) from an old owner.
    recvs: List[LocalTransfer]
    # Slice this rank already holds that stays local (old-local -> new-local copy).
    keeps: List[Transfer]

    @property
    def is_participant_old(self) -> bool:
        return self.rank < self.old_degree

    @property
    def is_participant_new(self) -> bool:
        return self.rank < self.new_degree


def rank_migration_plan(seq_len: int, old_degree: int, new_degree: int, rank: int) -> MigrationPlan:
    """Build the per-rank send/recv/keep plan (slice-intersection algorithm).

    ``rank`` indexes into BOTH layouts (contiguous-block CP groups reuse the same
    global rank ordering). A rank that is not a participant of a layout simply
    has no sends (if absent from old) or no recvs (if absent from new)."""
    transfers = slice_intersections(seq_len, old_degree, new_degree)
    sends: List[LocalTransfer] = []
    recvs: List[LocalTransfer] = []
    keeps: List[Transfer] = []

    old_lo = shard_bounds(seq_len, old_degree, rank)[0] if rank < old_degree else None
    new_lo = shard_bounds(seq_len, new_degree, rank)[0] if rank < new_degree else None

    for t in transfers:
        if t.is_local and t.src == rank:
            keeps.append(t)
            continue
        if t.src == rank:
            sends.append(
                LocalTransfer(
                    peer=t.dst,
                    local_offset=t.lo - old_lo,
                    count=t.count,
                    global_lo=t.lo,
                    global_hi=t.hi,
                )
            )
        if t.dst == rank:
            recvs.append(
                LocalTransfer(
                    peer=t.src,
                    local_offset=t.lo - new_lo,
                    count=t.count,
                    global_lo=t.lo,
                    global_hi=t.hi,
                )
            )
    sends.sort(key=lambda x: x.global_lo)
    recvs.sort(key=lambda x: x.global_lo)
    keeps.sort(key=lambda x: x.lo)
    return MigrationPlan(
        rank=rank,
        seq_len=seq_len,
        old_degree=old_degree,
        new_degree=new_degree,
        sends=sends,
        recvs=recvs,
        keeps=keeps,
    )


def reshard_full_to_shard(full: torch.Tensor, degree: int, rank: int, seq_dim: int = 1) -> torch.Tensor:
    """Slice a rank's contiguous shard out of a full sequence tensor."""
    lo, hi = shard_bounds(full.shape[seq_dim], degree, rank)
    return full.narrow(seq_dim, lo, hi - lo).contiguous()


def gather_full_from_shards(shards: Sequence[torch.Tensor], seq_dim: int = 1) -> torch.Tensor:
    """Concatenate per-rank shards (in rank order) into the full sequence."""
    return torch.cat([s.contiguous() for s in shards], dim=seq_dim)


def migrate_gather_scatter(
    local_shard: torch.Tensor,
    old_group,
    new_degree: int,
    new_rank: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Correctness-first migration: all-gather the full sequence on ``old_group``
    then slice this rank's NEW shard.

    Assumes ``old_group`` collectively holds the full sequence (the usual CP
    layout). Returns the new local shard for ``new_rank`` at ``new_degree``.
    """
    import torch.distributed as dist

    old_size = old_group.group_size
    local_shard = local_shard.contiguous()
    pieces = [torch.empty_like(local_shard) for _ in range(old_size)]
    dist.all_gather(pieces, local_shard, group=old_group.gpu_group)
    full = gather_full_from_shards(pieces, seq_dim=seq_dim)
    return reshard_full_to_shard(full, new_degree, new_rank, seq_dim=seq_dim)
