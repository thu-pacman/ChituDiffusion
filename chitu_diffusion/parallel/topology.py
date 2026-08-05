from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UspTopology:
    """Ulysses x Ring view for one currently active EPAC lane."""

    lane_ranks: tuple[int, ...]
    ulysses_ranks: tuple[int, ...]
    ring_ranks: tuple[int, ...]
    ulysses_rank: int
    ring_rank: int
    ulysses_degree: int
    ring_degree: int
    ulysses_process_group: object | None
    ring_process_group: object | None
