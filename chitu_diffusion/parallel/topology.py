from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agkv_transport import AgkvTransport
    from .ulysses_transport import UlyssesTransport


def select_ulysses_degree(
    cp_degree: int,
    heads: int,
    *,
    max_ulysses_degree: int,
) -> int:
    """Choose the largest head-compatible U factor for a U×R process mesh."""

    cp_degree = int(cp_degree)
    heads = int(heads)
    maximum = min(int(max_ulysses_degree), cp_degree)
    if cp_degree < 1 or heads < 1 or maximum < 1:
        raise ValueError("CP degree, heads, and maximum Ulysses degree must be positive")
    for degree in range(maximum, 0, -1):
        if cp_degree % degree == 0 and heads % degree == 0:
            return degree
    return 1


@dataclass(frozen=True, slots=True)
class UspTopology:
    """Unified Sequence Parallel view: Ulysses within groups, then Ring.

    ``ulysses_transport`` controls only the head/sequence all-to-all inside
    ``ulysses_ranks``. It does not implement or replace the Ring Attention
    phase across ``ring_ranks``.
    """

    lane_ranks: tuple[int, ...]
    ulysses_ranks: tuple[int, ...]
    ring_ranks: tuple[int, ...]
    ulysses_rank: int
    ring_rank: int
    ulysses_degree: int
    ring_degree: int
    ulysses_process_group: object | None
    ring_process_group: object | None
    ulysses_transport: UlyssesTransport | None = None
    agkv_transport: AgkvTransport | None = None
