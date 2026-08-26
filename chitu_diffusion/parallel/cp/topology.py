from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ulysses_transport import UlyssesTransport


@dataclass(frozen=True, slots=True)
class UlyssesTopology:
    """Static Ulysses context-parallel view for one active lane."""

    lane_ranks: tuple[int, ...]
    rank: int
    degree: int
    process_group: object | None
    transport: UlyssesTransport


@dataclass(frozen=True, slots=True)
class UspTopology:
    """Ulysses x Ring view used by packed multimodal attention."""

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
