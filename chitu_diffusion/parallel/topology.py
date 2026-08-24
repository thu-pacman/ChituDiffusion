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
