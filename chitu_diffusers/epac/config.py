from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Request limits and the slowest-lane planning horizon per pulse."""

    max_pending_requests: int = 64
    max_inflight_requests: int = 4
    pulse_steps: int = 5
    default_deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if self.max_pending_requests <= 0:
            raise ValueError("max_pending_requests must be positive")
        if self.max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be positive")
        if self.max_inflight_requests > self.max_pending_requests:
            raise ValueError(
                "max_inflight_requests must not exceed max_pending_requests"
            )
        if self.pulse_steps <= 0:
            raise ValueError("pulse_steps must be positive")
        if self.default_deadline_ms is not None and self.default_deadline_ms <= 0:
            raise ValueError("default_deadline_ms must be positive")
