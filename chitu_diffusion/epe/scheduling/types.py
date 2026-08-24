from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

ScheduleStrategy = Literal["elastic", "static_cp", "static_dp"]


@dataclass(frozen=True, slots=True)
class LaneConstraints:
    """Lane shapes allowed for one scheduling strategy.

    All strategies use the same planner and pulse executor. Static CP and DP
    are the two fixed-topology limits of the elastic lane space.
    """

    strategy: ScheduleStrategy
    world_size: int
    allowed_widths: tuple[int, ...]
    pin_running_lanes: bool = False

    @classmethod
    def create(
        cls,
        strategy: ScheduleStrategy,
        *,
        world_size: int,
        elastic_widths: Sequence[int],
    ) -> "LaneConstraints":
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        configured = tuple(sorted({int(width) for width in elastic_widths}))
        if not configured or any(
            width <= 0 or world_size % width for width in configured
        ):
            raise ValueError(
                "elastic lane widths must be positive divisors of world_size"
            )
        if strategy == "static_cp":
            widths = (world_size,)
        elif strategy == "static_dp":
            widths = (1,)
        elif strategy == "elastic":
            widths = configured
        else:
            raise ValueError(
                "schedule strategy must be one of: elastic, static_cp, static_dp"
            )
        return cls(
            strategy=strategy,
            world_size=world_size,
            allowed_widths=widths,
            pin_running_lanes=strategy == "static_dp",
        )


@dataclass(frozen=True, slots=True)
class RequestProfile:
    """Model-provided information consumed by EPE/cost-model policies."""

    total_steps: int
    completed_steps: int
    shape_key: tuple[int, ...] = ()
    image_tokens: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_steps(self) -> int:
        return max(0, self.total_steps - self.completed_steps)


@dataclass(frozen=True, slots=True)
class SchedulableRequest:
    request_id: str
    submitted_at: float
    deadline_at: float | None
    priority: int
    profile: RequestProfile
    current_ranks: tuple[int, ...] | None = None
    admitted: bool = True


@dataclass(frozen=True, slots=True)
class StepPlan:
    """One request's initial allocation within an EPE pulse."""

    request_id: str
    steps: int = 1
    lane_ranks: tuple[int, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("StepPlan.steps must be positive")
        if self.lane_ranks is not None and not self.lane_ranks:
            raise ValueError("lane_ranks must be None or non-empty")


class SchedulingPolicy(Protocol):
    """Produces the next pulse without knowing model implementation details."""

    def plan(
        self,
        requests: Sequence[SchedulableRequest],
        *,
        pulse_steps: int,
    ) -> Sequence[StepPlan]: ...


class RoundRobinPolicy:
    """Portable baseline: one step per active request per pulse."""

    def plan(
        self,
        requests: Sequence[SchedulableRequest],
        *,
        pulse_steps: int,
    ) -> Sequence[StepPlan]:
        del pulse_steps
        return tuple(
            StepPlan(request_id=request.request_id)
            for request in sorted(
                requests,
                key=lambda item: (-item.priority, item.submitted_at),
            )
            if request.profile.remaining_steps > 0
        )
