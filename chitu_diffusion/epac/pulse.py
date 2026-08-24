from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from .scheduling import SchedulableRequest, SchedulingPolicy, StepPlan


@runtime_checkable
class StepCostPredictor(Protocol):
    def predict_step_ms(self, request: SchedulableRequest, width: int) -> float: ...


@dataclass(frozen=True, slots=True)
class LaneLease:
    """Exclusive use of a rank set until the next pulse deadline."""

    epoch: int
    ranks: tuple[int, ...]
    request_id: str | None
    min_steps: int
    deadline_ms: float
    predicted_step_ms: float | None = None
    allow_pull: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError("lease epoch must be non-negative")
        if not self.ranks or len(set(self.ranks)) != len(self.ranks):
            raise ValueError("lease ranks must be non-empty and unique")
        if self.request_id is None and self.min_steps != 0:
            raise ValueError("an idle lease must have min_steps=0")
        if self.request_id is not None and self.min_steps <= 0:
            raise ValueError("a working lease must have positive min_steps")
        if self.predicted_step_ms is not None and self.predicted_step_ms <= 0:
            raise ValueError("predicted_step_ms must be positive")

    @property
    def width(self) -> int:
        return len(self.ranks)

    @property
    def scheduling_width(self) -> int:
        """Return CP width rather than expanded physical TP×CP rank count."""

        return int(self.metadata.get("cp_width", self.width))

    def can_start_step(
        self,
        now_ms: float,
        *,
        predicted_step_ms: float | None = None,
        guard_ms: float = 0.0,
    ) -> bool:
        """Whether an opportunistic step should finish before lease expiry."""
        step_ms = predicted_step_ms or self.predicted_step_ms
        if step_ms is None:
            return False
        return now_ms + step_ms + max(0.0, guard_ms) <= self.deadline_ms


@dataclass(frozen=True, slots=True)
class LaneReport:
    """Lane-leader progress snapshot submitted at a pulse boundary."""

    epoch: int
    ranks: tuple[int, ...]
    request_id: str | None
    current_step: int | None
    state_owner_rank: int | None
    completed_request_ids: tuple[str, ...] = ()
    observed_step_ms: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.ranks:
            raise ValueError("report ranks must not be empty")
        if self.request_id is None and self.current_step is not None:
            raise ValueError("an idle report cannot carry current_step")
        if self.request_id is not None and (
            self.current_step is None or self.current_step < 0
        ):
            raise ValueError("a working report requires a non-negative current_step")
        if (
            self.state_owner_rank is not None
            and self.state_owner_rank not in self.ranks
        ):
            raise ValueError("state owner must belong to the reporting lane")
        if self.observed_step_ms is not None and self.observed_step_ms <= 0:
            raise ValueError("observed_step_ms must be positive")


@dataclass(frozen=True, slots=True)
class PulsePlan:
    """One globally coordinated lease epoch without a data-plane barrier."""

    epoch: int
    opened_at_ms: float
    deadline_ms: float
    leases: tuple[LaneLease, ...]

    def __post_init__(self) -> None:
        if self.deadline_ms <= self.opened_at_ms:
            raise ValueError("pulse deadline must be later than its open time")
        occupied: set[int] = set()
        for lease in self.leases:
            if lease.epoch != self.epoch:
                raise ValueError("all leases must belong to the pulse epoch")
            if lease.deadline_ms != self.deadline_ms:
                raise ValueError("all leases must share the pulse deadline")
            if occupied.intersection(lease.ranks):
                raise ValueError("pulse leases must use disjoint ranks")
            occupied.update(lease.ranks)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(sorted(rank for lease in self.leases for rank in lease.ranks))


class PulseCoordinator:
    """Control-plane state machine for semi-asynchronous EPAC lanes.

    Workers may execute extra steps or pull another request inside a lease,
    but ranks become globally reclaimable after every lane reports the epoch.
    """

    def __init__(
        self,
        *,
        world_size: int,
        policy: SchedulingPolicy,
        pulse_steps: int,
        min_pulse_ms: float = 1.0,
        clock_ms=None,
    ) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if pulse_steps <= 0:
            raise ValueError("pulse_steps must be positive")
        if min_pulse_ms <= 0:
            raise ValueError("min_pulse_ms must be positive")
        self.world_size = int(world_size)
        self.policy = policy
        self.pulse_steps = int(pulse_steps)
        self.min_pulse_ms = float(min_pulse_ms)
        self._clock_ms = clock_ms or (lambda: time.monotonic() * 1000.0)
        self._epoch = 0
        self._current: PulsePlan | None = None
        self._reports: dict[tuple[int, ...], LaneReport] = {}

    @property
    def current(self) -> PulsePlan | None:
        return self._current

    @property
    def last_plan_version(self) -> int:
        return self._epoch - 1

    def open(self, requests: Sequence[SchedulableRequest]) -> PulsePlan:
        if self._current is not None and not self.ready_to_replan:
            raise RuntimeError("cannot open a pulse before all lane reports arrive")
        now_ms = float(self._clock_ms())
        step_plans = tuple(self.policy.plan(requests, pulse_steps=self.pulse_steps))
        horizon_ms = max(
            (
                float(plan.metadata.get("predicted_phase_ms", 0.0))
                for plan in step_plans
            ),
            default=0.0,
        )
        deadline_ms = now_ms + max(self.min_pulse_ms, horizon_ms)
        plan = PulsePlan(
            epoch=self._epoch,
            opened_at_ms=now_ms,
            deadline_ms=deadline_ms,
            leases=self._leases_for(step_plans, deadline_ms),
        )
        self._epoch += 1
        self._current = plan
        self._reports = {}
        return plan

    def _leases_for(
        self, plans: Sequence[StepPlan], deadline_ms: float
    ) -> tuple[LaneLease, ...]:
        leases = []
        occupied: set[int] = set()
        constraints = getattr(self.policy, "constraints", None)
        if (
            not plans
            and constraints is not None
            and constraints.strategy == "static_cp"
        ):
            return (
                LaneLease(
                    epoch=self._epoch,
                    ranks=tuple(range(self.world_size)),
                    request_id=None,
                    min_steps=0,
                    deadline_ms=deadline_ms,
                ),
            )
        for plan in plans:
            ranks = tuple(plan.lane_ranks or ())
            if not ranks:
                raise ValueError("pulse policies must assign explicit lane ranks")
            if occupied.intersection(ranks):
                raise ValueError("pulse policy returned overlapping lanes")
            occupied.update(ranks)
            predicted = plan.metadata.get("predicted_step_ms")
            leases.append(
                LaneLease(
                    epoch=self._epoch,
                    ranks=ranks,
                    request_id=plan.request_id,
                    min_steps=plan.steps,
                    deadline_ms=deadline_ms,
                    predicted_step_ms=(None if predicted is None else float(predicted)),
                    metadata=dict(plan.metadata),
                )
            )
        for rank in range(self.world_size):
            if rank not in occupied:
                leases.append(
                    LaneLease(
                        epoch=self._epoch,
                        ranks=(rank,),
                        request_id=None,
                        min_steps=0,
                        deadline_ms=deadline_ms,
                    )
                )
        return tuple(sorted(leases, key=lambda lease: lease.ranks))

    def report(self, report: LaneReport) -> None:
        plan = self._require_current()
        if report.epoch != plan.epoch:
            raise ValueError(
                f"stale lane report epoch {report.epoch}; expected {plan.epoch}"
            )
        expected = {lease.ranks for lease in plan.leases}
        if report.ranks not in expected:
            raise ValueError("lane report does not match a current lease")
        if report.ranks in self._reports:
            raise ValueError("lane reported the pulse more than once")
        self._reports[report.ranks] = report

    @property
    def ready_to_replan(self) -> bool:
        if self._current is None:
            return True
        return len(self._reports) == len(self._current.leases)

    def reports(self) -> tuple[LaneReport, ...]:
        plan = self._require_current()
        if not self.ready_to_replan:
            raise RuntimeError("pulse reports are incomplete")
        return tuple(self._reports[lease.ranks] for lease in plan.leases)

    def choose_pull_request(
        self,
        lease: LaneLease,
        pending: Sequence[SchedulableRequest],
        *,
        now_ms: float | None = None,
        guard_ms: float = 0.0,
    ) -> SchedulableRequest | None:
        """Select globally ordered work that can start inside a lane lease."""
        if not lease.allow_pull or not pending:
            return None
        current_ms = float(self._clock_ms() if now_ms is None else now_ms)
        predictor = self.policy
        if not isinstance(predictor, StepCostPredictor):
            return None
        ordered = sorted(
            pending,
            key=lambda request: (
                request.deadline_at is None,
                (float("inf") if request.deadline_at is None else request.deadline_at),
                -request.priority,
                request.submitted_at,
            ),
        )
        for request in ordered:
            step_ms = predictor.predict_step_ms(request, lease.scheduling_width)
            if lease.can_start_step(
                current_ms,
                predicted_step_ms=step_ms,
                guard_ms=guard_ms,
            ):
                return request
        return None

    def _require_current(self) -> PulsePlan:
        if self._current is None:
            raise RuntimeError("no pulse is open")
        return self._current
