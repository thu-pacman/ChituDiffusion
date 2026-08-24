from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .coordinator import LaneLease, LaneReport


@dataclass(frozen=True, slots=True)
class LaneTask:
    request_id: str
    current_step: int
    predicted_step_ms: float

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("lane task request_id must not be empty")
        if self.current_step < 0:
            raise ValueError("lane task current_step must be non-negative")
        if self.predicted_step_ms <= 0:
            raise ValueError("lane task predicted_step_ms must be positive")


@dataclass(frozen=True, slots=True)
class StepOutcome:
    current_step: int
    completed: bool
    observed_step_ms: float

    def __post_init__(self) -> None:
        if self.current_step < 0:
            raise ValueError("step outcome current_step must be non-negative")
        if self.observed_step_ms <= 0:
            raise ValueError("step outcome latency must be positive")


class LaneWorkHooks(Protocol):
    """Model/service hooks called by the model-independent lane loop."""

    def execute_step(self, task: LaneTask, ranks: tuple[int, ...]) -> StepOutcome: ...

    def complete_request(self, task: LaneTask) -> None: ...

    def pull_request(
        self, lease: LaneLease, completed_request_ids: tuple[str, ...]
    ) -> LaneTask | None: ...


class LaneWorker:
    """Runs one exclusive lane lease without synchronizing unrelated lanes."""

    def __init__(self, *, guard_ms: float = 0.0, clock_ms=None) -> None:
        if guard_ms < 0:
            raise ValueError("guard_ms must be non-negative")
        self.guard_ms = float(guard_ms)
        self._clock_ms = clock_ms or (lambda: time.monotonic() * 1000.0)

    def run_lease(
        self,
        lease: LaneLease,
        task: LaneTask | None,
        hooks: LaneWorkHooks,
    ) -> LaneReport:
        if lease.request_id is None and task is not None:
            raise ValueError("an idle lease cannot start with a task")
        if lease.request_id is not None and (
            task is None or task.request_id != lease.request_id
        ):
            raise ValueError("initial lane task must match the lease request")

        current = task
        required_steps = lease.min_steps
        completed: list[str] = []
        observed: list[float] = []

        if current is None and lease.allow_pull:
            candidate = hooks.pull_request(lease, ())
            if candidate is not None and lease.can_start_step(
                float(self._clock_ms()),
                predicted_step_ms=candidate.predicted_step_ms,
                guard_ms=self.guard_ms,
            ):
                current = candidate

        while current is not None:
            now_ms = float(self._clock_ms())
            required = required_steps > 0
            if not required and not lease.can_start_step(
                now_ms,
                predicted_step_ms=current.predicted_step_ms,
                guard_ms=self.guard_ms,
            ):
                break

            outcome = hooks.execute_step(current, lease.ranks)
            if outcome.current_step <= current.current_step:
                raise RuntimeError("lane step did not advance the request cursor")
            observed.append(outcome.observed_step_ms)
            current = LaneTask(
                request_id=current.request_id,
                current_step=outcome.current_step,
                predicted_step_ms=current.predicted_step_ms,
            )
            required_steps = max(0, required_steps - 1)

            if not outcome.completed:
                continue

            hooks.complete_request(current)
            completed.append(current.request_id)
            current = None
            required_steps = 0
            if lease.allow_pull:
                candidate = hooks.pull_request(lease, tuple(completed))
                if candidate is not None and lease.can_start_step(
                    float(self._clock_ms()),
                    predicted_step_ms=candidate.predicted_step_ms,
                    guard_ms=self.guard_ms,
                ):
                    current = candidate

        return LaneReport(
            epoch=lease.epoch,
            ranks=lease.ranks,
            request_id=None if current is None else current.request_id,
            current_step=None if current is None else current.current_step,
            state_owner_rank=None if current is None else lease.ranks[0],
            completed_request_ids=tuple(completed),
            observed_step_ms=(None if not observed else sum(observed) / len(observed)),
        )
