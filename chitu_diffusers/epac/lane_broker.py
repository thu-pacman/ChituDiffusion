from __future__ import annotations

import math
import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from .pulse import LaneLease, LaneReport, PulseCoordinator, PulsePlan
from .scheduling import SchedulableRequest
from .worker_pool import LaneWorkResult

logger = logging.getLogger(__name__)


class PulseLaneBroker:
    """Rank-0 broker for barrierless lane progress and pulse replanning."""

    def __init__(
        self,
        *,
        world_size: int,
        coordinator: PulseCoordinator,
        schedulable: Callable[[bool], Sequence[SchedulableRequest]],
        reserve: Callable[[str, tuple[int, ...]], None],
        payload: Callable[[str], Any],
        update_progress: Callable[[str, int, tuple[int, ...]], None],
        complete: Callable[[LaneWorkResult], None],
        should_stop: Callable[[], bool],
        guard_ms: float = 2.0,
        clock_ms=None,
    ) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if guard_ms < 0:
            raise ValueError("guard_ms must be non-negative")
        self.world_size = int(world_size)
        self.coordinator = coordinator
        self._schedulable = schedulable
        self._reserve = reserve
        self._payload = payload
        self._update_progress = update_progress
        self._complete = complete
        self._should_stop = should_stop
        self.guard_ms = float(guard_ms)
        self._clock_ms = clock_ms or (lambda: time.monotonic() * 1000.0)
        self._condition = threading.Condition()
        self._events: dict[tuple[Any, ...], dict[int, dict]] = {}
        self._responses: dict[tuple[Any, ...], dict] = {}
        self._consumed: dict[tuple[Any, ...], int] = {}
        self._current: PulsePlan | None = None
        self._retired_request_ids: set[str] = set()

    def exchange(self, rank: int, event: dict) -> dict:
        key, expected = self._event_key(event)
        if rank not in expected:
            raise ValueError(f"rank {rank} is not part of event {key}")
        with self._condition:
            batch = self._events.setdefault(key, {})
            if rank in batch:
                raise ValueError(f"rank {rank} submitted event {key} twice")
            batch[rank] = event
            if len(batch) == len(expected):
                self._responses[key] = self._process(event["kind"], batch)
                self._consumed[key] = 0
                self._condition.notify_all()
            while key not in self._responses:
                self._condition.wait()
            response = self._responses[key]
            self._consumed[key] += 1
            if self._consumed[key] == len(expected):
                del self._events[key]
                del self._responses[key]
                del self._consumed[key]
            return response

    def _event_key(self, event: dict) -> tuple[tuple[Any, ...], tuple[int, ...]]:
        kind = str(event.get("kind"))
        sync = int(event.get("sync", 0))
        epoch = int(event.get("epoch", -1))
        if kind == "pulse_report":
            return (kind, epoch, sync), tuple(range(self.world_size))
        if kind == "lane_progress":
            ranks = tuple(int(rank) for rank in event["ranks"])
            return (kind, epoch, ranks, sync), ranks
        raise ValueError(f"unsupported lane broker event: {kind}")

    def _process(self, kind: str, events: dict[int, dict]) -> dict:
        if kind == "pulse_report":
            return self._process_pulse_reports(events)
        return self._process_lane_progress(events)

    def _process_pulse_reports(self, events: dict[int, dict]) -> dict:
        if self._current is not None:
            for lease in self._current.leases:
                leader_event = events[lease.ranks[0]]
                raw = leader_event.get("report")
                if raw is None:
                    raise RuntimeError("working pulse is missing a lane report")
                report = LaneReport(
                    epoch=self._current.epoch,
                    ranks=lease.ranks,
                    request_id=raw.get("request_id"),
                    current_step=raw.get("current_step"),
                    state_owner_rank=raw.get("state_owner_rank"),
                    completed_request_ids=tuple(raw.get("completed_request_ids", ())),
                    observed_step_ms=raw.get("observed_step_ms"),
                    error=raw.get("error"),
                )
                self.coordinator.report(report)
            if not self.coordinator.ready_to_replan:
                raise RuntimeError("pulse reports did not cover every lease")
            self._current = None

        retired = sorted(self._retired_request_ids)
        self._retired_request_ids.clear()
        if self._should_stop():
            return {"action": "stop", "retired_request_ids": retired}
        requests = tuple(self._schedulable(True))
        if not requests:
            return {"action": "idle", "retired_request_ids": retired}
        by_id = {request.request_id: request for request in requests}
        plan = self.coordinator.open(requests)
        self._current = plan
        assignments = []
        for lease in plan.leases:
            if lease.request_id is None:
                continue
            request = by_id[lease.request_id]
            old_ranks = request.current_ranks
            self._reserve(lease.request_id, lease.ranks)
            assignments.append(
                {
                    "request_id": lease.request_id,
                    "ranks": list(lease.ranks),
                    "old_ranks": None if old_ranks is None else list(old_ranks),
                    "current_step": request.profile.completed_steps,
                    "steps": lease.min_steps,
                    "predicted_step_ms": lease.predicted_step_ms,
                    "payload": self._payload(lease.request_id),
                }
            )
        return {
            "action": "pulse",
            "retired_request_ids": retired,
            "epoch": plan.epoch,
            "deadline_ms": plan.deadline_ms,
            "leases": [
                {
                    "ranks": list(lease.ranks),
                    "request_id": lease.request_id,
                    "steps": lease.min_steps,
                    "predicted_step_ms": lease.predicted_step_ms,
                }
                for lease in plan.leases
            ],
            "assignments": assignments,
        }

    def _process_lane_progress(self, events: dict[int, dict]) -> dict:
        leader = events[min(events)]
        epoch = int(leader["epoch"])
        ranks = tuple(int(rank) for rank in leader["ranks"])
        lease = self._lease(epoch, ranks)
        errors = [
            str(event["error"]) for event in events.values() if event.get("error")
        ]
        result = leader.get("result")
        request_id = leader.get("request_id")
        current_step = leader.get("current_step")
        if errors:
            if request_id is not None:
                result = LaneWorkResult(
                    request_id=request_id,
                    error="; ".join(errors),
                    metadata={"lane_ranks": list(ranks), "strategy": "elastic"},
                )
        if result is not None:
            if not isinstance(result, LaneWorkResult):
                raise TypeError("lane leader returned an invalid result")
            self._complete(result)
            self._retired_request_ids.add(result.request_id)
            request_id = None
            current_step = None
        elif request_id is not None and current_step is not None:
            self._update_progress(request_id, int(current_step), ranks)

        if self._should_stop():
            return {"action": "end"}
        if request_id is not None:
            request = self._request_by_id(request_id, include_pending=True)
            steps = self._steps_that_fit(lease, request)
            if steps > 0:
                return {
                    "action": "run",
                    "request_id": request_id,
                    "steps": steps,
                    "predicted_step_ms": self.coordinator.policy.predict_step_ms(
                        request, lease.width
                    ),
                    "payload": self._payload(request_id),
                }
            return {"action": "end"}

        pending = tuple(self._schedulable(False))
        candidate = self.coordinator.choose_pull_request(
            lease,
            pending,
            now_ms=float(self._clock_ms()),
            guard_ms=self.guard_ms,
        )
        if candidate is None:
            return {"action": "end"}
        steps = self._steps_that_fit(lease, candidate)
        if steps <= 0:
            return {"action": "end"}
        self._reserve(candidate.request_id, ranks)
        logger.info(
            "EPAC lane pull: epoch=%d request=%s ranks=%s step=%d+%d",
            lease.epoch,
            candidate.request_id,
            list(ranks),
            candidate.profile.completed_steps,
            steps,
        )
        return {
            "action": "run",
            "request_id": candidate.request_id,
            "steps": steps,
            "predicted_step_ms": self.coordinator.policy.predict_step_ms(
                candidate, lease.width
            ),
            "payload": self._payload(candidate.request_id),
        }

    def _lease(self, epoch: int, ranks: tuple[int, ...]) -> LaneLease:
        if self._current is None or self._current.epoch != epoch:
            raise ValueError(f"stale lane progress for epoch {epoch}")
        for lease in self._current.leases:
            if lease.ranks == ranks:
                return lease
        raise ValueError(f"epoch {epoch} has no lease for ranks {ranks}")

    def _steps_that_fit(self, lease: LaneLease, request: SchedulableRequest) -> int:
        predicted = self.coordinator.policy.predict_step_ms(request, lease.width)
        budget_ms = lease.deadline_ms - float(self._clock_ms()) - self.guard_ms
        if budget_ms < predicted:
            return 0
        return min(
            request.profile.remaining_steps,
            max(0, math.floor(budget_ms / predicted)),
        )

    def _request_by_id(
        self, request_id: str, *, include_pending: bool
    ) -> SchedulableRequest:
        for request in self._schedulable(include_pending):
            if request.request_id == request_id:
                return request
        raise KeyError(f"request {request_id!r} is no longer schedulable")
