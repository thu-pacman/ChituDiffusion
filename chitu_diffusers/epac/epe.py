from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from .cost import MeasuredStepCostModel
from .scheduling import (
    LaneConstraints,
    SchedulableRequest,
    ScheduleStrategy,
    StepPlan,
)


@dataclass(frozen=True, slots=True)
class _EpeRequest:
    request_id: str
    sequence_length: int
    total_steps: int
    current_step: int
    submitted_at_ms: float
    deadline_at_ms: float | None
    priority: int
    batch_size: int
    conditions: int
    current_ranks: tuple[int, ...] | None
    admitted: bool

    @property
    def remaining_steps(self) -> int:
        return max(0, self.total_steps - self.current_step)


class EpeSchedulingPolicy:
    """Measured-cost SLO planner shared by all EPAC lane strategies."""

    def __init__(
        self,
        *,
        world_size: int,
        allowed_lane_widths: Iterable[int],
        cost_model: MeasuredStepCostModel,
        strategy: ScheduleStrategy = "elastic",
        switch_allowed_until_step: int = 0,
        balanced_k: bool = True,
        starvation_ms: float = 30_000.0,
        deadline_guard_ms: float = 0.0,
        max_active_requests: int | None = None,
        clock_ms=None,
    ) -> None:
        self.constraints = LaneConstraints.create(
            strategy,
            world_size=int(world_size),
            elastic_widths=tuple(allowed_lane_widths),
        )
        self.world_size = self.constraints.world_size
        self.allowed_lane_widths = self.constraints.allowed_widths
        self.cost_model = cost_model
        self.switch_allowed_until_step = int(switch_allowed_until_step)
        self.balanced_k = bool(balanced_k)
        self.starvation_ms = float(starvation_ms)
        self.deadline_guard_ms = float(deadline_guard_ms)
        if self.deadline_guard_ms < 0:
            raise ValueError("deadline_guard_ms must be non-negative")
        self.max_active_requests = (
            None if max_active_requests is None else int(max_active_requests)
        )
        if self.max_active_requests is not None and self.max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")
        self._clock_ms = clock_ms or (lambda: time.time() * 1000.0)
        self._current_lanes: dict[str, tuple[int, ...]] = {}

    def plan(
        self,
        requests: Sequence[SchedulableRequest],
        *,
        pulse_steps: int,
    ) -> Sequence[StepPlan]:
        return self.plan_at(
            requests,
            pulse_steps=pulse_steps,
            now_ms=float(self._clock_ms()),
        )

    def plan_at(
        self,
        requests: Sequence[SchedulableRequest],
        *,
        pulse_steps: int,
        now_ms: float,
    ) -> Sequence[StepPlan]:
        if pulse_steps <= 0:
            raise ValueError("pulse_steps must be positive")
        if not self.cost_model.ready:
            raise RuntimeError("EPE requires startup warmup cost measurements")
        active_ids = {request.request_id for request in requests}
        self._current_lanes = {
            request_id: lane
            for request_id, lane in self._current_lanes.items()
            if request_id in active_ids
        }
        active = [self._convert(request) for request in requests]
        active = [request for request in active if request.remaining_steps > 0]
        if not active:
            return ()
        ordered = self._request_order(active, now_ms)
        admitted_count = sum(request.admitted for request in ordered)
        admission_slots = (
            len(ordered)
            if self.max_active_requests is None
            else max(0, self.max_active_requests - admitted_count)
        )
        layout_requests = self._layout_candidates(ordered, admission_slots)
        layouts = self._enumerate_layouts(
            layout_requests,
            admission_slots=admission_slots,
        )
        if not layouts:
            raise RuntimeError("EPE planner found no feasible lane layout")
        layout = min(
            layouts,
            key=lambda candidate: self._score_layout(
                candidate, ordered, now_ms, pulse_steps
            ),
        )
        assigned = [request for request in ordered if request.request_id in layout]
        latencies = [
            self._predict(request, len(layout[request.request_id]))
            for request in assigned
        ]
        steps = self._step_caps(assigned, latencies, pulse_steps)
        plans = []
        for request, latency, count in zip(assigned, latencies, steps):
            ranks = layout[request.request_id]
            previous = request.current_ranks
            plans.append(
                StepPlan(
                    request_id=request.request_id,
                    steps=count,
                    lane_ranks=ranks,
                    metadata={
                        "switched": previous is not None and previous != ranks,
                        "predicted_step_ms": latency,
                        "predicted_phase_ms": latency * count,
                        "sequence_length": request.sequence_length,
                    },
                )
            )
            self._current_lanes[request.request_id] = ranks
        return tuple(plans)

    def _convert(self, request: SchedulableRequest) -> _EpeRequest:
        profile = request.profile
        sequence_length = profile.image_tokens
        if sequence_length is None:
            sequence_length = profile.attributes.get("sequence_length")
        if sequence_length is None:
            raise ValueError(
                f"request {request.request_id!r} has no sequence length profile"
            )
        return _EpeRequest(
            request_id=request.request_id,
            sequence_length=int(sequence_length),
            total_steps=profile.total_steps,
            current_step=profile.completed_steps,
            submitted_at_ms=request.submitted_at * 1000.0,
            deadline_at_ms=(
                None if request.deadline_at is None else request.deadline_at * 1000.0
            ),
            priority=request.priority,
            batch_size=int(profile.attributes.get("batch_size", 1)),
            conditions=int(profile.attributes.get("conditions", 1)),
            admitted=request.admitted,
            current_ranks=(
                request.current_ranks
                if request.current_ranks is not None
                else self._current_lanes.get(request.request_id)
            ),
        )

    def _predict(self, request: _EpeRequest, width: int) -> float:
        return self.cost_model.predict_step_ms(
            sequence_length=request.sequence_length,
            width=width,
            batch_size=request.batch_size,
            conditions=request.conditions,
        )

    def predict_step_ms(self, request: SchedulableRequest, width: int) -> float:
        """Expose the measured estimate used for opportunistic lane pulls."""
        if width not in self.allowed_lane_widths:
            raise ValueError(
                f"lane width {width} is not allowed for {self.constraints.strategy}"
            )
        return self._predict(self._convert(request), width)

    def _best_step_ms(self, request: _EpeRequest) -> float:
        return min(self._predict(request, width) for width in self.allowed_lane_widths)

    def _request_order(
        self, requests: Iterable[_EpeRequest], now_ms: float
    ) -> list[_EpeRequest]:
        def key(request: _EpeRequest) -> tuple[float, float, float, float]:
            remaining_ms = request.remaining_steps * self._best_step_ms(request)
            if request.deadline_at_ms is not None:
                slack = (
                    request.deadline_at_ms
                    - self.deadline_guard_ms
                    - now_ms
                    - remaining_ms
                )
                return -request.priority, 0.0, slack, request.submitted_at_ms
            return -request.priority, 1.0, remaining_ms, request.submitted_at_ms

        return sorted(requests, key=key)

    def _layout_candidates(
        self,
        requests: list[_EpeRequest],
        admission_slots: int,
    ) -> list[_EpeRequest]:
        admitted = [request for request in requests if request.admitted]
        pending = [request for request in requests if not request.admitted]
        if self.max_active_requests is None:
            return requests
        # Full pending metadata remains visible to the completion predictor. Only
        # a bounded urgent prefix participates in the combinatorial lane search.
        choice_count = min(
            len(pending),
            max(admission_slots * 2, self.world_size),
        )
        candidate_ids = {
            request.request_id for request in admitted + pending[:choice_count]
        }
        return [request for request in requests if request.request_id in candidate_ids]

    def _canonical_lanes(self) -> tuple[tuple[int, ...], ...]:
        lanes = []
        for width in reversed(self.allowed_lane_widths):
            lanes.extend(
                tuple(range(offset, offset + width))
                for offset in range(0, self.world_size, width)
            )
        return tuple(lanes)

    def _enumerate_layouts(
        self,
        requests: list[_EpeRequest],
        *,
        admission_slots: int,
    ) -> list[dict[str, tuple[int, ...]]]:
        fixed: dict[str, tuple[int, ...]] = {}
        occupied: set[int] = set()
        flexible = []
        for request in requests:
            if request.current_ranks is not None and (
                self.constraints.pin_running_lanes
                or request.current_step >= self.switch_allowed_until_step
            ):
                if occupied.intersection(request.current_ranks):
                    raise RuntimeError("pinned EPE lanes overlap")
                fixed[request.request_id] = request.current_ranks
                occupied.update(request.current_ranks)
            else:
                flexible.append(request)

        candidates: list[dict[str, tuple[int, ...]]] = []
        lanes = self._canonical_lanes()

        def visit(
            index: int,
            current: dict[str, tuple[int, ...]],
            used: set[int],
            admitted_pending: int,
        ) -> None:
            if len(candidates) >= 8192:
                return
            if index >= len(flexible):
                if current:
                    candidates.append(dict(current))
                return
            request = flexible[index]
            can_admit = request.admitted or admitted_pending < admission_slots
            if can_admit:
                for lane in lanes:
                    if not used.intersection(lane):
                        current[request.request_id] = lane
                        visit(
                            index + 1,
                            current,
                            used.union(lane),
                            admitted_pending + int(not request.admitted),
                        )
                        current.pop(request.request_id)
            visit(index + 1, current, used, admitted_pending)

        visit(0, dict(fixed), occupied, 0)
        if not candidates and fixed:
            candidates.append(fixed)
        return candidates

    def _step_caps(
        self,
        requests: list[_EpeRequest],
        step_ms: list[float],
        pulse_steps: int,
    ) -> list[int]:
        caps = []
        for request in requests:
            cap = request.remaining_steps
            if request.current_step < self.switch_allowed_until_step:
                cap = min(
                    cap,
                    self.switch_allowed_until_step - request.current_step,
                )
            caps.append(max(1, cap))
        if not self.balanced_k:
            base = min(pulse_steps, min(caps))
            return [min(base, cap) for cap in caps]

        default_pulse_ms = pulse_steps * max(step_ms)
        completion_pulse_ms = min(
            request.remaining_steps * latency
            for request, latency in zip(requests, step_ms)
        )
        pulse_ms = min(default_pulse_ms, completion_pulse_ms)
        return [
            min(cap, max(1, math.floor(pulse_ms / latency + 0.5)))
            for cap, latency in zip(caps, step_ms)
        ]

    def _score_layout(
        self,
        layout: dict[str, tuple[int, ...]],
        requests: list[_EpeRequest],
        now_ms: float,
        pulse_steps: int,
    ) -> tuple[float, ...]:
        del pulse_steps
        by_id = {request.request_id: request for request in requests}
        assigned = [by_id[request_id] for request_id in layout]
        latencies = [
            self._predict(request, len(layout[request.request_id]))
            for request in assigned
        ]
        finish = self._predict_queue_completions(layout, requests, now_ms)
        tardiness = [
            max(
                0.0,
                finish[request.request_id]
                - (request.deadline_at_ms - self.deadline_guard_ms),
            )
            for request in requests
            if request.deadline_at_ms is not None
        ]
        starved = sum(
            now_ms - request.submitted_at_ms >= self.starvation_ms
            and request.request_id not in layout
            for request in requests
        )
        # Normalize progress by each request's cp1 cost. Raw steps/s heavily
        # favors short sequences and can spend the whole pool on an inefficient
        # cp4 small request while larger queued work makes no progress.
        baseline_width = min(self.allowed_lane_widths)
        effective_throughput = sum(
            self._predict(request, baseline_width) / latency
            for request, latency in zip(assigned, latencies)
        )
        raw_throughput = sum(1.0 / latency for latency in latencies)
        flow = [
            finish[request.request_id] - request.submitted_at_ms for request in requests
        ]
        slowdown = [
            max(
                1.0,
                flow_ms
                / max(
                    1.0,
                    request.total_steps
                    * self._predict(request, min(self.allowed_lane_widths)),
                ),
            )
            for request, flow_ms in zip(requests, flow)
        ]
        switches = sum(
            request.current_ranks is not None
            and request.current_ranks != layout.get(request.request_id)
            for request in requests
        )
        used = sum(len(lane) for lane in layout.values())
        return (
            float(sum(value > 0 for value in tardiness)),
            float(math.ceil(max(tardiness, default=0.0) / 500.0)),
            float(math.ceil(sum(tardiness) / 500.0)),
            float(starved),
            -float(len(assigned)),
            -effective_throughput,
            float(math.ceil(max(slowdown, default=1.0) / 0.25)),
            float(sum(slowdown)),
            float(switches),
            float(max(flow, default=0.0)),
            float(sum(flow)),
            -raw_throughput,
            float(self.world_size - used),
        )

    def _predict_queue_completions(
        self,
        layout: dict[str, tuple[int, ...]],
        requests: list[_EpeRequest],
        now_ms: float,
    ) -> dict[str, float]:
        """Predict full-queue completion times for one rolling pulse decision."""
        by_id = {request.request_id: request for request in requests}
        finish: dict[str, float] = {}
        running: list[tuple[float, int, int, str]] = []
        sequence = 0
        used = 0
        for request_id, lane in layout.items():
            request = by_id[request_id]
            width = len(lane)
            completion = now_ms + request.remaining_steps * self._predict(
                request, width
            )
            heapq.heappush(running, (completion, sequence, width, request_id))
            sequence += 1
            used += width

        waiting = [request for request in requests if request.request_id not in layout]
        admitted_states = sum(request.admitted for request in requests) + sum(
            not by_id[request_id].admitted for request_id in layout
        )
        active_limit = self.max_active_requests or len(requests)
        free_ranks = self.world_size - used

        while waiting:
            if not running:
                raise RuntimeError("queue predictor stalled without a running lane")
            current_ms = running[0][0]
            while running and running[0][0] <= current_ms + 1e-6:
                completion, _, width, request_id = heapq.heappop(running)
                finish[request_id] = completion
                free_ranks += width
                admitted_states -= 1

            while free_ranks > 0:
                candidate_index = next(
                    (
                        index
                        for index, request in enumerate(waiting)
                        if request.admitted or admitted_states < active_limit
                    ),
                    None,
                )
                if candidate_index is None:
                    break
                request = waiting.pop(candidate_index)
                width = self._future_lane_width(
                    request,
                    free_ranks=free_ranks,
                    waiting_count=len(waiting) + 1,
                    start_ms=current_ms,
                )
                if width is None:
                    waiting.insert(candidate_index, request)
                    break
                if not request.admitted:
                    admitted_states += 1
                completion = current_ms + request.remaining_steps * self._predict(
                    request, width
                )
                heapq.heappush(
                    running,
                    (completion, sequence, width, request.request_id),
                )
                sequence += 1
                free_ranks -= width

        while running:
            completion, _, _, request_id = heapq.heappop(running)
            finish[request_id] = completion
        return finish

    def _future_lane_width(
        self,
        request: _EpeRequest,
        *,
        free_ranks: int,
        waiting_count: int,
        start_ms: float,
    ) -> int | None:
        widths = [width for width in self.allowed_lane_widths if width <= free_ranks]
        if not widths:
            return None
        if request.deadline_at_ms is not None:
            effective_deadline = request.deadline_at_ms - self.deadline_guard_ms
            viable = [
                width
                for width in widths
                if start_ms + request.remaining_steps * self._predict(request, width)
                <= effective_deadline
            ]
            if viable:
                return min(viable)
            return min(widths, key=lambda width: self._predict(request, width))
        fair_share = max(1, free_ranks // waiting_count)
        share_widths = [width for width in widths if width <= fair_share]
        return min(
            share_widths or widths,
            key=lambda width: self._predict(request, width),
        )
