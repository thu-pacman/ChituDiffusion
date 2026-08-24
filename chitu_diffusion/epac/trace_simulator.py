from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from .cost import MeasuredStepCostModel, MeasuredTransferCostModel
from .epe import EpeSchedulingPolicy
from .scheduling import RequestProfile, SchedulableRequest, ScheduleStrategy


TraceAblationMode = Literal[
    "static-cp-max",
    "elastic-no-resize",
    "elastic-resize",
]


@dataclass(frozen=True, slots=True)
class TraceRequest:
    """One model-neutral arrival in a diffusion serving trace.

    ``profile_key`` is an opaque adapter-provided grouping key. Model geometry
    belongs in ``metadata`` and cost-model inputs belong in ``cost_features``.
    """

    request_id: str
    arrival_ms: float
    sequence_length: int
    num_steps: int
    profile_key: str = "default"
    deadline_ms: float | None = None
    priority: int = 0
    batch_size: int = 1
    conditions: int = 1
    state_bytes: int = 0
    cost_features: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.arrival_ms < 0:
            raise ValueError("arrival_ms must be non-negative")
        if min(
            self.sequence_length,
            self.num_steps,
            self.batch_size,
            self.conditions,
        ) <= 0:
            raise ValueError("trace work sizes must be positive")
        if not self.profile_key:
            raise ValueError("profile_key must not be empty")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        if self.state_bytes < 0:
            raise ValueError("state_bytes must be non-negative")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TraceRequest":
        known = {
            "request_id",
            "arrival_ms",
            "sequence_length",
            "num_steps",
            "profile_key",
            "deadline_ms",
            "priority",
            "batch_size",
            "conditions",
            "state_bytes",
            "cost_features",
            "metadata",
        }
        metadata = dict(raw.get("metadata", {}))
        metadata.update({key: value for key, value in raw.items() if key not in known})
        return cls(
            request_id=str(raw["request_id"]),
            arrival_ms=float(raw["arrival_ms"]),
            sequence_length=int(raw["sequence_length"]),
            num_steps=int(raw["num_steps"]),
            profile_key=str(raw.get("profile_key", "default")),
            deadline_ms=(
                None
                if raw.get("deadline_ms") is None
                else float(raw["deadline_ms"])
            ),
            priority=int(raw.get("priority", 0)),
            batch_size=int(raw.get("batch_size", 1)),
            conditions=int(raw.get("conditions", 1)),
            state_bytes=int(raw.get("state_bytes", 0)),
            cost_features=dict(raw.get("cost_features", {})),
            metadata=metadata,
        )

    def to_schedulable(
        self,
        *,
        completed_steps: int,
        current_ranks: tuple[int, ...] | None,
        admitted: bool,
    ) -> SchedulableRequest:
        return SchedulableRequest(
            request_id=self.request_id,
            submitted_at=self.arrival_ms / 1000.0,
            deadline_at=(
                None
                if self.deadline_ms is None
                else (self.arrival_ms + self.deadline_ms) / 1000.0
            ),
            priority=self.priority,
            profile=RequestProfile(
                total_steps=self.num_steps,
                completed_steps=completed_steps,
                image_tokens=self.sequence_length,
                attributes={
                    "sequence_length": self.sequence_length,
                    "batch_size": self.batch_size,
                    "conditions": self.conditions,
                    "state_bytes": self.state_bytes,
                    "cost_features": dict(self.cost_features),
                },
            ),
            current_ranks=current_ranks,
            admitted=admitted,
        )


@dataclass(frozen=True, slots=True)
class TracePhase:
    request_id: str
    pulse: int
    start_ms: float
    compute_start_ms: float
    end_ms: float
    start_step: int
    steps: int
    cp_width: int
    physical_ranks: tuple[int, ...]
    switched: bool
    predicted_step_ms: float
    transfer_ms: float
    terminal_lane_ms: float


@dataclass(frozen=True, slots=True)
class TraceRequestResult:
    request_id: str
    arrival_ms: float
    first_start_ms: float
    lane_complete_ms: float
    completion_ms: float
    latency_ms: float
    queue_ms: float
    deadline_ms: float | None
    deadline_met: bool | None
    profile_key: str
    sequence_length: int
    num_steps: int
    switches: int


@dataclass(frozen=True, slots=True)
class TraceSimulationResult:
    strategy: str
    ablation_mode: str
    metrics: Mapping[str, Any]
    requests: tuple[TraceRequestResult, ...]
    timeline: tuple[TracePhase, ...]
    planner: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "ablation_mode": self.ablation_mode,
            "metrics": dict(self.metrics),
            "requests": [asdict(request) for request in self.requests],
            "timeline": [asdict(phase) for phase in self.timeline],
            "planner": dict(self.planner),
        }


@dataclass(slots=True)
class _TraceState:
    request: TraceRequest
    completed_steps: int = 0
    current_ranks: tuple[int, ...] | None = None
    admitted: bool = False
    first_start_ms: float | None = None
    lane_complete_ms: float | None = None
    completion_ms: float | None = None
    switches: int = 0


class EpeTraceSimulator:
    """Pulse-level replay using the production EPE scheduling policy."""

    def __init__(
        self,
        *,
        cost_model: MeasuredStepCostModel,
        transfer_cost_model: MeasuredTransferCostModel | None = None,
        tp_degree: int = 2,
        cp_world_size: int = 4,
        allowed_lane_widths: Sequence[int] = (1, 2, 4),
        strategy: ScheduleStrategy = "elastic",
        pulse_steps: int = 2,
        switch_allowed_until_step: int = 20,
        balanced_k: bool = True,
        starvation_ms: float = 30_000.0,
        deadline_guard_ms: float = 0.0,
        min_resize_gain_ms: float = 0.0,
        resize_hysteresis_ms: float = 0.0,
        resize_control_cost_ms: float = 0.0,
        max_active_requests: int | None = None,
        ablation_mode: TraceAblationMode | None = None,
        planner_overhead_ms: float = 0.0,
    ) -> None:
        if pulse_steps <= 0:
            raise ValueError("pulse_steps must be positive")
        if planner_overhead_ms < 0:
            raise ValueError("planner_overhead_ms must be non-negative")
        if ablation_mode is None:
            ablation_mode = (
                "static-cp-max" if strategy == "static_cp" else "elastic-resize"
            )
        if ablation_mode not in (
            "static-cp-max",
            "elastic-no-resize",
            "elastic-resize",
        ):
            raise ValueError(f"unknown trace ablation mode: {ablation_mode}")
        expected_strategy: ScheduleStrategy = (
            "static_cp" if ablation_mode == "static-cp-max" else "elastic"
        )
        if strategy != expected_strategy:
            raise ValueError(
                f"{ablation_mode} requires strategy={expected_strategy!r}"
            )
        self.cost_model = cost_model
        self.transfer_cost_model = transfer_cost_model
        self.tp_degree = int(tp_degree)
        self.cp_world_size = int(cp_world_size)
        self.allowed_lane_widths = tuple(int(value) for value in allowed_lane_widths)
        self.strategy = strategy
        self.ablation_mode = ablation_mode
        self.pulse_steps = int(pulse_steps)
        self.planner_overhead_ms = float(planner_overhead_ms)
        self.policy = EpeSchedulingPolicy(
            world_size=self.cp_world_size,
            tp_degree=self.tp_degree,
            allowed_lane_widths=self.allowed_lane_widths,
            cost_model=self.cost_model,
            transfer_cost_model=self.transfer_cost_model,
            strategy=self.strategy,
            switch_allowed_until_step=switch_allowed_until_step,
            balanced_k=balanced_k,
            starvation_ms=starvation_ms,
            deadline_guard_ms=deadline_guard_ms,
            min_resize_gain_ms=min_resize_gain_ms,
            resize_hysteresis_ms=resize_hysteresis_ms,
            resize_control_cost_ms=resize_control_cost_ms,
            max_active_requests=max_active_requests,
        )
        if self.ablation_mode in ("static-cp-max", "elastic-no-resize"):
            self.policy.constraints = replace(
                self.policy.constraints,
                pin_running_lanes=True,
            )

    def run(self, trace: Sequence[TraceRequest]) -> TraceSimulationResult:
        ordered = sorted(trace, key=lambda item: (item.arrival_ms, item.request_id))
        if not ordered:
            raise ValueError("trace must contain at least one request")
        if len({request.request_id for request in ordered}) != len(ordered):
            raise ValueError("trace request IDs must be unique")

        states = {request.request_id: _TraceState(request) for request in ordered}
        timeline: list[TracePhase] = []
        now_ms = ordered[0].arrival_ms
        pulse = 0
        candidate_total = 0
        planning_total_ms = 0.0
        simulated_planning_ms = 0.0
        dispatch_total_ms = 0.0
        pulse_barrier_idle_cp_ms = 0.0
        queue_size_total = 0
        queue_size_max = 0

        while any(state.completion_ms is None for state in states.values()):
            visible = [
                state
                for state in states.values()
                if state.request.arrival_ms <= now_ms + 1e-9
                and state.lane_complete_ms is None
            ]
            if not visible:
                future = [
                    state.request.arrival_ms
                    for state in states.values()
                    if state.request.arrival_ms > now_ms
                ]
                if not future:
                    break
                now_ms = min(future)
                continue

            if self.ablation_mode == "elastic-resize":
                # Simulator state is authoritative; policy-local lanes may be
                # stale after opportunistic pulls within the previous lease.
                self.policy._current_lanes.clear()
                occupied: set[int] = set()
                for state in sorted(
                    visible,
                    key=lambda item: (
                        not item.admitted,
                        -item.completed_steps,
                        item.request.arrival_ms,
                    ),
                ):
                    ranks = state.current_ranks
                    scheduling_ranks = (
                        None
                        if ranks is None
                        else {rank % self.cp_world_size for rank in ranks}
                    )
                    if scheduling_ranks is not None and occupied.intersection(
                        scheduling_ranks
                    ):
                        state.current_ranks = None
                        self.policy._current_lanes.pop(
                            state.request.request_id,
                            None,
                        )
                    elif scheduling_ranks is not None:
                        occupied.update(scheduling_ranks)
            plans = self.policy.plan_at(
                [
                    state.request.to_schedulable(
                        completed_steps=state.completed_steps,
                        current_ranks=state.current_ranks,
                        admitted=state.admitted,
                    )
                    for state in visible
                ],
                pulse_steps=self.pulse_steps,
                now_ms=now_ms,
            )
            if not plans:
                raise RuntimeError("EPE trace replay stalled without a plan")
            if self.ablation_mode == "elastic-resize":
                planned_ids = {plan.request_id for plan in plans}
                # A request omitted from the new layout no longer owns its old
                # lane. Keeping that stale allocation would make the next
                # current-layout candidate overlap the lanes assigned here.
                for state in visible:
                    if state.request.request_id not in planned_ids:
                        state.current_ranks = None
                        self.policy._current_lanes.pop(
                            state.request.request_id,
                            None,
                        )
            candidate_total += int(self.policy.last_plan_stats["candidate_count"])
            planning_total_ms += float(self.policy.last_plan_stats["planning_ms"])
            queue_size = int(self.policy.last_plan_stats["queue_size"])
            queue_size_total += queue_size
            queue_size_max = max(queue_size_max, queue_size)
            if self.planner_overhead_ms:
                now_ms += self.planner_overhead_ms
                simulated_planning_ms += self.planner_overhead_ms
            pulse_horizon_ms = max(
                float(plan.metadata["predicted_phase_ms"]) for plan in plans
            )
            dispatch_total_ms += max(
                0.0,
                float(plans[0].metadata["predicted_dispatch_ms"])
                - sum(
                    float(plan.metadata["predicted_transfer_ms"])
                    for plan in plans
                ),
            )
            lane_runs = []

            for plan in plans:
                state = states[plan.request_id]
                metadata = plan.metadata
                start_step = state.completed_steps
                steps = min(plan.steps, state.request.num_steps - start_step)
                phase_ms = float(metadata["predicted_dispatch_ms"]) + float(
                    metadata["predicted_step_ms"]
                ) * steps
                terminal_lane_ms = 0.0
                if start_step + steps >= state.request.num_steps:
                    terminal_lane_ms = float(metadata["predicted_terminal_ms"])
                    phase_ms += terminal_lane_ms
                physical_ranks = tuple(plan.lane_ranks or ())
                transfer_ms = float(metadata["predicted_transfer_ms"])
                compute_start_ms = now_ms + float(
                    metadata["predicted_dispatch_ms"]
                )
                phase_end_ms = now_ms + phase_ms
                switched = bool(metadata["switched"])
                timeline.append(
                    TracePhase(
                        request_id=state.request.request_id,
                        pulse=pulse,
                        start_ms=now_ms,
                        compute_start_ms=compute_start_ms,
                        end_ms=phase_end_ms,
                        start_step=start_step,
                        steps=steps,
                        cp_width=int(metadata["cp_width"]),
                        physical_ranks=physical_ranks,
                        switched=switched,
                        predicted_step_ms=float(metadata["predicted_step_ms"]),
                        transfer_ms=transfer_ms,
                        terminal_lane_ms=terminal_lane_ms,
                    )
                )
                if state.first_start_ms is None:
                    state.first_start_ms = compute_start_ms
                state.admitted = True
                state.completed_steps += steps
                state.current_ranks = physical_ranks
                state.switches += int(switched)
                if state.completed_steps >= state.request.num_steps:
                    self._finish_state(
                        state,
                        lane_complete_ms=phase_end_ms,
                        cp_width=int(metadata["cp_width"]),
                        terminal_lane_ms=terminal_lane_ms,
                    )
                lane_runs.append(
                    (
                        phase_end_ms,
                        int(metadata["cp_width"]),
                        physical_ranks,
                        state.request.request_id,
                    )
                )

            pulse_deadline_ms = now_ms + pulse_horizon_ms
            self._run_opportunistic_work(
                states,
                timeline,
                lane_runs=lane_runs,
                pulse=pulse,
                deadline_ms=pulse_deadline_ms,
            )
            occupied_cp_ms = sum(
                cp_width * max(0.0, min(end_ms, pulse_deadline_ms) - now_ms)
                for end_ms, cp_width, _, _ in lane_runs
            )
            pulse_barrier_idle_cp_ms += max(
                0.0,
                self.cp_world_size * pulse_horizon_ms - occupied_cp_ms,
            )
            now_ms = pulse_deadline_ms
            pulse += 1

        results = tuple(
            self._request_result(states[request.request_id]) for request in ordered
        )
        metrics = self._metrics(
            results,
            timeline,
            overhead={
                "planner_wall_ms": planning_total_ms,
                "planner_simulated_ms": simulated_planning_ms,
                "dispatch_ms": dispatch_total_ms,
                "migration_ms": sum(phase.transfer_ms for phase in timeline),
                "terminal_lane_ms": sum(
                    phase.terminal_lane_ms for phase in timeline
                ),
                "pulse_barrier_idle_cp_ms": pulse_barrier_idle_cp_ms,
            },
        )
        return TraceSimulationResult(
            strategy=self.strategy,
            ablation_mode=self.ablation_mode,
            metrics=metrics,
            requests=results,
            timeline=tuple(timeline),
            planner={
                "tp_degree": self.tp_degree,
                "cp_world_size": self.cp_world_size,
                "physical_world_size": self.tp_degree * self.cp_world_size,
                "allowed_lane_widths": list(self.policy.allowed_lane_widths),
                "pulse_steps": self.pulse_steps,
                "pulses": pulse,
                "candidate_count": candidate_total,
                "planning_ms": planning_total_ms,
                "planner_overhead_ms": self.planner_overhead_ms,
                "queue_size_mean": queue_size_total / max(1, pulse),
                "queue_size_max": queue_size_max,
            },
        )

    def _run_opportunistic_work(
        self,
        states: Mapping[str, _TraceState],
        timeline: list[TracePhase],
        *,
        lane_runs: Sequence[tuple[float, int, tuple[int, ...], str]],
        pulse: int,
        deadline_ms: float,
    ) -> None:
        """Model the serve broker continuing or pulling work inside a lease."""

        queue = [
            (available_ms, index, cp_width, ranks, request_id)
            for index, (available_ms, cp_width, ranks, request_id) in enumerate(
                lane_runs
            )
        ]
        queue.sort()
        reserved = {
            request_id
            for _, _, _, request_id in lane_runs
            if states[request_id].lane_complete_ms is None
        }
        sequence = len(queue)
        while queue:
            available_ms, _, cp_width, ranks, request_id = queue.pop(0)
            if available_ms >= deadline_ms - 1e-9:
                continue
            state = states[request_id]
            if state.lane_complete_ms is not None:
                reserved.discard(request_id)
                candidates = [
                    candidate
                    for candidate in states.values()
                    if candidate.request.arrival_ms <= available_ms + 1e-9
                    and candidate.lane_complete_ms is None
                    and not candidate.admitted
                    and candidate.request.request_id not in reserved
                ]
                candidates.sort(
                    key=lambda candidate: (
                        candidate.request.deadline_ms is None,
                        (
                            math.inf
                            if candidate.request.deadline_ms is None
                            else candidate.request.arrival_ms
                            + candidate.request.deadline_ms
                        ),
                        -candidate.request.priority,
                        candidate.request.arrival_ms,
                    )
                )
                state = next(
                    (
                        candidate
                        for candidate in candidates
                        if available_ms
                        + self._predict_step_ms(candidate, cp_width)
                        <= deadline_ms
                    ),
                    None,
                )
                if state is None:
                    continue
                request_id = state.request.request_id
                state.admitted = True
                reserved.add(request_id)

            predicted_step_ms = self._predict_step_ms(state, cp_width)
            remaining = state.request.num_steps - state.completed_steps
            budget_ms = deadline_ms - available_ms
            steps = min(remaining, max(0, math.floor(budget_ms / predicted_step_ms)))
            terminal_lane_ms = 0.0
            if steps >= remaining:
                terminal_lane_ms = self._predict_terminal_ms(state, cp_width)
                if remaining * predicted_step_ms + terminal_lane_ms > budget_ms:
                    steps = remaining - 1
                    terminal_lane_ms = 0.0
            if steps <= 0:
                continue

            start_step = state.completed_steps
            end_ms = (
                available_ms + steps * predicted_step_ms + terminal_lane_ms
            )
            switched = state.current_ranks is not None and state.current_ranks != ranks
            timeline.append(
                TracePhase(
                    request_id=request_id,
                    pulse=pulse,
                    start_ms=available_ms,
                    compute_start_ms=available_ms,
                    end_ms=end_ms,
                    start_step=start_step,
                    steps=steps,
                    cp_width=cp_width,
                    physical_ranks=ranks,
                    switched=switched,
                    predicted_step_ms=predicted_step_ms,
                    transfer_ms=0.0,
                    terminal_lane_ms=terminal_lane_ms,
                )
            )
            if state.first_start_ms is None:
                state.first_start_ms = available_ms
            state.completed_steps += steps
            state.current_ranks = ranks
            state.switches += int(switched)
            if state.completed_steps >= state.request.num_steps:
                self._finish_state(
                    state,
                    lane_complete_ms=end_ms,
                    cp_width=cp_width,
                    terminal_lane_ms=terminal_lane_ms,
                )
            queue.append((end_ms, sequence, cp_width, ranks, request_id))
            sequence += 1
            queue.sort()

    def _finish_state(
        self,
        state: _TraceState,
        *,
        lane_complete_ms: float,
        cp_width: int,
        terminal_lane_ms: float,
    ) -> None:
        state.lane_complete_ms = lane_complete_ms
        completion_tail_ms = self._predict_completion_ms(state, cp_width)
        state.completion_ms = lane_complete_ms + max(
            0.0, completion_tail_ms - terminal_lane_ms
        )

    def _predict_step_ms(self, state: _TraceState, width: int) -> float:
        return self.policy.predict_step_ms(
            state.request.to_schedulable(
                completed_steps=state.completed_steps,
                current_ranks=state.current_ranks,
                admitted=state.admitted,
            ),
            width,
        )

    def _predict_terminal_ms(self, state: _TraceState, width: int) -> float:
        return self.policy.predict_terminal_ms(
            state.request.to_schedulable(
                completed_steps=state.completed_steps,
                current_ranks=state.current_ranks,
                admitted=state.admitted,
            ),
            width,
        )

    def _predict_completion_ms(self, state: _TraceState, width: int) -> float:
        predictor = getattr(self.cost_model, "predict_profile_completion_ms", None)
        profile_aware = predictor is not None
        if predictor is None:
            predictor = self.cost_model.predict_completion_ms
        return float(
            predictor(
                sequence_length=state.request.sequence_length,
                width=width,
                batch_size=state.request.batch_size,
                conditions=state.request.conditions,
                **(
                    {"cost_features": dict(state.request.cost_features)}
                    if profile_aware
                    else {}
                ),
            )
        )

    @staticmethod
    def _request_result(state: _TraceState) -> TraceRequestResult:
        if (
            state.first_start_ms is None
            or state.lane_complete_ms is None
            or state.completion_ms is None
        ):
            raise RuntimeError(f"request {state.request.request_id} did not complete")
        latency_ms = state.completion_ms - state.request.arrival_ms
        deadline_met = (
            None
            if state.request.deadline_ms is None
            else latency_ms <= state.request.deadline_ms
        )
        return TraceRequestResult(
            request_id=state.request.request_id,
            arrival_ms=state.request.arrival_ms,
            first_start_ms=state.first_start_ms,
            lane_complete_ms=state.lane_complete_ms,
            completion_ms=state.completion_ms,
            latency_ms=latency_ms,
            queue_ms=state.first_start_ms - state.request.arrival_ms,
            deadline_ms=state.request.deadline_ms,
            deadline_met=deadline_met,
            profile_key=state.request.profile_key,
            sequence_length=state.request.sequence_length,
            num_steps=state.request.num_steps,
            switches=state.switches,
        )

    def _metrics(
        self,
        requests: Sequence[TraceRequestResult],
        timeline: Sequence[TracePhase],
        *,
        overhead: Mapping[str, float],
    ) -> dict[str, Any]:
        first_arrival = min(request.arrival_ms for request in requests)
        last_completion = max(request.completion_ms for request in requests)
        last_lane_completion = max(request.lane_complete_ms for request in requests)
        makespan_ms = last_completion - first_arrival
        lane_window_ms = max(1e-9, last_lane_completion - first_arrival)
        cp_lane_ms = sum(
            phase.cp_width * (phase.end_ms - phase.start_ms) for phase in timeline
        )
        deadline_values = [
            request.deadline_met
            for request in requests
            if request.deadline_met is not None
        ]
        by_profile: dict[str, list[float]] = {}
        for request in requests:
            by_profile.setdefault(request.profile_key, []).append(request.latency_ms)
        met_count = sum(bool(value) for value in deadline_values)
        return {
            "request_count": len(requests),
            "makespan_ms": makespan_ms,
            "throughput_rps": len(requests) * 1000.0 / max(1e-9, makespan_ms),
            "step_throughput_sps": sum(request.num_steps for request in requests)
            * 1000.0
            / max(1e-9, makespan_ms),
            "latency_mean_ms": sum(request.latency_ms for request in requests)
            / len(requests),
            "latency_p50_ms": _percentile(
                [request.latency_ms for request in requests], 50
            ),
            "latency_p95_ms": _percentile(
                [request.latency_ms for request in requests], 95
            ),
            "latency_p99_ms": _percentile(
                [request.latency_ms for request in requests], 99
            ),
            "queue_p95_ms": _percentile(
                [request.queue_ms for request in requests], 95
            ),
            "deadline_attainment": (
                None
                if not deadline_values
                else met_count / len(deadline_values)
            ),
            "deadline_met_count": met_count,
            "deadline_miss_count": len(deadline_values) - met_count,
            "slo_goodput_rps": (
                None
                if not deadline_values
                else met_count * 1000.0 / max(1e-9, makespan_ms)
            ),
            "cp_utilization": cp_lane_ms
            / (self.cp_world_size * lane_window_ms),
            "switch_count": sum(request.switches for request in requests),
            "latency_by_profile_ms": {
                key: {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "p95": _percentile(values, 95),
                }
                for key, values in sorted(by_profile.items())
            },
            "overhead_ms": dict(overhead),
        }


def load_trace(path: str | Path) -> tuple[TraceRequest, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    values = raw["requests"] if isinstance(raw, dict) else raw
    if not isinstance(values, list):
        raise TypeError("trace JSON must be a list or contain a requests list")
    return tuple(TraceRequest.from_mapping(value) for value in values)


def save_trace(path: str | Path, trace: Sequence[TraceRequest]) -> None:
    payload = {"requests": [asdict(request) for request in trace]}
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_warmup_models(
    path: str | Path,
) -> tuple[MeasuredStepCostModel, MeasuredTransferCostModel]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("warmup report must contain non-empty rows")
    cost_model = MeasuredStepCostModel()
    cost_model.initialize(rows)
    transfer_model = MeasuredTransferCostModel()
    transfer_rows = report.get("transfer_rows", [])
    if not isinstance(transfer_rows, list):
        raise TypeError("warmup transfer_rows must be a list")
    transfer_model.initialize(transfer_rows)
    return cost_model, transfer_model


def compare_epe_to_static_cp(
    trace: Sequence[TraceRequest],
    *,
    cost_model: MeasuredStepCostModel,
    transfer_cost_model: MeasuredTransferCostModel | None = None,
    tp_degree: int = 2,
    cp_world_size: int = 4,
    allowed_lane_widths: Sequence[int] = (1, 2, 4),
    pulse_steps: int = 2,
    switch_allowed_until_step: int = 20,
    balanced_k: bool = True,
    starvation_ms: float = 30_000.0,
    deadline_guard_ms: float = 0.0,
    min_resize_gain_ms: float = 0.0,
    resize_hysteresis_ms: float = 0.0,
    resize_control_cost_ms: float = 0.0,
    max_active_requests: int | None = None,
    planner_overhead_ms: float = 0.0,
) -> dict[str, Any]:
    common = {
        "cost_model": cost_model,
        "transfer_cost_model": transfer_cost_model,
        "tp_degree": tp_degree,
        "cp_world_size": cp_world_size,
        "allowed_lane_widths": allowed_lane_widths,
        "pulse_steps": pulse_steps,
        "switch_allowed_until_step": switch_allowed_until_step,
        "balanced_k": balanced_k,
        "starvation_ms": starvation_ms,
        "deadline_guard_ms": deadline_guard_ms,
        "min_resize_gain_ms": min_resize_gain_ms,
        "resize_hysteresis_ms": resize_hysteresis_ms,
        "resize_control_cost_ms": resize_control_cost_ms,
        "max_active_requests": max_active_requests,
        "planner_overhead_ms": planner_overhead_ms,
    }
    elastic = EpeTraceSimulator(
        strategy="elastic",
        ablation_mode="elastic-resize",
        **common,
    ).run(trace)
    static = EpeTraceSimulator(
        strategy="static_cp",
        ablation_mode="static-cp-max",
        **common,
    ).run(trace)
    elastic_throughput = float(elastic.metrics["throughput_rps"])
    static_throughput = float(static.metrics["throughput_rps"])
    elastic_p95 = float(elastic.metrics["latency_p95_ms"])
    static_p95 = float(static.metrics["latency_p95_ms"])
    return {
        "comparison": {
            "baseline": f"tp{tp_degree}cp{cp_world_size}",
            "throughput_speedup": elastic_throughput / static_throughput,
            "p95_latency_reduction": 1.0 - elastic_p95 / static_p95,
        },
        "elastic": elastic.to_dict(),
        "static_tp_cp": static.to_dict(),
    }


def generate_profile_trace(
    *,
    request_count: int,
    arrival_rate_rps: float,
    profiles: Sequence[Mapping[str, Any]],
    deadline_ms: float | None = None,
    seed: int = 0,
    cycle_profiles: bool = False,
) -> tuple[TraceRequest, ...]:
    """Generate Poisson arrivals from opaque model-provided profiles."""

    if request_count <= 0 or arrival_rate_rps <= 0:
        raise ValueError("request_count and arrival_rate_rps must be positive")
    if not profiles:
        raise ValueError("profiles must not be empty")
    rng = random.Random(seed)
    arrival_ms = 0.0
    output = []
    for index in range(request_count):
        if index:
            arrival_ms += rng.expovariate(arrival_rate_rps) * 1000.0
        profile = dict(
            profiles[index % len(profiles)]
            if cycle_profiles
            else rng.choice(tuple(profiles))
        )
        profile.update(
            request_id=f"req-{index:06d}",
            arrival_ms=arrival_ms,
        )
        if deadline_ms is not None:
            profile["deadline_ms"] = deadline_ms
        output.append(TraceRequest.from_mapping(profile))
    return tuple(output)


# Backward-compatible import name; new callers should use generate_profile_trace.
generate_variable_trace = generate_profile_trace


def measure_isolated_baseline(
    trace: Sequence[TraceRequest],
    *,
    cost_model: MeasuredStepCostModel,
    transfer_cost_model: MeasuredTransferCostModel | None = None,
    tp_degree: int = 2,
    cp_world_size: int = 4,
    allowed_lane_widths: Sequence[int] = (1, 2, 4),
    pulse_steps: int = 2,
    planner_overhead_ms: float = 0.0,
) -> dict[str, Any]:
    """Measure no-queue static CP-max latency by opaque request profile."""

    if not trace:
        raise ValueError("trace must contain at least one request")
    samples: dict[str, list[float]] = {}
    lane_samples = []
    for request in trace:
        isolated = replace(request, arrival_ms=0.0, deadline_ms=None)
        result = EpeTraceSimulator(
            cost_model=cost_model,
            transfer_cost_model=transfer_cost_model,
            tp_degree=tp_degree,
            cp_world_size=cp_world_size,
            allowed_lane_widths=allowed_lane_widths,
            strategy="static_cp",
            ablation_mode="static-cp-max",
            pulse_steps=pulse_steps,
            planner_overhead_ms=planner_overhead_ms,
        ).run((isolated,))
        samples.setdefault(request.profile_key, []).append(
            result.requests[0].latency_ms
        )
        lane_samples.append(result.requests[0].lane_complete_ms)
    isolated_p95 = {
        key: _percentile(values, 95) for key, values in sorted(samples.items())
    }
    return {
        "isolated_p95_ms": isolated_p95,
        "isolated_samples_ms": {
            key: list(values) for key, values in sorted(samples.items())
        },
        "static_capacity_rps": 1000.0
        / (sum(lane_samples) / len(lane_samples)),
    }


def apply_isolated_p95_deadlines(
    trace: Sequence[TraceRequest],
    isolated_p95_ms: Mapping[str, float],
    *,
    alpha: float,
) -> tuple[TraceRequest, ...]:
    """Set each relative deadline to alpha times its profile's isolated P95."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    missing = sorted(
        {request.profile_key for request in trace} - set(isolated_p95_ms)
    )
    if missing:
        raise ValueError(f"missing isolated P95 profiles: {missing}")
    return tuple(
        replace(
            request,
            deadline_ms=alpha * float(isolated_p95_ms[request.profile_key]),
        )
        for request in trace
    )


def normalize_trace_load(
    trace: Sequence[TraceRequest],
    *,
    normalized_load: float,
    static_capacity_rps: float,
) -> tuple[TraceRequest, ...]:
    """Scale inter-arrivals to a fraction of static CP-max capacity."""

    if not trace:
        raise ValueError("trace must contain at least one request")
    if normalized_load <= 0 or static_capacity_rps <= 0:
        raise ValueError("load and capacity must be positive")
    ordered = sorted(trace, key=lambda request: (request.arrival_ms, request.request_id))
    target_rps = normalized_load * static_capacity_rps
    first = ordered[0].arrival_ms
    span = ordered[-1].arrival_ms - first
    if len(ordered) == 1:
        arrivals = [first]
    elif span <= 1e-9:
        arrivals = [
            first + index * 1000.0 / target_rps
            for index in range(len(ordered))
        ]
    else:
        observed_rps = (len(ordered) - 1) * 1000.0 / span
        scale = observed_rps / target_rps
        arrivals = [
            first + (request.arrival_ms - first) * scale for request in ordered
        ]
    return tuple(
        replace(request, arrival_ms=arrival, deadline_ms=None)
        for request, arrival in zip(ordered, arrivals)
    )


def run_slo_baseline_sweep(
    trace: Sequence[TraceRequest],
    *,
    cost_model: MeasuredStepCostModel,
    transfer_cost_model: MeasuredTransferCostModel | None = None,
    alphas: Sequence[float] = (1.5, 2.0, 3.0, 5.0),
    normalized_loads: Sequence[float] = (0.5, 0.7, 0.9, 1.1, 1.3),
    tp_degree: int = 2,
    cp_world_size: int = 4,
    allowed_lane_widths: Sequence[int] = (1, 2, 4),
    pulse_steps: int = 2,
    switch_allowed_until_step: int = 20,
    balanced_k: bool = True,
    starvation_ms: float = 30_000.0,
    deadline_guard_ms: float = 0.0,
    min_resize_gain_ms: float = 0.0,
    resize_hysteresis_ms: float = 0.0,
    resize_control_cost_ms: float = 0.0,
    max_active_requests: int | None = None,
    planner_overhead_ms: float = 0.0,
) -> dict[str, Any]:
    """Run matched traces across load, SLO alpha, and three ablation modes."""

    if not alphas or not normalized_loads:
        raise ValueError("alphas and normalized_loads must not be empty")
    if min(alphas) <= 0 or min(normalized_loads) <= 0:
        raise ValueError("alphas and normalized_loads must be positive")
    baseline = measure_isolated_baseline(
        trace,
        cost_model=cost_model,
        transfer_cost_model=transfer_cost_model,
        tp_degree=tp_degree,
        cp_world_size=cp_world_size,
        allowed_lane_widths=allowed_lane_widths,
        pulse_steps=pulse_steps,
        planner_overhead_ms=planner_overhead_ms,
    )
    common = {
        "cost_model": cost_model,
        "transfer_cost_model": transfer_cost_model,
        "tp_degree": tp_degree,
        "cp_world_size": cp_world_size,
        "allowed_lane_widths": allowed_lane_widths,
        "pulse_steps": pulse_steps,
        "switch_allowed_until_step": switch_allowed_until_step,
        "balanced_k": balanced_k,
        "starvation_ms": starvation_ms,
        "deadline_guard_ms": deadline_guard_ms,
        "min_resize_gain_ms": min_resize_gain_ms,
        "resize_hysteresis_ms": resize_hysteresis_ms,
        "resize_control_cost_ms": resize_control_cost_ms,
        "max_active_requests": max_active_requests,
        "planner_overhead_ms": planner_overhead_ms,
    }
    scenarios = []
    for normalized_load in normalized_loads:
        loaded = normalize_trace_load(
            trace,
            normalized_load=float(normalized_load),
            static_capacity_rps=float(baseline["static_capacity_rps"]),
        )
        for alpha in alphas:
            matched = apply_isolated_p95_deadlines(
                loaded,
                baseline["isolated_p95_ms"],
                alpha=float(alpha),
            )
            results = {}
            for mode in (
                "static-cp-max",
                "elastic-no-resize",
                "elastic-resize",
            ):
                strategy: ScheduleStrategy = (
                    "static_cp" if mode == "static-cp-max" else "elastic"
                )
                results[mode] = EpeTraceSimulator(
                    strategy=strategy,
                    ablation_mode=mode,
                    **common,
                ).run(matched).to_dict()
            scenarios.append(
                {
                    "normalized_load": float(normalized_load),
                    "offered_load_rps": (
                        float(normalized_load)
                        * float(baseline["static_capacity_rps"])
                    ),
                    "alpha": float(alpha),
                    "results": results,
                }
            )
    return {
        "isolated_baseline": baseline,
        "alphas": [float(value) for value in alphas],
        "normalized_loads": [float(value) for value in normalized_loads],
        "scenarios": scenarios,
    }


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
