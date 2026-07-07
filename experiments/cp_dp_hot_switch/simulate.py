#!/usr/bin/env python3
"""Offline simulator for DiT CP/DP hotswitch and batching policies.

The simulator is intentionally dependency-free so it can run before the runtime
prototype exists. It models denoise work at step granularity and compares static
CP, static DP, admission control, early hotswitch, and shape-aware policies.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Literal

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chitu_diffusion.runtime.cost_model import (  # noqa: E402
    CostModel,
    default_cost_model,
    image_token_count,
)

# Profiled cost models produced by profile_worker.py; fall back to analytical.
# ``cost_models/`` holds per-hardware calibrations (e.g. h20.json, rtx4090.json);
# the legacy top-level ``cost_model.json`` is kept as the default for backward compat.
COST_MODEL_PATH = Path(__file__).with_name("cost_model.json")
COST_MODELS_DIR = Path(__file__).with_name("cost_models")


def resolve_cost_model(name_or_path: str | Path | None) -> Path:
    """Resolve a ``--cost-model`` value to a JSON path.

    Accepts an explicit file path, or a bare hardware name that is looked up
    inside ``cost_models/`` (e.g. ``rtx4090`` -> ``cost_models/rtx4090.json``).
    Falls back to the legacy top-level ``cost_model.json`` when nothing is given.
    """
    if name_or_path is None:
        return COST_MODEL_PATH
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    named = COST_MODELS_DIR / f"{name_or_path}.json"
    if named.is_file():
        return named
    named_raw = COST_MODELS_DIR / str(name_or_path)
    if named_raw.is_file():
        return named_raw
    return candidate


PolicyName = Literal[
    "static_dp",
    "static_cp",
    "static_sp",
    "admission_control",
    "early_hot_switch",
    "shape_aware_ratio",
    "oracle",
]


@dataclass(frozen=True)
class StepLatencyProfile:
    """Per-step latency table for a request shape and execution mode."""

    name: str
    dp_1gpu_step_ms: list[float]
    cp_2gpu_step_ms: list[float]
    cp_4gpu_step_ms: list[float] | None = None
    dp_batch_step_ms: dict[int, list[float]] = field(default_factory=dict)
    seq_len: int = 0
    resolution: str = "unknown"

    def step_ms(self, mode: str, step_index: int, *, batch: int = 1) -> float:
        if mode == "dp1":
            values = self.dp_batch_step_ms.get(int(batch), self.dp_1gpu_step_ms) if batch > 1 else self.dp_1gpu_step_ms
        elif mode == "cp2":
            values = self.cp_2gpu_step_ms
        elif mode == "cp4":
            values = self.cp_4gpu_step_ms or self.cp_2gpu_step_ms
        else:
            raise ValueError(f"unknown execution mode {mode!r}")
        if not values:
            raise ValueError(f"profile {self.name!r} has no latency values for mode {mode}")
        return float(values[min(step_index, len(values) - 1)])


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_ms: float
    num_steps: int
    profile: str
    priority: int = 0
    deadline_ms: float | None = None


@dataclass
class SimRequest:
    spec: RequestSpec
    current_step: int = 0
    started_ms: float | None = None
    completed_ms: float | None = None
    current_mode: str | None = None
    switched: bool = False
    switch_count: int = 0
    flexcache_reduced_steps: int = 0
    continuous_batch_count: int = 0
    # ``effective_steps`` is the (possibly FlexCache-reduced) step budget; it starts
    # equal to the request's original ``num_steps`` and only ``slo_elastic`` shrinks it.
    effective_steps: int | None = None
    degree_timeline: list[tuple[int, int]] = field(default_factory=list)  # (step_index, width)

    def __post_init__(self) -> None:
        if self.effective_steps is None:
            self.effective_steps = self.spec.num_steps

    @property
    def original_steps(self) -> int:
        return self.spec.num_steps

    @property
    def remaining_steps(self) -> int:
        return max(0, int(self.effective_steps) - self.current_step)

    @property
    def is_done(self) -> bool:
        return self.current_step >= int(self.effective_steps)


@dataclass
class DpRunningStep:
    slot_index: int
    req: SimRequest
    step_index: int
    start_ms: float
    end_ms: float
    reason: str


@dataclass(frozen=True)
class SwitchCostModel:
    switch_cost_ms: float = 0.0
    graph_recapture_cost_ms: float = 0.0
    cache_migration_cost_ms: float = 0.0
    communicator_cost_ms: float = 0.0
    latent_migration_cost_ms: float = 0.0
    switch_allowed_until_step: int = 4

    @property
    def total_ms(self) -> float:
        return (
            self.switch_cost_ms
            + self.graph_recapture_cost_ms
            + self.cache_migration_cost_ms
            + self.communicator_cost_ms
            + self.latent_migration_cost_ms
        )


@dataclass(frozen=True)
class SimulationConfig:
    total_gpus: int = 2
    admission_wait_ms: float = 50.0
    long_request_seq_len: int = 4096
    switch: SwitchCostModel = field(default_factory=SwitchCostModel)
    # --- slo_elastic knobs (ignored by the other policies) ---
    horizon_events: int = 6  # rolling-horizon depth (step completions) for layout scoring
    starvation_wait_ms: float = 30000.0  # queue wait after which a request counts as starved
    fairness_beta: float = 3.0  # target max slowdown (soft; used only for logging)
    enable_flexcache: bool = False  # allow the simulator-only emergency step-reduction action
    enable_continuous_batch: bool = False  # let pending same-shape DP requests join at step boundaries
    max_batch_items: int = 8  # cap for one continuous-batch DP lane
    flexcache_min_steps: int = 4  # never reduce a request below this many executed steps
    flexcache_max_reduce_steps: int = 8  # max steps a single FlexCache action may drop
    flexcache_emergency_tardiness_ms: float = 1.0  # predicted tardiness above which FlexCache unlocks
    flexcache_emergency_slack_ms: float = 2000.0  # negative-slack magnitude that also unlocks FlexCache
    decision_log: bool = False  # record a per-decision explanation trail


@dataclass
class StepTrace:
    request_id: str
    start_ms: float
    end_ms: float
    mode: str
    step_index: int
    gpus: int
    event: str = "step"
    reason: str = ""


@dataclass
class SimulationResult:
    policy: str
    metrics: dict[str, float]
    requests: list[dict[str, Any]]
    trace: list[StepTrace]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "metrics": self.metrics,
            "requests": self.requests,
            "trace": [asdict(item) for item in self.trace],
        }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _normalize_policy(policy: str) -> PolicyName:
    if policy == "static_sp":
        return "static_cp"
    valid = {
        "static_dp",
        "static_cp",
        "admission_control",
        "early_hot_switch",
        "shape_aware_ratio",
        "oracle",
    }
    if policy not in valid:
        raise ValueError(f"unknown policy {policy!r}; choose one of {sorted(valid)}")
    return policy  # type: ignore[return-value]


def _first_pending(pending: list[SimRequest]) -> SimRequest | None:
    if not pending:
        return None
    pending.sort(key=lambda req: (req.spec.priority, req.spec.arrival_ms, req.spec.request_id))
    return pending.pop(0)


def _profile_for(req: SimRequest, profiles: dict[str, StepLatencyProfile]) -> StepLatencyProfile:
    try:
        return profiles[req.spec.profile]
    except KeyError as exc:
        raise ValueError(f"request {req.spec.request_id!r} references missing profile {req.spec.profile!r}") from exc


def _remaining_work_ms(req: SimRequest, profile: StepLatencyProfile, mode: str) -> float:
    return sum(profile.step_ms(mode, step) for step in range(req.current_step, req.spec.num_steps))


def _should_hot_switch(
    active_cp: SimRequest,
    pending: list[SimRequest],
    profiles: dict[str, StepLatencyProfile],
    config: SimulationConfig,
    *,
    oracle: bool = False,
) -> bool:
    if not pending or active_cp.current_mode != "cp2" or active_cp.switched:
        return False
    if active_cp.current_step > config.switch.switch_allowed_until_step:
        return False
    cp_profile = _profile_for(active_cp, profiles)
    queued = pending[0]
    queued_profile = _profile_for(queued, profiles)
    cp_finish_ms = _remaining_work_ms(active_cp, cp_profile, "cp2")
    dp_parallel_ms = max(
        _remaining_work_ms(active_cp, cp_profile, "dp1"),
        _remaining_work_ms(queued, queued_profile, "dp1"),
    )
    saved_ms = cp_finish_ms + _remaining_work_ms(queued, queued_profile, "dp1") - dp_parallel_ms
    if oracle:
        return saved_ms > config.switch.total_ms
    return saved_ms > max(config.switch.total_ms, 0.05 * cp_finish_ms)


def _dp_has_running(dp_slots: list[DpRunningStep | None]) -> bool:
    return any(slot is not None for slot in dp_slots)


def _harvest_dp_slots(
    now_ms: float,
    dp_slots: list[DpRunningStep | None],
    ready: list[SimRequest],
    completed: list[SimRequest],
) -> None:
    for slot_index, running in enumerate(dp_slots):
        if running is None or running.end_ms > now_ms:
            continue
        req = running.req
        req.current_step = running.step_index + 1
        if req.is_done:
            req.completed_ms = running.end_ms
            completed.append(req)
        else:
            ready.append(req)
        dp_slots[slot_index] = None


def _run_dp_replicas(
    now_ms: float,
    ready: list[SimRequest],
    profiles: dict[str, StepLatencyProfile],
    trace: list[StepTrace],
    total_gpus: int,
    completed: list[SimRequest],
    dp_slots: list[DpRunningStep | None],
    *,
    reason: str,
) -> float:
    slots = max(1, total_gpus)
    if len(dp_slots) != slots:
        raise ValueError(f"dp_slots has {len(dp_slots)} slots, expected {slots}")

    _harvest_dp_slots(now_ms, dp_slots, ready, completed)

    for slot_index in range(slots):
        if dp_slots[slot_index] is not None:
            continue
        req = _first_pending(ready)
        if req is None:
            break
        profile = _profile_for(req, profiles)
        if req.started_ms is None:
            req.started_ms = now_ms
        req.current_mode = "dp1"
        step_index = req.current_step
        step_ms = profile.step_ms("dp1", step_index)
        end_ms = now_ms + step_ms
        trace.append(
            StepTrace(
                request_id=req.spec.request_id,
                start_ms=now_ms,
                end_ms=end_ms,
                mode="dp1",
                step_index=step_index,
                gpus=1,
                reason=reason,
            )
        )
        dp_slots[slot_index] = DpRunningStep(
            slot_index=slot_index,
            req=req,
            step_index=step_index,
            start_ms=now_ms,
            end_ms=end_ms,
            reason=reason,
        )

    end_times = [slot.end_ms for slot in dp_slots if slot is not None]
    if not end_times:
        return now_ms
    return min(end_times)


def _run_cp_step(
    now_ms: float,
    req: SimRequest,
    profiles: dict[str, StepLatencyProfile],
    trace: list[StepTrace],
    *,
    reason: str,
) -> float:
    profile = _profile_for(req, profiles)
    if req.started_ms is None:
        req.started_ms = now_ms
    req.current_mode = "cp2"
    step_ms = profile.step_ms("cp2", req.current_step)
    end_ms = now_ms + step_ms
    trace.append(
        StepTrace(
            request_id=req.spec.request_id,
            start_ms=now_ms,
            end_ms=end_ms,
            mode="cp2",
            step_index=req.current_step,
            gpus=2,
            reason=reason,
        )
    )
    req.current_step += 1
    if req.is_done:
        req.completed_ms = end_ms
    return end_ms


def simulate(
    requests: Iterable[RequestSpec],
    profiles: dict[str, StepLatencyProfile],
    policy: str,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    config = config or SimulationConfig()
    policy_name = _normalize_policy(policy)
    arrivals = [SimRequest(copy.deepcopy(req)) for req in sorted(requests, key=lambda item: item.arrival_ms)]
    pending: list[SimRequest] = []
    completed: list[SimRequest] = []
    trace: list[StepTrace] = []
    now_ms = arrivals[0].spec.arrival_ms if arrivals else 0.0
    active_cp: SimRequest | None = None
    dp_slots: list[DpRunningStep | None] = [None for _ in range(max(1, config.total_gpus))]

    def admit_arrivals(until_ms: float) -> None:
        while arrivals and arrivals[0].spec.arrival_ms <= until_ms:
            pending.append(arrivals.pop(0))

    while arrivals or pending or active_cp is not None or _dp_has_running(dp_slots):
        if not pending and active_cp is None and not _dp_has_running(dp_slots) and arrivals:
            now_ms = max(now_ms, arrivals[0].spec.arrival_ms)
        admit_arrivals(now_ms)

        if active_cp is not None and active_cp.is_done:
            completed.append(active_cp)
            active_cp = None

        if _dp_has_running(dp_slots):
            now_ms = _run_dp_replicas(
                now_ms,
                pending,
                profiles,
                trace,
                config.total_gpus,
                completed,
                dp_slots,
                reason=policy_name,
            )

        elif policy_name == "static_dp":
            if not pending and arrivals:
                continue
            now_ms = _run_dp_replicas(
                now_ms,
                pending,
                profiles,
                trace,
                config.total_gpus,
                completed,
                dp_slots,
                reason="static_dp",
            )

        elif policy_name == "static_cp":
            if active_cp is None:
                active_cp = _first_pending(pending)
                if active_cp is None:
                    continue
            now_ms = _run_cp_step(now_ms, active_cp, profiles, trace, reason="static_cp")

        elif policy_name == "admission_control":
            if active_cp is None and pending:
                oldest = pending[0]
                next_arrival_ms = arrivals[0].spec.arrival_ms if arrivals else math.inf
                wait_deadline = oldest.spec.arrival_ms + config.admission_wait_ms
                if len(pending) == 1 and next_arrival_ms <= wait_deadline:
                    now_ms = next_arrival_ms
                    continue
                if len(pending) >= 2:
                    now_ms = _run_dp_replicas(
                        now_ms,
                        pending,
                        profiles,
                        trace,
                        config.total_gpus,
                        completed,
                        dp_slots,
                        reason="admission_dp",
                    )
                else:
                    active_cp = _first_pending(pending)
            elif active_cp is None:
                continue
            if active_cp is not None:
                now_ms = _run_cp_step(now_ms, active_cp, profiles, trace, reason="admission_cp")

        elif policy_name in {"early_hot_switch", "oracle"}:
            if active_cp is None:
                active_cp = _first_pending(pending)
                if active_cp is None:
                    continue
            if _should_hot_switch(active_cp, pending, profiles, config, oracle=policy_name == "oracle"):
                switch_start = now_ms
                now_ms += config.switch.total_ms
                active_cp.switched = True
                trace.append(
                    StepTrace(
                        request_id=active_cp.spec.request_id,
                        start_ms=switch_start,
                        end_ms=now_ms,
                        mode="cp2_to_dp1",
                        step_index=active_cp.current_step,
                        gpus=config.total_gpus,
                        event="switch",
                        reason=policy_name,
                    )
                )
                pending.insert(0, active_cp)
                active_cp = None
                now_ms = _run_dp_replicas(
                    now_ms,
                    pending,
                    profiles,
                    trace,
                    config.total_gpus,
                    completed,
                    dp_slots,
                    reason=policy_name,
                )
            else:
                now_ms = _run_cp_step(now_ms, active_cp, profiles, trace, reason=policy_name)

        elif policy_name == "shape_aware_ratio":
            if len(pending) >= 2:
                now_ms = _run_dp_replicas(
                    now_ms,
                    pending,
                    profiles,
                    trace,
                    config.total_gpus,
                    completed,
                    dp_slots,
                    reason="shape_dp",
                )
            else:
                if active_cp is None:
                    candidate = _first_pending(pending)
                    if candidate is None:
                        continue
                    profile = _profile_for(candidate, profiles)
                    if profile.seq_len and profile.seq_len < config.long_request_seq_len:
                        pending.insert(0, candidate)
                        now_ms = _run_dp_replicas(
                            now_ms,
                            pending,
                            profiles,
                            trace,
                            config.total_gpus,
                            completed,
                            dp_slots,
                            reason="shape_short_dp",
                        )
                        continue
                    active_cp = candidate
                now_ms = _run_cp_step(now_ms, active_cp, profiles, trace, reason="shape_long_cp")

        admit_arrivals(now_ms)
        if active_cp is not None and active_cp.is_done:
            completed.append(active_cp)
            active_cp = None
        done_from_pending = [req for req in pending if req.is_done]
        if done_from_pending:
            pending[:] = [req for req in pending if not req.is_done]
            completed.extend(done_from_pending)

    completed.sort(key=lambda req: req.spec.request_id)
    latencies = [float(req.completed_ms - req.spec.arrival_ms) for req in completed if req.completed_ms is not None]
    queue_delays = [
        float((req.started_ms if req.started_ms is not None else req.spec.arrival_ms) - req.spec.arrival_ms)
        for req in completed
    ]
    makespan = max((req.completed_ms or 0.0 for req in completed), default=0.0) - min(
        (req.spec.arrival_ms for req in completed), default=0.0
    )
    gpu_busy_ms = sum((item.end_ms - item.start_ms) * item.gpus for item in trace if item.event == "step")
    total_gpu_ms = max(makespan, 0.0) * max(1, config.total_gpus)
    metrics = {
        "num_requests": float(len(completed)),
        "makespan_ms": float(max(makespan, 0.0)),
        "throughput_req_per_s": float(len(completed) * 1000.0 / makespan) if makespan > 0 else 0.0,
        "mean_latency_ms": mean(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "mean_queue_delay_ms": mean(queue_delays) if queue_delays else 0.0,
        "gpu_busy_ratio": gpu_busy_ms / total_gpu_ms if total_gpu_ms > 0 else 0.0,
        "switch_count": float(sum(1 for item in trace if item.event == "switch")),
    }
    return SimulationResult(
        policy=policy_name,
        metrics=metrics,
        requests=[
            {
                "request_id": req.spec.request_id,
                "arrival_ms": req.spec.arrival_ms,
                "started_ms": req.started_ms,
                "completed_ms": req.completed_ms,
                "latency_ms": (req.completed_ms - req.spec.arrival_ms) if req.completed_ms is not None else None,
                "profile": req.spec.profile,
                "steps": req.spec.num_steps,
                "switched": req.switched,
            }
            for req in completed
        ],
        trace=trace,
    )


# --------------------------------------------------------------------------- #
# N-GPU pool engine (variable SP width {1,2,4}) + elastic hot-switch policy
# --------------------------------------------------------------------------- #
# The legacy ``simulate()`` above is a 2-GPU, single-active-CP, cp2-only toy kept
# for the abstract scenario and its regression test. ``simulate_pool()`` below is
# a proper N-GPU pool: every in-flight request holds a *width* k in {1,2,4} (its
# SP degree), the invariant ``sum(active widths) <= N`` holds at all times, and a
# request may change width only at a denoise-step boundary (paying the switch
# cost). It records a GPU-index-aware per-segment trace so timelines can show
# width changes across a hot switch.

POOL_POLICIES = (
    "static_dp",
    "static_sp2",
    "static_sp4",
    "shape_aware",
    "elastic_hot_switch",
    "oracle",
    "slo_elastic",
)

_WIDTH_TO_MODE = {1: "dp1", 2: "cp2", 4: "cp4"}


@dataclass
class PoolSegment:
    """One executed unit on a set of GPUs: a denoise step or a switch."""

    request_id: str
    start_ms: float
    end_ms: float
    step_index: int
    width: int
    gpus: list[int]
    event: str = "step"  # "step" | "switch"
    reason: str = ""
    batch_size: int = 1
    request_ids: list[str] = field(default_factory=list)


@dataclass
class PoolRunning:
    req: SimRequest
    width: int
    gpus: list[int]
    step_end_ms: float | None  # None == between steps (a boundary), needs (re)start
    members: list[SimRequest] = field(default_factory=list)


@dataclass
class PoolResult:
    policy: str
    total_gpus: int
    metrics: dict[str, float]
    requests: list[dict[str, Any]]
    segments: list[PoolSegment]
    decision_log: list[dict[str, Any]] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "total_gpus": self.total_gpus,
            "decision_log": self.decision_log,
            "metrics": self.metrics,
            "requests": self.requests,
            "segments": [asdict(seg) for seg in self.segments],
        }


def _mode_for_width(width: int) -> str:
    return _WIDTH_TO_MODE.get(int(width), "dp1")


def _run_members(run: PoolRunning) -> list[SimRequest]:
    if not run.members:
        run.members = [run.req]
    return run.members


def _refresh_run_leader(run: PoolRunning) -> None:
    members = _run_members(run)
    if members:
        run.req = members[0]


def _largest_valid_width(cap: int, max_width: int = 4) -> int:
    """Largest width in {1,2,4} that is <= min(cap, max_width); 0 if cap < 1."""
    best = 0
    for w in (1, 2, 4):
        if w <= cap and w <= max_width:
            best = w
    return best


def _normalize_pool_policy(policy: str) -> str:
    alias = {"static_cp": "static_sp2", "static_sp": "static_sp2", "sp2": "static_sp2", "sp4": "static_sp4"}
    name = alias.get(policy, policy)
    if name not in POOL_POLICIES:
        raise ValueError(f"unknown pool policy {policy!r}; choose one of {list(POOL_POLICIES)}")
    return name


def _shape_aware_degree(profile: StepLatencyProfile, *, max_k: int = 4, threshold: float = 0.15) -> int:
    """Largest SP degree whose per-step latency beats the next-lower degree by > threshold.

    Uses the profile's per-mode step latency (which itself comes from the cost
    model), so the choice self-tunes per hardware: on a no-NVLink box the comm
    term keeps short/low-res shapes at k=1 automatically.
    """
    k = 1
    for nxt in (2, 4):
        if nxt > max_k:
            break
        cur = profile.step_ms(_mode_for_width(k), 0)
        upgraded = profile.step_ms(_mode_for_width(nxt), 0)
        if upgraded < cur * (1.0 - threshold):
            k = nxt
        else:
            break
    return k


def _pool_target_degree(req: SimRequest, profiles: dict[str, StepLatencyProfile], policy: str, max_k: int) -> int:
    if policy == "static_dp":
        return 1
    if policy == "static_sp2":
        return min(2, max_k)
    if policy == "static_sp4":
        return min(4, max_k)
    return _shape_aware_degree(_profile_for(req, profiles), max_k=max_k)


def _pool_start_step(
    now_ms: float,
    run: PoolRunning,
    profiles: dict[str, StepLatencyProfile],
    segments: list[PoolSegment],
    reason: str,
    *,
    switched: bool,
    switch_total_ms: float,
) -> None:
    """Start (or restart) a denoise step for ``run`` at its current width."""
    req = run.req
    profile = _profile_for(req, profiles)
    members = _run_members(run)
    batch_members = members if run.width == 1 else [req]
    batch_size = len(batch_members)
    request_ids = [member.spec.request_id for member in batch_members]
    start_ms = now_ms
    if switched:
        req.switched = True
        req.switch_count += 1
        segments.append(
            PoolSegment(
                request_id=req.spec.request_id,
                start_ms=now_ms,
                end_ms=now_ms + switch_total_ms,
                step_index=req.current_step,
                width=run.width,
                gpus=list(run.gpus),
                event="switch",
                reason=reason,
                batch_size=1,
                request_ids=[req.spec.request_id],
            )
        )
        start_ms = now_ms + switch_total_ms
    for member in batch_members:
        if member.started_ms is None:
            member.started_ms = now_ms
        member.current_mode = _mode_for_width(run.width)
        member.degree_timeline.append((member.current_step, run.width))
    req.current_mode = _mode_for_width(run.width)
    step_ms = profile.step_ms(req.current_mode, req.current_step, batch=batch_size)
    end_ms = start_ms + step_ms
    segments.append(
        PoolSegment(
            request_id=req.spec.request_id,
            start_ms=start_ms,
            end_ms=end_ms,
            step_index=req.current_step,
            width=run.width,
            gpus=list(run.gpus),
            event="batch_step" if batch_size > 1 else "step",
            reason=reason,
            batch_size=batch_size,
            request_ids=request_ids,
        )
    )
    run.step_end_ms = end_ms


def _pool_schedule(
    now_ms: float,
    policy: str,
    running: list[PoolRunning],
    pending: list[SimRequest],
    profiles: dict[str, StepLatencyProfile],
    config: SimulationConfig,
    segments: list[PoolSegment],
    decision_log: list[dict[str, Any]] | None = None,
) -> None:
    """Assign GPUs and (re)start steps at a decision point (step boundary/arrival)."""
    total = max(1, config.total_gpus)
    max_k = min(4, total)
    switch_total = config.switch.total_ms
    oracle = policy == "oracle"
    eff_window = math.inf if oracle else float(config.switch.switch_allowed_until_step)

    firm = [r for r in running if r.step_end_ms is not None]  # mid-step, GPUs held firmly
    boundary = [r for r in running if r.step_end_ms is None]  # between steps, resizable

    # ---- SLO-aware, lexicographic, rolling-horizon scheduler ---- #
    if policy == "slo_elastic":
        _slo_elastic_schedule(now_ms, running, firm, boundary, pending, profiles, config, segments, decision_log)
        return

    # ---- Static / fixed-shape policies: no dynamic resizing of running work ---- #
    if policy in ("static_dp", "static_sp2", "static_sp4", "shape_aware"):
        for run in boundary:
            _pool_start_step(now_ms, run, profiles, segments, policy, switched=False, switch_total_ms=switch_total)
        used_idx = {i for r in running for i in r.gpus}
        free_idx = [i for i in range(total) if i not in used_idx]
        for req in list(pending):
            if not free_idx:
                break
            desired = _pool_target_degree(req, profiles, policy, max_k)
            if policy.startswith("static_"):
                width = desired if len(free_idx) >= desired else 0
                if width == 0:
                    break  # fixed-width server busy; preserve FIFO head-of-line
            else:  # shape_aware: right-size to what fits, fixed thereafter
                width = _largest_valid_width(min(desired, len(free_idx)), max_width=max_k)
                if width <= 0:
                    break
            gpus = free_idx[:width]
            del free_idx[:width]
            run = PoolRunning(req=req, width=width, gpus=gpus, step_end_ms=None)
            running.append(run)
            pending.remove(req)
            _pool_start_step(now_ms, run, profiles, segments, policy, switched=False, switch_total_ms=switch_total)
        return

    # ---- Elastic / oracle: resize running work at step boundaries ---- #
    # Default is to KEEP every running request at its previous width (no gratuitous
    # switching). We only (a) fill genuinely idle GPUs with pending work, (b) upshift
    # a running request into leftover idle GPUs when beneficial, and (c) downshift a
    # running SP request to admit a blocked pending request -- but only when a pairwise
    # completion-time benefit test says the switch actually helps.
    firm_used = sum(r.width for r in firm)
    flex = [r for r in boundary if oracle or r.req.current_step <= eff_window]  # resizable
    cap = total - firm_used
    free = cap - sum(r.width for r in boundary)  # genuinely idle GPUs
    target: dict[int, int] = {id(r): r.width for r in boundary}
    admitted: list[list[Any]] = []  # [req, width]

    def _shape_deg(req: SimRequest) -> int:
        return _shape_aware_degree(_profile_for(req, profiles), max_k=max_k)

    def _step(req: SimRequest, width: int) -> float:
        return _profile_for(req, profiles).step_ms(_mode_for_width(width), req.current_step)

    def _min_running_remaining() -> float:
        rems: list[float] = []
        for r in firm:
            partial = max(0.0, r.step_end_ms - now_ms)
            rest = max(0, r.req.spec.num_steps - (r.req.current_step + 1)) * _step(r.req, r.width)
            rems.append(partial + rest)
        for r in boundary:
            rems.append(max(0, r.req.remaining_steps) * _step(r.req, target[id(r)]))
        return min(rems) if rems else 0.0

    # (a) admit pending onto idle GPUs (no switch), shape-capped by what fits. Filling
    # idle GPUs is a strict win for utilization; the shape cap keeps SP-negative shapes
    # (e.g. 512 on a no-NVLink box) at width 1 automatically.
    while free > 0 and pending:
        head = pending[0]
        width = _largest_valid_width(min(_shape_deg(head), free), max_width=max_k)
        if width < 1:
            break
        admitted.append([head, width])
        free -= width
        pending.pop(0)

    def _cur_width(kind: str, run: PoolRunning | None, entry: list[Any] | None) -> int:
        return target[id(run)] if kind == "flex" else entry[1]

    # (b) elastic upshift: pour leftover idle GPUs into a running/admitted request when
    # the latency it saves outweighs the switch cost (free for a not-yet-started admit).
    upgradable = [("flex", r, None) for r in flex] + [("new", None, e) for e in admitted]
    while free > 0 and not pending:
        best = None
        best_save = 0.0
        best_cost = 0
        for kind, run, entry in upgradable:
            req = run.req if kind == "flex" else entry[0]
            cur = _cur_width(kind, run, entry)
            nxt = cur * 2
            if nxt not in (2, 4) or nxt > min(max_k, _shape_deg(req)) or (nxt - cur) > free:
                continue
            save = req.remaining_steps * (_step(req, cur) - _step(req, nxt))
            thresh = 0.0 if (kind == "new" or oracle) else switch_total
            if save > thresh and save > best_save:
                best, best_save, best_cost = (kind, run, entry), save, (nxt - cur)
        if best is None:
            break
        kind, run, entry = best
        if kind == "flex":
            target[id(run)] += best_cost
        else:
            entry[1] += best_cost
        free -= best_cost

    # (c) burst-triggered downshift (the hot switch): peel GPUs off a running SP request
    # to admit a blocked pending request, only when the pairwise objective improves.
    while pending:
        head = pending[0]
        deg_head = _shape_deg(head)
        wait = _min_running_remaining()
        best_r = None
        best_gain = 0.0
        for r in flex:
            width = target[id(r)]
            if width <= 1:
                continue
            freed = width // 2
            rem_r = r.req.remaining_steps
            # keep: R finishes at its current width; head waits for the soonest free GPU
            # then runs at its own shape degree.
            obj_keep = rem_r * _step(r.req, width) + wait + head.remaining_steps * _step(head, deg_head)
            # switch: R slows to width/2 and pays the switch; head starts now on freed GPUs.
            w_head = _largest_valid_width(min(deg_head, freed), max_width=max_k)
            obj_switch = (
                switch_total + rem_r * _step(r.req, width // 2)
                + switch_total + head.remaining_steps * _step(head, w_head)
            )
            gain = obj_keep - obj_switch
            if gain > (0.0 if oracle else 1e-9) and gain > best_gain:
                best_r, best_gain = r, gain
        if best_r is None:
            break
        freed = target[id(best_r)] // 2
        target[id(best_r)] //= 2
        free += freed
        w_head = _largest_valid_width(min(deg_head, free), max_width=max_k)
        if w_head < 1:  # safety; undo
            target[id(best_r)] *= 2
            free -= freed
            break
        admitted.append([head, w_head])
        free -= w_head
        pending.pop(0)

    # ---- materialize: build the start plan, assign GPU indices, (re)start steps ----
    to_start: list[dict[str, Any]] = [
        {"run": r, "width": target[id(r)], "prev": r.width, "new": False} for r in boundary
    ]
    for req, width in admitted:
        run = PoolRunning(req=req, width=width, gpus=[], step_end_ms=None)
        running.append(run)
        to_start.append({"run": run, "width": width, "prev": 0, "new": True})

    reserved_idx = {i for r in firm for i in r.gpus}
    free_idx = [i for i in range(total) if i not in reserved_idx]
    for entry in to_start:
        run = entry["run"]
        width = entry["width"]
        chosen: list[int] = []
        if not entry["new"]:
            for i in run.gpus:
                if i in free_idx and len(chosen) < width:
                    chosen.append(i)
                    free_idx.remove(i)
        while len(chosen) < width and free_idx:
            chosen.append(free_idx.pop(0))
        run.gpus = chosen
        run.width = width

    for entry in to_start:
        switched = (not entry["new"]) and (entry["width"] != entry["prev"])
        _pool_start_step(now_ms, entry["run"], profiles, segments, policy, switched=switched, switch_total_ms=switch_total)


# --------------------------------------------------------------------------- #
# slo_elastic: SLO-aware, lexicographic, rolling-horizon scheduler + FlexCache
# --------------------------------------------------------------------------- #
# Lexicographic objective (compared field-by-field, earlier fields dominate). See
# slo_elastic_scheduler_strategy.md section 5.
SLO_OBJECTIVE_FIELDS = (
    "slo_miss_count",
    "max_tardiness_ms",
    "total_tardiness_ms",
    "max_slowdown",
    "starvation_count",
    "flexcache_used_count",
    "flexcache_reduced_steps_total",
    "switch_count",
    "total_switch_cost_ms",
    "total_flow_ms",  # sum of predicted (completion - arrival); minimizing this is latency/throughput-optimal
    "mean_slowdown",
)


def _solo_service_ms(req: SimRequest, profiles: dict[str, StepLatencyProfile]) -> float:
    """Single-GPU DP service time -- the fairness baseline for slowdown (strategy §3)."""
    return req.original_steps * _profile_for(req, profiles).step_ms("dp1", 0)


def _slo_shape_deg(req: SimRequest, profiles: dict[str, StepLatencyProfile], max_k: int) -> int:
    return _shape_aware_degree(_profile_for(req, profiles), max_k=max_k)


def _slo_step_ms(req: SimRequest, width: int, profiles: dict[str, StepLatencyProfile]) -> float:
    return _profile_for(req, profiles).step_ms(_mode_for_width(width), req.current_step)


def _rem_steps(req: SimRequest, override: dict[str, int] | None) -> int:
    eff = override.get(req.spec.request_id, req.effective_steps) if override else req.effective_steps
    return max(0, int(eff) - req.current_step)


def _slo_urgency(req: SimRequest, now_ms: float, profiles: dict[str, StepLatencyProfile], max_k: int,
                 override: dict[str, int] | None = None) -> tuple[int, float]:
    """Ordering key: deadline-bearing requests by smallest slack first, then SRPT."""
    prof = _profile_for(req, profiles)
    best_step = min(prof.step_ms(_mode_for_width(k), 0) for k in (1, 2, 4) if k <= max_k)
    rem_time = _rem_steps(req, override) * best_step
    if req.spec.deadline_ms is None:
        return (1, rem_time)
    return (0, req.spec.deadline_ms - now_ms - rem_time)


def _slo_key_pending(pending: list[SimRequest], now_ms: float, profiles: dict[str, StepLatencyProfile],
                     max_k: int, limit: int = 5) -> list[SimRequest]:
    ranked = sorted(pending, key=lambda r: _slo_urgency(r, now_ms, profiles, max_k))
    return ranked[:limit]


def _slo_admit_continuous_batches(now_ms: float, boundary: list[PoolRunning], pending: list[SimRequest],
                                  profiles: dict[str, StepLatencyProfile], config: SimulationConfig) -> int:
    """Greedily admit urgent same-shape pending requests into DP lanes at step boundaries.

    This models continuous batching as a fine-grained admission action: a request no
    longer has to wait for a whole in-flight request to finish, only for a compatible
    DP lane's next denoise-step boundary. We keep it conservative: same profile,
    width=1, and capped batch size.
    """
    if not config.enable_continuous_batch or config.max_batch_items <= 1:
        return 0
    total_admitted = 0
    max_k = min(4, max(1, config.total_gpus))
    for run in boundary:
        if run.width != 1:
            continue
        members = _run_members(run)
        if len(members) >= config.max_batch_items:
            continue
        profile = run.req.spec.profile
        candidates = [
            req for req in pending
            if req.spec.profile == profile and req.current_step == 0
        ]
        candidates.sort(key=lambda r: _slo_urgency(r, now_ms, profiles, max_k))
        while candidates and len(members) < config.max_batch_items:
            req = candidates.pop(0)
            if req not in pending:
                continue
            cur_batch = len(members)
            new_batch = cur_batch + 1
            prof = _profile_for(req, profiles)
            cur_step = prof.step_ms("dp1", req.current_step, batch=cur_batch)
            new_step = prof.step_ms("dp1", req.current_step, batch=new_batch)
            solo_step = prof.step_ms("dp1", req.current_step, batch=1)
            earliest_lane_free = min(member.remaining_steps for member in members) * cur_step
            before: list[tuple[SimRequest, float]] = [
                (member, now_ms + member.remaining_steps * cur_step)
                for member in members
            ]
            before.append((req, now_ms + earliest_lane_free + req.remaining_steps * solo_step))
            after = [
                (member, now_ms + member.remaining_steps * new_step)
                for member in (*members, req)
            ]

            def slo_score(items: list[tuple[SimRequest, float]]) -> tuple[int, float]:
                miss = 0
                tard = 0.0
                for item_req, comp in items:
                    dl = item_req.spec.deadline_ms
                    if dl is None:
                        continue
                    item_tard = max(0.0, comp - dl)
                    tard += item_tard
                    if item_tard > 1e-9:
                        miss += 1
                return miss, tard

            miss_before, tard_before = slo_score(before)
            miss_after, tard_after = slo_score(after)
            if miss_after > miss_before or tard_after >= tard_before - 1e-9:
                continue
            pending.remove(req)
            req.continuous_batch_count += 1
            members.append(req)
            total_admitted += 1
        run.members = members
        _refresh_run_leader(run)
    return total_admitted


def _gpu_partitions(capacity: int, max_parts: int) -> list[tuple[int, ...]]:
    """All non-increasing width multisets from {1,2,4} with sum <= ``capacity`` and at
    most ``max_parts`` parts (>=1). Sums below capacity are kept so a lone SP-negative
    request can run at width 1 and leave GPUs idle rather than being forced into SP;
    the work-conserving filter in the layout builder discards under-packing when there
    is still pending demand. Small for G<=8, so enumeration is cheap."""
    results: set[tuple[int, ...]] = set()

    def rec(remaining: int, max_w: int, parts: list[int]) -> None:
        if parts:
            results.add(tuple(parts))
        if len(parts) >= max_parts:
            return
        for w in (4, 2, 1):
            if w <= remaining and w <= max_w:
                rec(remaining - w, w, parts + [w])

    if capacity > 0 and max_parts > 0:
        rec(capacity, 4, [])
    return sorted(results, key=lambda t: (-len(t), t))


def _slo_candidate_layouts(now_ms: float, firm: list[PoolRunning], boundary: list[PoolRunning],
                           pending: list[SimRequest], profiles: dict[str, StepLatencyProfile],
                           config: SimulationConfig) -> list[dict[str, Any]]:
    total = max(1, config.total_gpus)
    max_k = min(4, total)
    window = float(config.switch.switch_allowed_until_step)
    # Batched DP lanes stay at width=1 until their current members drain; reshaping a
    # mixed-membership batch into SP would require ragged SP batching, which is out of scope.
    beyond = [r for r in boundary if r.req.current_step > window or len(_run_members(r)) > 1]
    flexb = [r for r in boundary if r not in beyond]  # width is a decision variable
    fixed_used = sum(r.width for r in firm) + sum(r.width for r in beyond)
    reassignable = total - fixed_used

    must = [{"kind": "boundary", "run": r, "req": r.req, "prev": r.width} for r in flexb]
    key_pending = _slo_key_pending(pending, now_ms, profiles, max_k, limit=5)
    kcells = [{"kind": "pending", "run": None, "req": r, "prev": 0} for r in key_pending]
    assignable = must + kcells
    max_parts = max(1, len(assignable))

    layouts: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for parts in _gpu_partitions(reassignable, max_parts):
        num = len(parts)
        if num < len(must):
            continue  # cannot run every in-flight (boundary) request
        # Work-conserving: only leave GPUs idle once every assignable request is placed
        # (otherwise a request would wait while a GPU sits idle).
        if sum(parts) < reassignable and num < len(assignable):
            continue
        chosen = must + kcells[: num - len(must)]
        widths = sorted(parts, reverse=True)
        # Give the widest SP groups to the most SP-friendly / heaviest requests; DP-friendly
        # and short requests naturally fall to width 1.
        order = sorted(
            chosen,
            key=lambda c: (-_slo_shape_deg(c["req"], profiles, max_k), -c["req"].remaining_steps, c["req"].spec.arrival_ms),
        )
        assign = [
            {"kind": c["kind"], "run": c["run"], "req": c["req"], "prev": c["prev"], "width": w}
            for c, w in zip(order, widths)
        ]
        switch_count = sum(1 for a in assign if a["kind"] == "boundary" and a["width"] != a["prev"])
        key = tuple(sorted((a["req"].spec.request_id, a["width"]) for a in assign))
        if key in seen:
            continue
        seen.add(key)
        layouts.append({"assign": assign, "beyond": beyond, "switch_count": switch_count})

    if not layouts:  # e.g. no reassignable capacity: keep flexb where they are
        assign = [{"kind": "boundary", "run": r, "req": r.req, "prev": r.width, "width": r.width} for r in flexb]
        layouts.append({"assign": assign, "beyond": beyond, "switch_count": 0})
    return layouts


def _slo_items_for_layout(now_ms: float, firm: list[PoolRunning], layout: dict[str, Any],
                          pending: list[SimRequest], profiles: dict[str, StepLatencyProfile],
                          config: SimulationConfig, override: dict[str, int] | None = None):
    """Predicted (start, completion, width) for everything running under this layout."""
    switch_total = config.switch.total_ms
    active: list[dict[str, Any]] = []
    chosen: set[str] = set()

    def add_run_members(run: PoolRunning, start: float, first_step_end: float, width: int) -> None:
        members = _run_members(run) if width == 1 else [run.req]
        batch = len(members) if width == 1 else 1
        lane = f"run:{id(run)}"
        for member in members:
            rem_after = max(0, _rem_steps(member, override) - 1)
            step_ms = _profile_for(member, profiles).step_ms(_mode_for_width(width), member.current_step, batch=batch)
            comp = first_step_end + rem_after * step_ms
            active.append({
                "id": member.spec.request_id,
                "req": member,
                "width": width,
                "lane": lane,
                "start": member.started_ms if member.started_ms is not None else start,
                "comp": comp,
            })
            chosen.add(member.spec.request_id)

    for run in firm:
        add_run_members(run, now_ms, run.step_end_ms, run.width)
    for run in layout["beyond"]:
        batch = len(_run_members(run)) if run.width == 1 else 1
        first_step_end = now_ms + _profile_for(run.req, profiles).step_ms(_mode_for_width(run.width), run.req.current_step, batch=batch)
        add_run_members(run, now_ms, first_step_end, run.width)
    for a in layout["assign"]:
        req = a["req"]
        switched = a["kind"] == "boundary" and a["width"] != a["prev"]
        start = now_ms + (switch_total if switched else 0.0)
        comp = start + _rem_steps(req, override) * _slo_step_ms(req, a["width"], profiles)
        active.append({
            "id": req.spec.request_id,
            "req": req,
            "width": a["width"],
            "lane": f"req:{req.spec.request_id}",
            "start": start,
            "comp": comp,
        })
        chosen.add(req.spec.request_id)
    waiting = [r for r in pending if r.spec.request_id not in chosen]
    return active, waiting


def _slo_forward(now_ms: float, active: list[dict[str, Any]], waiting: list[SimRequest], total: int,
                 profiles: dict[str, StepLatencyProfile], config: SimulationConfig,
                 override: dict[str, int] | None = None) -> dict[str, tuple[float, float, int]]:
    """List-schedule the whole remaining queue to completion with a load-aware fallback.

    Fallback: when GPUs free, place the most urgent waiting request; if the queue is at
    least as deep as the free GPUs, place it as DP (width 1) to maximize concurrency
    (this is what makes the objective prefer DP under saturation and SP when idle)."""
    max_k = min(4, total)
    completions: dict[str, tuple[float, float, int]] = {
        it["id"]: (it["start"], it["comp"], it["width"]) for it in active
    }
    lanes: dict[str, tuple[float, int]] = {}
    for it in active:
        lane = str(it.get("lane", it["id"]))
        comp, width = lanes.get(lane, (0.0, int(it["width"])))
        lanes[lane] = (max(comp, float(it["comp"])), width)
    free = total - sum(width for _comp, width in lanes.values())
    heap = [(comp, width) for comp, width in lanes.values()]
    heapq.heapify(heap)
    queue = sorted(waiting, key=lambda r: _slo_urgency(r, now_ms, profiles, max_k, override))
    t = now_ms
    guard = 0
    while queue:
        guard += 1
        if guard > 200000:
            break
        while queue and free > 0:
            width = 1 if len(queue) >= free else _largest_valid_width(min(_slo_shape_deg(queue[0], profiles, max_k), free), max_width=max_k)
            if width < 1:
                break
            req = queue.pop(0)
            comp = t + _rem_steps(req, override) * _slo_step_ms(req, width, profiles)
            completions[req.spec.request_id] = (t, comp, width)
            heapq.heappush(heap, (comp, width))
            free -= width
        if not queue or not heap:
            break
        comp, width = heapq.heappop(heap)
        t = max(t, comp)
        free += width
    return completions


def _slo_objective(now_ms: float, completions: dict[str, tuple[float, float, int]],
                   by_id: dict[str, SimRequest], config: SimulationConfig,
                   profiles: dict[str, StepLatencyProfile], switch_count: int,
                   flex_used: int, flex_steps: int) -> tuple[float, ...]:
    slo_miss = 0
    tardiness: list[float] = []
    slowdowns: list[float] = []
    starvation = 0
    total_flow = 0.0
    for rid, (start, comp, _width) in completions.items():
        req = by_id[rid]
        dl = req.spec.deadline_ms
        tard = max(0.0, comp - dl) if dl is not None else 0.0
        tardiness.append(tard)
        if dl is not None and comp > dl + 1e-9:
            slo_miss += 1
        solo = _solo_service_ms(req, profiles)
        slowdowns.append((comp - req.spec.arrival_ms) / solo if solo > 0 else 0.0)
        if start - req.spec.arrival_ms > config.starvation_wait_ms:
            starvation += 1
        total_flow += comp - req.spec.arrival_ms
    # Quantize the continuous SLO/fairness fields that rank *above* switch_count so that
    # only a *meaningful* predicted improvement (not myopic prediction noise) is worth a
    # switch. Big wins (e.g. saturated 1024: DP cuts tardiness by seconds) survive; sub-
    # bucket differences collapse to a tie and the switch-count field then keeps things put.
    def _q(value: float, bucket: float) -> float:
        return round(value / bucket) * bucket

    return (
        float(slo_miss),
        _q(max(tardiness, default=0.0), 500.0),
        _q(sum(tardiness), 500.0),
        _q(max(slowdowns, default=0.0), 0.25),
        float(starvation),
        float(flex_used),
        float(flex_steps),
        float(switch_count),
        float(switch_count) * config.switch.total_ms,
        total_flow,
        mean(slowdowns) if slowdowns else 0.0,
    )


def _slo_extreme_risk(completions: dict[str, tuple[float, float, int]], by_id: dict[str, SimRequest],
                      config: SimulationConfig) -> bool:
    for rid, (start, comp, _w) in completions.items():
        dl = by_id[rid].spec.deadline_ms
        if dl is None:
            continue
        if comp - dl > config.flexcache_emergency_tardiness_ms:
            return True
        if (dl - comp) < -config.flexcache_emergency_slack_ms:
            return True
    return False


def _slo_flexcache_augment(now_ms: float, firm: list[PoolRunning], layout: dict[str, Any],
                           pending: list[SimRequest], profiles: dict[str, StepLatencyProfile],
                           config: SimulationConfig, by_id: dict[str, SimRequest]):
    """Emergency-only step reduction. Greedily shrink the worst-missing requests by the
    smallest amount that helps, re-scoring each time. Returns (obj, layout, comps, actions)
    or ``None`` if FlexCache cannot improve the objective."""
    override: dict[str, int] = {}
    actions: dict[str, int] = {}

    def score(ovr: dict[str, int]):
        active, waiting = _slo_items_for_layout(now_ms, firm, layout, pending, profiles, config, ovr)
        comps = _slo_forward(now_ms, active, waiting, max(1, config.total_gpus), profiles, config, ovr)
        flex_used = sum(1 for _ in actions)
        flex_steps = sum(actions.values())
        return _slo_objective(now_ms, comps, by_id, config, profiles, layout["switch_count"], flex_used, flex_steps), comps

    best_obj, best_comps = score(override)
    improved = True
    while improved:
        improved = False
        # worst-missing request first
        misses = sorted(
            (
                (rid, comp)
                for rid, (_s, comp, _w) in best_comps.items()
                if by_id[rid].spec.deadline_ms is not None and comp - by_id[rid].spec.deadline_ms > config.flexcache_emergency_tardiness_ms
            ),
            key=lambda x: x[1] - by_id[x[0]].spec.deadline_ms,
            reverse=True,
        )
        for rid, _comp in misses:
            req = by_id[rid]
            floor = max(config.flexcache_min_steps, req.current_step)
            current_eff = override.get(rid, req.effective_steps)
            if current_eff <= floor:
                continue
            best_local = None
            for reduce in range(1, config.flexcache_max_reduce_steps + 1):
                new_eff = max(floor, current_eff - reduce)
                if new_eff >= current_eff:
                    break
                trial = dict(override)
                trial[rid] = new_eff
                trial_actions = dict(actions)
                trial_actions[rid] = trial_actions.get(rid, 0) + (current_eff - new_eff)
                active, waiting = _slo_items_for_layout(now_ms, firm, layout, pending, profiles, config, trial)
                comps = _slo_forward(now_ms, active, waiting, max(1, config.total_gpus), profiles, config, trial)
                obj = _slo_objective(now_ms, comps, by_id, config, profiles, layout["switch_count"],
                                     sum(1 for _ in trial_actions), sum(trial_actions.values()))
                if obj < best_obj:
                    best_local = (obj, comps, new_eff, current_eff - new_eff)
                    break  # smallest reduction that helps this request
            if best_local is not None:
                obj, comps, new_eff, reduced = best_local
                override[rid] = new_eff
                actions[rid] = actions.get(rid, 0) + reduced
                best_obj, best_comps = obj, comps
                improved = True
                break
    if not actions:
        return None
    action_list = [{"req": by_id[rid], "new_effective": override[rid], "reduced": actions[rid]} for rid in actions]
    return best_obj, layout, best_comps, action_list


def _slo_apply_layout(now_ms: float, layout: dict[str, Any], profiles: dict[str, StepLatencyProfile],
                      config: SimulationConfig, segments: list[PoolSegment],
                      running: list[PoolRunning], pending: list[SimRequest]) -> None:
    total = max(1, config.total_gpus)
    switch_total = config.switch.total_ms
    to_start: list[dict[str, Any]] = []
    for run in layout["beyond"]:
        to_start.append({"run": run, "width": run.width, "prev": run.width, "new": False})
    for a in layout["assign"]:
        if a["kind"] == "boundary":
            to_start.append({"run": a["run"], "width": a["width"], "prev": a["prev"], "new": False})
        else:
            run = PoolRunning(req=a["req"], width=a["width"], gpus=[], step_end_ms=None)
            running.append(run)
            if a["req"] in pending:
                pending.remove(a["req"])
            to_start.append({"run": run, "width": a["width"], "prev": 0, "new": True})

    reserved = {i for r in running if r.step_end_ms is not None for i in r.gpus}  # firm hold their GPUs
    free_idx = [i for i in range(total) if i not in reserved]
    for entry in to_start:
        run = entry["run"]
        width = entry["width"]
        chosen: list[int] = []
        if not entry["new"]:
            for i in run.gpus:
                if i in free_idx and len(chosen) < width:
                    chosen.append(i)
                    free_idx.remove(i)
        while len(chosen) < width and free_idx:
            chosen.append(free_idx.pop(0))
        run.gpus = chosen
        run.width = width
    for entry in to_start:
        switched = (not entry["new"]) and (entry["width"] != entry["prev"])
        _pool_start_step(now_ms, entry["run"], profiles, segments, "slo_elastic",
                         switched=switched, switch_total_ms=switch_total)


def _slo_elastic_schedule(now_ms: float, running: list[PoolRunning], firm: list[PoolRunning],
                          boundary: list[PoolRunning], pending: list[SimRequest],
                          profiles: dict[str, StepLatencyProfile], config: SimulationConfig,
                          segments: list[PoolSegment], decision_log: list[dict[str, Any]] | None) -> None:
    total = max(1, config.total_gpus)
    free_capacity = total - sum(r.width for r in firm) - sum(r.width for r in boundary)
    cb_admitted = 0
    if free_capacity <= 0:
        cb_admitted = _slo_admit_continuous_batches(now_ms, boundary, pending, profiles, config)
    layouts = _slo_candidate_layouts(now_ms, firm, boundary, pending, profiles, config)
    by_id = {r.spec.request_id: r for r in (*(member for run in running for member in _run_members(run)), *pending)}

    scored: list[tuple[tuple[float, ...], dict[str, Any], dict[str, tuple[float, float, int]]]] = []
    for layout in layouts:
        active, waiting = _slo_items_for_layout(now_ms, firm, layout, pending, profiles, config)
        comps = _slo_forward(now_ms, active, waiting, total, profiles, config)
        obj = _slo_objective(now_ms, comps, by_id, config, profiles, layout["switch_count"], 0, 0)
        scored.append((obj, layout, comps))
    scored.sort(key=lambda x: x[0])
    best_obj, best_layout, best_comps = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else None

    flex_actions: list[dict[str, Any]] = []
    if config.enable_flexcache and _slo_extreme_risk(best_comps, by_id, config):
        aug = _slo_flexcache_augment(now_ms, firm, best_layout, pending, profiles, config, by_id)
        if aug is not None and aug[0] < best_obj:
            best_obj, best_layout, best_comps, flex_actions = aug

    if decision_log is not None and config.decision_log:
        deciding = None
        if runner_up is not None:
            for idx, (a, b) in enumerate(zip(best_obj, runner_up)):
                if abs(a - b) > 1e-9:
                    deciding = SLO_OBJECTIVE_FIELDS[idx]
                    break
        decision_log.append({
            "now_ms": now_ms,
            "chosen_layout": [(a["req"].spec.request_id, a["width"]) for a in best_layout["assign"]],
            "chosen_objective": list(best_obj),
            "runner_up_objective": list(runner_up) if runner_up is not None else None,
            "deciding_field": deciding,
            "switch_count": best_layout["switch_count"],
            "continuous_batch_admitted": cb_admitted,
            "flexcache_actions": [(a["req"].spec.request_id, a["reduced"]) for a in flex_actions],
        })

    for act in flex_actions:
        req = act["req"]
        req.effective_steps = int(act["new_effective"])
        req.flexcache_reduced_steps += int(act["reduced"])

    _slo_apply_layout(now_ms, best_layout, profiles, config, segments, running, pending)


def _pool_metrics(completed: list[SimRequest], segments: list[PoolSegment], total: int,
                  profiles: dict[str, StepLatencyProfile], config: SimulationConfig):
    """Base throughput/latency metrics plus SLO / fairness / FlexCache metrics, and the
    per-request records. SLO fields are meaningful whenever deadlines are present."""
    latencies = [float(req.completed_ms - req.spec.arrival_ms) for req in completed if req.completed_ms is not None]
    queue_delays = [
        float((req.started_ms if req.started_ms is not None else req.spec.arrival_ms) - req.spec.arrival_ms)
        for req in completed
    ]
    makespan = max((req.completed_ms or 0.0 for req in completed), default=0.0) - min(
        (req.spec.arrival_ms for req in completed), default=0.0
    )
    busy_gpu_ms = sum((seg.end_ms - seg.start_ms) * len(seg.gpus) for seg in segments)
    total_gpu_ms = max(makespan, 0.0) * total

    tardiness: list[float] = []
    slowdowns: list[float] = []
    useful_token_steps = 0.0
    slo_miss = 0
    starvation = 0
    flex_used = 0
    flex_steps = 0
    cb_used = 0
    per_request: list[dict[str, Any]] = []
    for req in completed:
        comp = req.completed_ms if req.completed_ms is not None else req.spec.arrival_ms
        latency = comp - req.spec.arrival_ms
        solo = _solo_service_ms(req, profiles)
        slowdown = latency / solo if solo > 0 else 0.0
        slowdowns.append(slowdown)
        dl = req.spec.deadline_ms
        tard = max(0.0, comp - dl) if dl is not None else 0.0
        tardiness.append(tard)
        if dl is not None and comp > dl + 1e-9:
            slo_miss += 1
        queue_ms = (req.started_ms if req.started_ms is not None else req.spec.arrival_ms) - req.spec.arrival_ms
        if queue_ms > config.starvation_wait_ms:
            starvation += 1
        if req.flexcache_reduced_steps > 0:
            flex_used += 1
            flex_steps += req.flexcache_reduced_steps
        if req.continuous_batch_count > 0:
            cb_used += 1
        profile = profiles[req.spec.profile]
        req_token_steps = float(profile.seq_len * int(req.effective_steps))
        useful_token_steps += req_token_steps
        per_request.append({
            "request_id": req.spec.request_id,
            "arrival_ms": req.spec.arrival_ms,
            "deadline_ms": dl,
            "started_ms": req.started_ms,
            "completed_ms": req.completed_ms,
            "latency_ms": (req.completed_ms - req.spec.arrival_ms) if req.completed_ms is not None else None,
            "queue_ms": queue_ms,
            "tardiness_ms": tard,
            "slowdown": slowdown,
            "profile": req.spec.profile,
            "resolution": profile.resolution,
            "image_tokens": profile.seq_len,
            "original_steps": req.original_steps,
            "executed_steps": int(req.effective_steps),
            "useful_token_steps": req_token_steps,
            "flexcache_reduced_steps": req.flexcache_reduced_steps,
            "continuous_batch_count": req.continuous_batch_count,
            "switch_count": req.switch_count,
            "switched": req.switched,
            "degree_timeline": req.degree_timeline,
        })

    n = len(completed)
    metrics = {
        "num_requests": float(n),
        "total_gpus": float(total),
        "makespan_ms": float(max(makespan, 0.0)),
        "throughput_req_per_s": float(n * 1000.0 / makespan) if makespan > 0 else 0.0,
        "mean_latency_ms": mean(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "mean_queue_delay_ms": mean(queue_delays) if queue_delays else 0.0,
        "p95_queue_delay_ms": _percentile(queue_delays, 0.95),
        "gpu_busy_ratio": busy_gpu_ms / total_gpu_ms if total_gpu_ms > 0 else 0.0,
        "useful_token_steps": useful_token_steps,
        "useful_token_steps_per_gpu_s": useful_token_steps * 1000.0 / total_gpu_ms if total_gpu_ms > 0 else 0.0,
        "busy_token_steps_per_gpu_s": useful_token_steps * 1000.0 / busy_gpu_ms if busy_gpu_ms > 0 else 0.0,
        "switch_count": float(sum(1 for seg in segments if seg.event == "switch")),
        "slo_miss_count": float(slo_miss),
        "slo_miss_rate": float(slo_miss / n) if n else 0.0,
        "max_tardiness_ms": max(tardiness, default=0.0),
        "total_tardiness_ms": sum(tardiness),
        "mean_tardiness_ms": mean(tardiness) if tardiness else 0.0,
        "mean_slowdown": mean(slowdowns) if slowdowns else 0.0,
        "max_slowdown": max(slowdowns, default=0.0),
        "starvation_count": float(starvation),
        "flexcache_used_count": float(flex_used),
        "flexcache_request_fraction": float(flex_used / n) if n else 0.0,
        "flexcache_reduced_steps_total": float(flex_steps),
        "flexcache_reduced_steps_mean": float(flex_steps / flex_used) if flex_used else 0.0,
        "continuous_batch_used_count": float(cb_used),
        "continuous_batch_request_fraction": float(cb_used / n) if n else 0.0,
        "continuous_batch_step_count": float(sum(1 for seg in segments if seg.event == "batch_step")),
    }
    return metrics, per_request


def simulate_pool(
    requests: Iterable[RequestSpec],
    profiles: dict[str, StepLatencyProfile],
    policy: str,
    config: SimulationConfig | None = None,
) -> PoolResult:
    """Event-driven N-GPU pool simulator with variable SP width and hot switching."""
    config = config or SimulationConfig()
    policy_name = _normalize_pool_policy(policy)
    total = max(1, config.total_gpus)

    arrivals = [SimRequest(copy.deepcopy(req)) for req in sorted(requests, key=lambda item: item.arrival_ms)]
    pending: list[SimRequest] = []
    running: list[PoolRunning] = []
    completed: list[SimRequest] = []
    segments: list[PoolSegment] = []
    decision_log: list[dict[str, Any]] = []
    now_ms = arrivals[0].spec.arrival_ms if arrivals else 0.0

    def admit(until_ms: float) -> None:
        moved = False
        while arrivals and arrivals[0].spec.arrival_ms <= until_ms:
            pending.append(arrivals.pop(0))
            moved = True
        if moved:
            pending.sort(key=lambda r: (r.spec.priority, r.spec.arrival_ms, r.spec.request_id))

    guard = 0
    while arrivals or pending or running:
        guard += 1
        if guard > 5_000_000:
            raise RuntimeError("simulate_pool did not converge")
        admit(now_ms)

        # Harvest steps that finished at or before now.
        survivors: list[PoolRunning] = []
        for run in running:
            if run.step_end_ms is not None and run.step_end_ms <= now_ms + 1e-9:
                remaining_members: list[SimRequest] = []
                for member in _run_members(run):
                    member.current_step += 1
                    if member.is_done:
                        member.completed_ms = run.step_end_ms
                        completed.append(member)
                    else:
                        remaining_members.append(member)
                if remaining_members:
                    run.members = remaining_members
                    _refresh_run_leader(run)
                    run.step_end_ms = None  # boundary, awaiting next decision
                    survivors.append(run)
            else:
                survivors.append(run)
        running = survivors

        if not running and not pending:
            if arrivals:
                now_ms = max(now_ms, arrivals[0].spec.arrival_ms)
                continue
            break

        _pool_schedule(now_ms, policy_name, running, pending, profiles, config, segments, decision_log)

        step_ends = [r.step_end_ms for r in running if r.step_end_ms is not None]
        next_end = min(step_ends) if step_ends else math.inf
        next_arrival = arrivals[0].spec.arrival_ms if arrivals else math.inf
        nxt = min(next_end, next_arrival)
        if nxt == math.inf:
            break  # pending remain but nothing running and no arrivals (unreachable guard)
        now_ms = nxt

    completed.sort(key=lambda req: req.spec.request_id)
    metrics, requests_out = _pool_metrics(completed, segments, total, profiles, config)
    return PoolResult(
        policy=policy_name,
        total_gpus=total,
        metrics=metrics,
        requests=requests_out,
        segments=segments,
        decision_log=decision_log,
    )


def load_cost_model(cost_model: str | Path | None = None) -> CostModel:
    """Profiled cost model if available (profile_worker.py), else analytical."""
    path = resolve_cost_model(cost_model)
    if path.is_file():
        return CostModel.load(path)
    return default_cost_model()


def profile_from_cost_model(
    model: CostModel,
    *,
    name: str,
    width: int,
    height: int,
    num_steps: int,
) -> StepLatencyProfile:
    """Build a step-latency profile for one shape by querying the cost model.

    Each execution mode maps to a sequence-parallel degree: dp1 -> k=1 (a whole
    request on one GPU), cp2 -> k=2, cp4 -> k=4. Steps are near-uniform for DiT so
    the per-step value is repeated across ``num_steps``.
    """
    img_tokens = image_token_count(width, height)
    dp1 = model.predict_step_ms(img_tokens=img_tokens, batch=1, sp_degree=1)
    cp2 = model.predict_step_ms(img_tokens=img_tokens, batch=1, sp_degree=2)
    cp4 = model.predict_step_ms(img_tokens=img_tokens, batch=1, sp_degree=4)
    dp_batch = {
        batch: [model.predict_step_ms(img_tokens=img_tokens, batch=batch, sp_degree=1)] * num_steps
        for batch in range(2, 17)
    }
    return StepLatencyProfile(
        name=name,
        seq_len=img_tokens,
        resolution=f"{width}x{height}",
        dp_1gpu_step_ms=[dp1] * num_steps,
        cp_2gpu_step_ms=[cp2] * num_steps,
        cp_4gpu_step_ms=[cp4] * num_steps,
        dp_batch_step_ms=dp_batch,
    )


def default_profiles(cost_model: str | Path | None = None) -> dict[str, StepLatencyProfile]:
    """Derive step latencies from the decoupled cost model (roofline + comm)."""
    model = load_cost_model(cost_model)
    return {
        "short_1024": profile_from_cost_model(model, name="short_1024", width=512, height=512, num_steps=20),
        "long_4096": profile_from_cost_model(model, name="long_4096", width=1024, height=1024, num_steps=30),
    }


def default_requests() -> list[RequestSpec]:
    return [
        RequestSpec("long_a", 0.0, 24, "long_4096"),
        RequestSpec("short_b", 500.0, 20, "short_1024"),
        RequestSpec("short_c", 650.0, 20, "short_1024"),
        RequestSpec("long_d", 2200.0, 24, "long_4096"),
    ]


def load_input(path: Path) -> tuple[list[RequestSpec], dict[str, StepLatencyProfile], SimulationConfig]:
    data = json.loads(path.read_text())
    profiles = {
        item["name"]: StepLatencyProfile(
            name=item["name"],
            dp_1gpu_step_ms=[float(v) for v in item["dp_1gpu_step_ms"]],
            cp_2gpu_step_ms=[float(v) for v in item["cp_2gpu_step_ms"]],
            cp_4gpu_step_ms=[float(v) for v in item["cp_4gpu_step_ms"]] if item.get("cp_4gpu_step_ms") else None,
            dp_batch_step_ms={
                int(k): [float(v) for v in values]
                for k, values in item.get("dp_batch_step_ms", {}).items()
            },
            seq_len=int(item.get("seq_len", 0)),
            resolution=str(item.get("resolution", "unknown")),
        )
        for item in data.get("profiles", [])
    }
    requests = [
        RequestSpec(
            request_id=str(item["request_id"]),
            arrival_ms=float(item["arrival_ms"]),
            num_steps=int(item["num_steps"]),
            profile=str(item["profile"]),
            priority=int(item.get("priority", 0)),
            deadline_ms=float(item["deadline_ms"]) if item.get("deadline_ms") is not None else None,
        )
        for item in data.get("requests", [])
    ]
    switch = SwitchCostModel(**{k: v for k, v in data.get("switch", {}).items() if k in SwitchCostModel.__dataclass_fields__})
    config_data = data.get("config", {})
    config = SimulationConfig(
        total_gpus=int(config_data.get("total_gpus", 2)),
        admission_wait_ms=float(config_data.get("admission_wait_ms", 50.0)),
        long_request_seq_len=int(config_data.get("long_request_seq_len", 4096)),
        switch=switch,
    )
    return requests, profiles, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON file with profiles, requests, switch, and config sections.")
    parser.add_argument(
        "--cost-model",
        default=None,
        help="Cost model to derive default Z-Image profiles from: a hardware name "
        "(e.g. 'rtx4090', resolved under cost_models/) or an explicit JSON path. "
        "Ignored when --input is given. Defaults to the legacy cost_model.json.",
    )
    parser.add_argument(
        "--policy",
        default="all",
        help="Policy to run, or 'all'. Choices: static_dp, static_cp, admission_control, early_hot_switch, shape_aware_ratio, oracle.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON result to this path instead of stdout.")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    if args.input:
        requests, profiles, config = load_input(args.input)
    else:
        requests, profiles, config = default_requests(), default_profiles(args.cost_model), SimulationConfig(
            switch=SwitchCostModel(
                switch_cost_ms=30.0,
                graph_recapture_cost_ms=20.0,
                communicator_cost_ms=25.0,
                latent_migration_cost_ms=10.0,
                switch_allowed_until_step=6,
            )
        )

    policies = (
        ["static_dp", "static_cp", "admission_control", "early_hot_switch", "shape_aware_ratio", "oracle"]
        if args.policy == "all"
        else [args.policy]
    )
    output = {
        "config": {
            "total_gpus": config.total_gpus,
            "admission_wait_ms": config.admission_wait_ms,
            "long_request_seq_len": config.long_request_seq_len,
            "switch": asdict(config.switch),
        },
        "results": [simulate(requests, profiles, policy, config).to_jsonable() for policy in policies],
    }
    text = json.dumps(output, indent=args.indent, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
