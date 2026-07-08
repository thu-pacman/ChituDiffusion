#!/usr/bin/env python3
"""Offline simulator for DiT CP/DP hotswitch and batching policies.

The simulator is intentionally dependency-free so it can run before the runtime
prototype exists. It models denoise work at step granularity and compares static
CP, static DP, admission control, early hotswitch, and shape-aware policies.
"""

from __future__ import annotations

import argparse
import copy
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

# The pool-scheduler core (dataclasses + the slo_elastic layout policy) lives in the
# runtime package so the live engine and this simulator share a single implementation.
from chitu_diffusion.runtime.elastic_policy import (  # noqa: E402,F401
    SLO_OBJECTIVE_FIELDS,
    LaneAssignment,
    PoolLayoutPlan,
    PoolRunning,
    PoolSegment,
    RequestSpec,
    SimRequest,
    SimulationConfig,
    StepLatencyProfile,
    SwitchCostModel,
    _largest_valid_width,
    _mode_for_width,
    _pool_start_step,
    _profile_for,
    _refresh_run_leader,
    _run_members,
    _shape_aware_degree,
    _slo_elastic_schedule,
    _solo_service_ms,
    _WIDTH_TO_MODE,
    plan_pool_layout,
    profile_from_cost_model,
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


@dataclass
class DpRunningStep:
    slot_index: int
    req: SimRequest
    step_index: int
    start_ms: float
    end_ms: float
    reason: str


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


def _normalize_pool_policy(policy: str) -> str:
    alias = {"static_cp": "static_sp2", "static_sp": "static_sp2", "sp2": "static_sp2", "sp4": "static_sp4"}
    name = alias.get(policy, policy)
    if name not in POOL_POLICIES:
        raise ValueError(f"unknown pool policy {policy!r}; choose one of {list(POOL_POLICIES)}")
    return name


def _pool_target_degree(req: SimRequest, profiles: dict[str, StepLatencyProfile], policy: str, max_k: int) -> int:
    if policy == "static_dp":
        return 1
    if policy == "static_sp2":
        return min(2, max_k)
    if policy == "static_sp4":
        return min(4, max_k)
    return _shape_aware_degree(_profile_for(req, profiles), max_k=max_k)


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
