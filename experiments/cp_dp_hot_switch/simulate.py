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

# Profiled cost model produced by profile_worker.py; falls back to analytical.
COST_MODEL_PATH = Path(__file__).with_name("cost_model.json")


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
    seq_len: int = 0
    resolution: str = "unknown"

    def step_ms(self, mode: str, step_index: int) -> float:
        if mode == "dp1":
            values = self.dp_1gpu_step_ms
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

    @property
    def remaining_steps(self) -> int:
        return max(0, self.spec.num_steps - self.current_step)

    @property
    def is_done(self) -> bool:
        return self.current_step >= self.spec.num_steps


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


def load_cost_model() -> CostModel:
    """Profiled cost model if available (profile_worker.py), else analytical."""
    if COST_MODEL_PATH.is_file():
        return CostModel.load(COST_MODEL_PATH)
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
    return StepLatencyProfile(
        name=name,
        seq_len=img_tokens,
        resolution=f"{width}x{height}",
        dp_1gpu_step_ms=[dp1] * num_steps,
        cp_2gpu_step_ms=[cp2] * num_steps,
        cp_4gpu_step_ms=[cp4] * num_steps,
    )


def default_profiles() -> dict[str, StepLatencyProfile]:
    """Derive step latencies from the decoupled cost model (roofline + comm)."""
    model = load_cost_model()
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
        requests, profiles, config = default_requests(), default_profiles(), SimulationConfig(
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
