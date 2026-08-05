"""Regression tests for the N-GPU pool simulator + elastic CP/DP hot-switch policy.

Torch-free. Guards the pool event-model invariants that matter for policy
comparisons, and asserts the elastic hot-switch policy is adaptively no worse
than every static extreme on a bursty high-resolution scenario.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SIM_DIR = os.path.join(ROOT, "experiments", "cp_dp_hot_switch")
for path in (ROOT, SIM_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from simulate import (  # noqa: E402
    POOL_POLICIES,
    RequestSpec,
    SimulationConfig,
    StepLatencyProfile,
    SwitchCostModel,
    simulate_pool,
)

POLICIES = list(POOL_POLICIES)


def _profiles() -> dict[str, StepLatencyProfile]:
    return {
        "long": StepLatencyProfile(
            name="long",
            dp_1gpu_step_ms=[500.0] * 20,
            cp_2gpu_step_ms=[280.0] * 20,
            cp_4gpu_step_ms=[170.0] * 20,
            seq_len=4096,
            resolution="1024x1024",
        ),
        "short": StepLatencyProfile(
            name="short",
            dp_1gpu_step_ms=[120.0] * 20,
            cp_2gpu_step_ms=[130.0] * 20,  # SP is net-negative for the short shape
            cp_4gpu_step_ms=[135.0] * 20,
            seq_len=1024,
            resolution="512x512",
        ),
    }


def _bursty_requests() -> list[RequestSpec]:
    reqs = [RequestSpec("L0", 0.0, 20, "long")]
    for i in range(3):
        reqs.append(RequestSpec(f"S{i}", 200.0 + i * 60.0, 20, "short"))
    return reqs


def _config(gpus: int = 4, window: int = 8) -> SimulationConfig:
    return SimulationConfig(
        total_gpus=gpus,
        switch=SwitchCostModel(switch_cost_ms=85.0, switch_allowed_until_step=window),
    )


def _event_times(segments) -> list[float]:
    ts = sorted({s.start_ms for s in segments} | {s.end_ms for s in segments})
    # sample just inside each interval to test "active at t"
    mids = []
    for a, b in zip(ts, ts[1:]):
        mids.append((a + b) / 2.0)
    return mids


def test_pool_capacity_and_no_double_book():
    """sum(active widths) <= N and no GPU index is used twice at any instant."""
    profiles = _profiles()
    requests = _bursty_requests()
    config = _config()
    for policy in POLICIES:
        result = simulate_pool(requests, profiles, policy, config)
        assert result.metrics["num_requests"] == float(len(requests)), policy
        assert result.metrics["gpu_busy_ratio"] <= 1.0 + 1e-9, policy
        segments = result.segments
        for t in _event_times(segments):
            active = [s for s in segments if s.start_ms <= t < s.end_ms]
            used = [g for s in active for g in s.gpus]
            assert len(used) == len(set(used)), (policy, t, used)  # no double-book
            assert len(used) <= config.total_gpus, (policy, t, len(used))


def test_no_request_runs_two_steps_at_once():
    profiles = _profiles()
    requests = _bursty_requests()
    config = _config()
    for policy in POLICIES:
        result = simulate_pool(requests, profiles, policy, config)
        by_req: dict[str, list[tuple[float, float]]] = {}
        for seg in result.segments:
            if seg.event != "step":
                continue
            by_req.setdefault(seg.request_id, []).append((seg.start_ms, seg.end_ms))
        for rid, spans in by_req.items():
            spans.sort()
            for prev, cur in zip(spans, spans[1:]):
                assert cur[0] >= prev[1] - 1e-9, (policy, rid, prev, cur)


def test_switch_only_within_window():
    profiles = _profiles()
    requests = _bursty_requests()
    window = 8
    config = _config(window=window)
    # elastic_hot_switch must respect the switch window; oracle is exempt by design.
    result = simulate_pool(requests, profiles, "elastic_hot_switch", config)
    switch_segs = [s for s in result.segments if s.event == "switch"]
    assert switch_segs, "expected at least one hot switch in the bursty scenario"
    for seg in switch_segs:
        assert seg.step_index <= window, (seg.request_id, seg.step_index)


def test_elastic_beats_every_static_extreme():
    """Adaptive elastic should be no worse than each fixed static policy, and here strictly better."""
    profiles = _profiles()
    requests = _bursty_requests()
    config = _config()
    means = {p: simulate_pool(requests, profiles, p, config).metrics["mean_latency_ms"] for p in POLICIES}
    best_static = min(means["static_dp"], means["static_sp2"], means["static_sp4"])
    assert means["elastic_hot_switch"] <= best_static + 1e-6, means
    # On this crafted bursty high-res scenario the win should be real, not a tie.
    assert means["elastic_hot_switch"] < best_static, means
    assert simulate_pool(requests, profiles, "elastic_hot_switch", config).metrics["switch_count"] >= 1.0


def test_low_res_only_never_switches():
    """When only the SP-negative short shape is present, elastic == static_dp (no switch)."""
    profiles = _profiles()
    requests = [RequestSpec(f"S{i}", i * 40.0, 20, "short") for i in range(6)]
    config = _config()
    elastic = simulate_pool(requests, profiles, "elastic_hot_switch", config).metrics
    dp = simulate_pool(requests, profiles, "static_dp", config).metrics
    assert elastic["switch_count"] == 0.0, elastic
    assert abs(elastic["mean_latency_ms"] - dp["mean_latency_ms"]) <= 1e-6, (elastic, dp)


if __name__ == "__main__":
    test_pool_capacity_and_no_double_book()
    test_no_request_runs_two_steps_at_once()
    test_switch_only_within_window()
    test_elastic_beats_every_static_extreme()
    test_low_res_only_never_switches()
    print("all serve_sim tests passed")
