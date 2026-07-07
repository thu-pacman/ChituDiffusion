"""Regression tests for the SLO-aware elastic scheduler (``slo_elastic``).

Torch-free. Covers the must-test scenarios from
``experiments/cp_dp_hot_switch/slo_elastic_scheduler_strategy.md`` section 12:
DP-keep on SP-negative shapes, no-regression vs M7 elastic on mixed/realistic,
the homogeneous-saturated DP fix, head-of-line fairness, the FlexCache emergency
gate (fires under impossible SLO, silent otherwise), switch-window legality, and
the GPU capacity invariant.
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
    RequestSpec,
    SimulationConfig,
    StepLatencyProfile,
    SwitchCostModel,
    simulate_pool,
)

# SP-positive "big" shape (CP/SP speeds it up, sub-linearly) and SP-negative
# "small" shape (any SP degree is slower than plain DP).
BIG = StepLatencyProfile("big", [1000.0] * 20, [560.0] * 20, [330.0] * 20, seq_len=4096, resolution="1024x1024")
SMALL = StepLatencyProfile("small", [100.0] * 20, [120.0] * 20, [150.0] * 20, seq_len=1024, resolution="512x512")
PROFILES = {"big": BIG, "small": SMALL}


def _fastest_solo_ms(profile: StepLatencyProfile, steps: int) -> float:
    return steps * min(profile.step_ms(m, 0) for m in ("dp1", "cp2", "cp4"))


def _spec(rid: str, arrival: float, steps: int, prof: str, slo_factor: float | None = 4.0) -> RequestSpec:
    deadline = None
    if slo_factor is not None:
        deadline = arrival + slo_factor * _fastest_solo_ms(PROFILES[prof], steps)
    return RequestSpec(rid, arrival, steps, prof, deadline_ms=deadline)


def _config(gpus: int = 4, window: int = 8, **kw) -> SimulationConfig:
    return SimulationConfig(
        total_gpus=gpus,
        switch=SwitchCostModel(switch_cost_ms=85.0, switch_allowed_until_step=window),
        **kw,
    )


def _widths(result) -> set[int]:
    return {w for r in result.requests for (_step, w) in r["degree_timeline"]}


# --- scenario 9: GPU capacity invariant + no double-book (all policies) --- #
def test_gpu_capacity_invariant():
    reqs = [_spec("B", 0.0, 20, "big")] + [_spec(f"S{i}", 150.0 + 60 * i, 20, "small") for i in range(5)]
    config = _config()
    result = simulate_pool(reqs, PROFILES, "slo_elastic", config)
    segs = result.segments
    ts = sorted({s.start_ms for s in segs} | {s.end_ms for s in segs})
    for a, b in zip(ts, ts[1:]):
        t = (a + b) / 2.0
        active = [s for s in segs if s.start_ms <= t < s.end_ms]
        used = [g for s in active for g in s.gpus]
        assert len(used) == len(set(used)), (t, used)
        assert len(used) <= config.total_gpus, (t, len(used))


# --- scenario 8: switches only inside the window --- #
def test_switch_only_within_window():
    reqs = [_spec("B", 0.0, 20, "big")] + [_spec(f"S{i}", 150.0 + 60 * i, 20, "small") for i in range(6)]
    window = 6
    result = simulate_pool(reqs, PROFILES, "slo_elastic", _config(window=window))
    for seg in (s for s in result.segments if s.event == "switch"):
        assert seg.step_index <= window, (seg.request_id, seg.step_index)


# --- scenario 1: 512-only (SP-negative) stays on DP, never uses SP --- #
def test_low_res_only_keeps_dp():
    reqs = [_spec(f"S{i}", i * 40.0, 20, "small") for i in range(6)]
    result = simulate_pool(reqs, PROFILES, "slo_elastic", _config())
    assert _widths(result) == {1}, _widths(result)
    assert result.metrics["switch_count"] == 0.0


# --- scenario 4: homogeneous saturated 1024 downshifts to DP concurrency --- #
def test_saturated_homogeneous_fixes_pareto_loss():
    # 10 big requests arriving faster than any single SP4 stream can drain.
    reqs = [_spec(f"B{i}", i * 600.0, 20, "big") for i in range(10)]
    config = _config()
    slo = simulate_pool(reqs, PROFILES, "slo_elastic", config).metrics
    sp4 = simulate_pool(reqs, PROFILES, "static_sp4", config).metrics
    # slo_elastic must recover DP-like throughput and a better tail than pure SP4.
    assert slo["throughput_req_per_s"] >= sp4["throughput_req_per_s"] - 1e-9, (slo, sp4)
    assert slo["p95_latency_ms"] <= sp4["p95_latency_ms"] + 1e-6, (slo, sp4)
    assert slo["max_slowdown"] <= sp4["max_slowdown"] + 1e-6, (slo, sp4)


# --- scenario 2/3: mixed + realistic-ish -> SLO no worse than M7 elastic --- #
def test_mixed_no_worse_slo_than_elastic():
    reqs = [_spec("B", 0.0, 20, "big")] + [_spec(f"S{i}", 200.0 + 60 * i, 20, "small") for i in range(6)]
    config = _config()
    slo = simulate_pool(reqs, PROFILES, "slo_elastic", config).metrics
    elastic = simulate_pool(reqs, PROFILES, "elastic_hot_switch", config).metrics
    assert slo["slo_miss_count"] <= elastic["slo_miss_count"], (slo, elastic)
    # fairness objective: worst slowdown should not regress vs the M7 heuristic.
    assert slo["max_slowdown"] <= elastic["max_slowdown"] + 1e-6, (slo, elastic)


# --- scenario 5: a long SP request must not head-of-line-block short ones --- #
def test_long_does_not_block_short():
    reqs = [_spec("LONG", 0.0, 20, "big")] + [_spec(f"S{i}", 50.0 + 30 * i, 20, "small") for i in range(4)]
    config = _config()
    slo = simulate_pool(reqs, PROFILES, "slo_elastic", config)
    sp4 = simulate_pool(reqs, PROFILES, "static_sp4", config)

    def worst_small_slowdown(res):
        return max(r["slowdown"] for r in res.requests if r["profile"] == "small")

    # Short requests must fare at least as well as under a naive SP4-everything server,
    # i.e. they are not starved behind the long request.
    assert worst_small_slowdown(slo) <= worst_small_slowdown(sp4) + 1e-6


# --- scenario 6: impossible SLO -> FlexCache reduces misses/tardiness --- #
def test_flexcache_rescues_impossible_slo():
    reqs = [_spec(f"B{i}", i * 500.0, 20, "big", slo_factor=1.2) for i in range(10)]
    base = simulate_pool(reqs, PROFILES, "slo_elastic", _config()).metrics
    fc = simulate_pool(
        reqs,
        PROFILES,
        "slo_elastic",
        _config(enable_flexcache=True, flexcache_min_steps=12, flexcache_max_reduce_steps=8),
    ).metrics
    assert fc["flexcache_used_count"] > 0.0, fc
    assert fc["total_tardiness_ms"] <= base["total_tardiness_ms"] + 1e-6, (fc, base)
    assert fc["slo_miss_count"] <= base["slo_miss_count"], (fc, base)


# --- scenario 7: comfortable SLO -> FlexCache stays idle even when enabled --- #
def test_flexcache_idle_when_not_needed():
    reqs = [_spec(f"S{i}", i * 200.0, 20, "small", slo_factor=6.0) for i in range(4)]
    fc = simulate_pool(
        reqs,
        PROFILES,
        "slo_elastic",
        _config(enable_flexcache=True, flexcache_min_steps=12),
    ).metrics
    assert fc["slo_miss_count"] == 0.0, fc
    assert fc["flexcache_used_count"] == 0.0, fc


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all slo_scheduler tests passed")
