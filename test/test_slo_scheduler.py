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
from chitu_diffusion.runtime.elastic_policy import (  # noqa: E402
    PoolRunning,
    SimRequest,
    _slo_barrier_forward,
    _slo_items_for_layout,
    plan_balanced_lane_steps,
    plan_pool_layout,
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


# --- straggler widening: a deep 1024^2 straggler absorbs idle GPUs on the tail --- #
def test_straggler_widen_absorbs_idle_gpus():
    # 1 big (SP-positive) + 3 small (SP-negative) arrive together on 4 GPUs, so the planner
    # runs everything DP width-1 (concurrency). The smalls drain in ~2s; by then big is past
    # the tiny switch window (=1). Without widening it is locked at width 1 while 3 GPUs idle;
    # with widening it must absorb them (small is SP-negative, so any width>1 is necessarily
    # the big request).
    reqs = [_spec("BIG", 0.0, 20, "big", slo_factor=None)] + [
        _spec(f"S{i}", 0.0, 20, "small", slo_factor=None) for i in range(3)
    ]
    locked = simulate_pool(reqs, PROFILES, "slo_elastic", _config(window=1)).metrics
    widen = simulate_pool(
        reqs, PROFILES, "slo_elastic", _config(window=1, straggler_widen=True)
    )
    # The big straggler must reach a width > 1 once the smalls free their GPUs.
    assert max(_widths(widen)) > 1, _widths(widen)
    # ...and finishing it wider must cut the makespan/tail vs the locked baseline.
    assert widen.metrics["makespan_ms"] < locked["makespan_ms"] - 1e-6, (
        widen.metrics["makespan_ms"], locked["makespan_ms"],
    )


def test_straggler_widen_off_keeps_deep_straggler_locked():
    reqs = [_spec("BIG", 0.0, 20, "big", slo_factor=None)] + [
        _spec(f"S{i}", 0.0, 20, "small", slo_factor=None) for i in range(3)
    ]
    # Default (flag off): the deep straggler stays width 1 -> only SP-negative-safe width 1.
    assert _widths(simulate_pool(reqs, PROFILES, "slo_elastic", _config(window=1))) == {1}


# --- P1 #0: barrier-aware forward sees the near-retirement spin --- #
def _sim_req(rid: str, prof: str, step: int, steps: int = 20) -> SimRequest:
    r = SimRequest(spec=RequestSpec(rid, 0.0, steps, prof, deadline_ms=None))
    r.current_step = step
    return r


def _w1_layout(*runs: PoolRunning) -> dict:
    return {
        "assign": [
            {"kind": "boundary", "run": r, "req": r.req, "prev": r.width, "width": r.width}
            for r in runs
        ],
        "beyond": [],
        "switch_count": 0,
    }


def test_barrier_forward_sees_near_retirement_spin():
    # A fast small lane (100ms/step) with only 2 steps left retires almost immediately, then
    # idles at the world barrier while a slow big lane (1000ms/step) runs the full phase_k=5.
    # Hand-computed: phase budget = 5 * 1000 = 5000ms; small runs min(round(5000/100),2)=2
    # steps (wall 200ms), big runs 5 (wall 5000ms) -> phase_wall 5000 -> small spins 4800ms.
    cfg = SimulationConfig(total_gpus=2, phase_steps=5, balanced_k=True, barrier_aware=True)
    big = PoolRunning(req=_sim_req("BIG", "big", 0), width=1, gpus=[0], step_end_ms=None)
    small = PoolRunning(req=_sim_req("SMALL", "small", 18), width=1, gpus=[1], step_end_ms=None)
    layout = _w1_layout(big, small)
    active, waiting = _slo_items_for_layout(0.0, [], layout, [], PROFILES, cfg)
    comps, spin = _slo_barrier_forward(0.0, active, waiting, 2, PROFILES, cfg)
    assert abs(spin - 4800.0) < 1e-6, spin
    assert abs(comps["SMALL"][1] - 200.0) < 1e-6, comps["SMALL"]
    assert abs(comps["BIG"][1] - 20000.0) < 1e-6, comps["BIG"]
    # Latency-balanced, equal lanes -> no spin.
    b2 = PoolRunning(req=_sim_req("B2", "big", 0), width=1, gpus=[1], step_end_ms=None)
    _, spin_bal = _slo_barrier_forward(
        0.0, *_slo_items_for_layout(0.0, [], _w1_layout(big, b2), [], PROFILES, cfg), 2, PROFILES, cfg
    )
    assert spin_bal == 0.0, spin_bal


def test_planner_emits_latency_balanced_lane_steps():
    assert plan_balanced_lane_steps(
        [1000.0, 100.0],
        [20, 20],
        base_steps=5,
        enabled=True,
    ) == [5, 20]
    cfg = SimulationConfig(
        total_gpus=2,
        phase_steps=5,
        balanced_k=True,
        barrier_aware=True,
    )
    big = PoolRunning(req=_sim_req("BIG", "big", 0), width=1, gpus=[0], step_end_ms=None)
    small = PoolRunning(req=_sim_req("SMALL", "small", 18), width=1, gpus=[1], step_end_ms=None)
    plan = plan_pool_layout([], [big, small], [], PROFILES, cfg, now_ms=0.0)
    planned = {lane.request_id: lane.planned_steps for lane in plan.assignments}
    # The near-retirement lane clamps the heartbeat to two slow-lane steps so its
    # GPU can be reallocated at the nearest boundary instead of waiting for K=5.
    assert planned == {"BIG": 2, "SMALL": 2}, planned


# --- P1 #1: no GPU idles while the queue is non-empty (hole-fill) --- #
def test_hole_fill_no_idle_gpu_when_queue_nonempty():
    # 1 in-flight big lane + 5 queued smalls on 4 GPUs: every card must run something.
    big = PoolRunning(req=_sim_req("BIG", "big", 2), width=1, gpus=[0], step_end_ms=None)
    pend = [_sim_req(f"S{i}", "small", 0) for i in range(5)]
    plan = plan_pool_layout([], [big], pend, PROFILES, _config(gpus=4), now_ms=0.0)
    used = sum(a.width for a in plan.assignments)
    assert used == 4, (used, [(a.request_id, a.width) for a in plan.assignments])
    assert plan.idle_gpus == [], plan.idle_gpus


# --- P1 #2: equal-width in-flight lanes keep their width AND GPU block (no swap) --- #
def test_equal_width_lanes_do_not_swap():
    runs = [
        PoolRunning(req=_sim_req(f"S{i}", "small", 3), width=1, gpus=[i], step_end_ms=None)
        for i in range(3)
    ]
    plan = plan_pool_layout([], runs, [], PROFILES, _config(gpus=4), now_ms=0.0)
    assert all(a.width == 1 for a in plan.assignments), [(a.request_id, a.width) for a in plan.assignments]
    assert not any(a.switched for a in plan.assignments), [(a.request_id, a.switched) for a in plan.assignments]
    # each request stays pinned to its original single GPU (block affinity).
    by_id = {a.request_id: a.gpus for a in plan.assignments}
    assert by_id["S0"] == [0] and by_id["S1"] == [1] and by_id["S2"] == [2], by_id


def test_new_admits_do_not_displace_inflight_lanes():
    # In-flight width-1 lanes on GPUs 0,1,2 must keep those cards even when several new
    # pending lanes are admitted the same round to fill the rest of an 8-GPU pool -- a new
    # lane sorting earlier in the plan must not grab a survivor's GPU (needless swap).
    runs = [
        PoolRunning(req=_sim_req("A", "big", 5), width=1, gpus=[0], step_end_ms=None),
        PoolRunning(req=_sim_req("B", "small", 3), width=1, gpus=[1], step_end_ms=None),
        PoolRunning(req=_sim_req("C", "small", 3), width=1, gpus=[2], step_end_ms=None),
    ]
    pend = [_sim_req(f"P{i}", "big" if i % 2 else "small", 0) for i in range(6)]
    plan = plan_pool_layout([], runs, pend, PROFILES, _config(gpus=8), now_ms=0.0)
    g = {a.request_id: a.gpus for a in plan.assignments}
    assert g["A"] == [0] and g["B"] == [1] and g["C"] == [2], g
    assert sum(a.width for a in plan.assignments) == 8 and plan.idle_gpus == [], (g, plan.idle_gpus)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all slo_scheduler tests passed")
