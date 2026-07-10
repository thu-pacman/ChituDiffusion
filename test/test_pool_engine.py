#!/usr/bin/env python3
"""Runtime invariants for the slo_elastic concurrent heterogeneous-lane pool engine.

These are pure-logic tests (no GPUs / no torch.distributed): they validate that the
engine maps a policy layout onto *canonical, width-aligned* contiguous GPU blocks --
the precondition the pre-warmed lane subgroups (``set_active_lane_group``) rely on.
The heavy end-to-end 4x RTX 4090 serving run is exercised separately via
``test/test_z_image_serve.py`` + ``test/configs/z_image_serve_pool.yaml``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chitu_diffusion.runtime.generator import Generator
from chitu_diffusion.runtime.scheduler import DiffusionScheduler


def _assert_canonical(blocks, widths, total):
    seen: set[int] = set()
    assert len(blocks) == len(widths)
    for block, width in zip(blocks, widths):
        assert len(block) == width
        offset = min(block)
        # aligned block: offset % width == 0 and contiguous
        assert offset % width == 0, f"non-canonical offset={offset} width={width}"
        assert block == list(range(offset, offset + width))
        assert all(0 <= g < total for g in block)
        assert not (set(block) & seen), "lanes overlap"
        seen.update(block)
    # never over-subscribes the pool
    assert sum(widths) <= total
    assert len(seen) == sum(widths)


@pytest.mark.parametrize(
    "widths",
    [
        [4],
        [2, 2],
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [1, 1, 1, 1],
        [2],
        [1],
        [1, 1],
    ],
)
def test_canonical_placement_is_aligned_and_disjoint(widths):
    total = 4
    blocks = Generator._canonical_placement(list(widths), total)
    # order is preserved w.r.t. the input lanes (blocks[i] belongs to widths[i])
    _assert_canonical(blocks, widths, total)


def test_canonical_placement_rejects_oversubscription():
    # A width-4 lane occupies the whole pool; a second lane cannot fit -> the engine
    # must refuse rather than silently double-book a GPU.
    with pytest.raises(RuntimeError):
        Generator._canonical_placement([4, 1], 4)


def test_canonical_placement_width2_never_lands_on_odd_offset():
    # Regression: greedy front-fill (the policy's index assignment) can place a
    # width-2 lane at offset 1; the engine must re-align it to an even offset.
    blocks = Generator._canonical_placement([1, 2, 1], 4)
    width2 = next(b for b in blocks if len(b) == 2)
    assert min(width2) % 2 == 0


# ---------------------------------------------------------------------------
# Stage 2: placement preservation (affinity). A request that keeps its width is
# pinned to its previous aligned block so no latent has to migrate.
# ---------------------------------------------------------------------------
def test_affinity_pins_requests_to_previous_blocks():
    # Two width-1 lanes that previously sat on GPUs 2 and 0 must stay there
    # (order preserved), not get repacked to 0,1.
    blocks = Generator._canonical_placement([1, 1], 4, prev_blocks=[[2], [0]])
    assert blocks == [[2], [0]]


def test_affinity_pins_width2_block():
    # A width-2 lane previously on [2,3] and a width-1 previously on [0] keep their blocks.
    blocks = Generator._canonical_placement([2, 1], 4, prev_blocks=[[2, 3], [0]])
    assert blocks == [[2, 3], [0]]
    _assert_canonical(blocks, [2, 1], 4)


def test_affinity_falls_back_when_previous_block_invalid():
    # A misaligned previous block (width-2 at odd offset [1,2]) cannot be honored;
    # placement must fall back to a valid aligned packing rather than crash.
    blocks = Generator._canonical_placement([2, 2], 4, prev_blocks=[[1, 2], []])
    _assert_canonical(blocks, [2, 2], 4)


def test_affinity_no_prev_matches_default_packing():
    # With empty prev blocks, affinity mode yields the same layout as no-affinity.
    with_affinity = Generator._canonical_placement([2, 1, 1], 4, prev_blocks=[[], [], []])
    plain = Generator._canonical_placement([2, 1, 1], 4)
    assert with_affinity == plain


def test_affinity_keeps_tail_width1_lanes_regardless_of_lane_order():
    # Regression for the GPU6/7 "swap-type rearrangement" bug: on an 8-GPU node the tail
    # width-1 lanes must stay on their previous cards even when the planner reorders the
    # lane list (there is no comm-locality, so a swap only wastes a group re-warm + latent
    # migration). This is the property the rank-0 full-placement memo restores: as long as
    # the *correct* prev block is supplied per request, order must not induce a swap.
    widths = [2, 1, 1, 1, 1, 1, 1]  # req0003 + six width-1 (req0004,05,07,08, then 10,09)
    prev = [[0, 1], [2], [3], [4], [5], [7], [6]]  # req0010 was on 7, req0009 on 6
    blocks = Generator._canonical_placement(widths, 8, prev_blocks=prev)
    assert blocks == prev  # every request pinned to its own previous block; no 6<->7 swap
    _assert_canonical(blocks, widths, 8)


# ---------------------------------------------------------------------------
# Stage 7: residual runtime cost model. Defaults keep the planner (and the
# offline simulator) byte-for-byte unchanged; configured values add exactly the
# per-step lane tax + amortized per-phase-boundary tax to completion predictions.
# ---------------------------------------------------------------------------
def test_residual_overhead_defaults_to_zero():
    from chitu_diffusion.runtime.elastic_policy import SimulationConfig, _residual_overhead_ms

    cfg = SimulationConfig(total_gpus=4)
    assert _residual_overhead_ms(cfg, 50) == 0.0
    assert _residual_overhead_ms(cfg, 0) == 0.0


def test_residual_overhead_per_step_only():
    from chitu_diffusion.runtime.elastic_policy import SimulationConfig, _residual_overhead_ms

    cfg = SimulationConfig(total_gpus=4, per_step_lane_cost_ms=2.0)
    assert _residual_overhead_ms(cfg, 10) == pytest.approx(20.0)
    # no boundary term unless both boundary cost AND phase length are set
    assert _residual_overhead_ms(cfg, 10) == pytest.approx(20.0)


def test_residual_overhead_amortizes_boundary_over_phase():
    from chitu_diffusion.runtime.elastic_policy import SimulationConfig, _residual_overhead_ms

    cfg = SimulationConfig(
        total_gpus=4, per_step_lane_cost_ms=1.0, phase_boundary_cost_ms=100.0, phase_steps=5
    )
    # 12 remaining steps => ceil(12/5)=3 boundaries; 12*1 + 3*100 = 312
    assert _residual_overhead_ms(cfg, 12) == pytest.approx(312.0)
    # boundary term is inert when phase length is unknown (phase_steps=0)
    cfg0 = SimulationConfig(total_gpus=4, phase_boundary_cost_ms=100.0, phase_steps=0)
    assert _residual_overhead_ms(cfg0, 12) == 0.0


# ---------------------------------------------------------------------------
# Stage 5: thread-safe arrival ingress. A background thread may DiffusionTaskPool.add
# concurrently with the engine reading the pool; the lock must keep pool/id_list
# consistent (no lost adds, no duplicated ids, snapshots never crash mid-mutation).
# ---------------------------------------------------------------------------
def test_pool_concurrent_add_is_thread_safe():
    import threading

    from chitu_diffusion.runtime.task import DiffusionTask, DiffusionTaskPool

    DiffusionTaskPool.reset()
    try:
        n_per_thread = 250
        n_threads = 4

        def _adder(base: int) -> None:
            for i in range(n_per_thread):
                tid = f"t{base}_{i}"
                DiffusionTaskPool.add(DiffusionTask(task_id=tid, req=SimpleNamespace()))

        readers_stop = threading.Event()

        def _reader() -> None:
            # Concurrent snapshot iteration must never raise "changed size during iteration".
            while not readers_stop.is_set():
                _ = DiffusionTaskPool.items_snapshot()
                _ = DiffusionTaskPool.pending_task_ids()

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        adders = [threading.Thread(target=_adder, args=(b,)) for b in range(n_threads)]
        for t in adders:
            t.start()
        for t in adders:
            t.join()
        readers_stop.set()
        reader.join(timeout=2.0)

        expected = n_per_thread * n_threads
        assert len(DiffusionTaskPool.pool) == expected
        assert len(DiffusionTaskPool.id_list) == expected
        # id_list and pool keys agree exactly (no lost / duplicated adds)
        assert set(DiffusionTaskPool.id_list) == set(DiffusionTaskPool.pool.keys())
        assert len(set(DiffusionTaskPool.id_list)) == expected
    finally:
        DiffusionTaskPool.reset()




# ---------------------------------------------------------------------------
# Fixed-layout baseline policies (pure_dp / pure_sp) that reuse the pool engine.
# Pure-logic (no GPUs): only the scheduler's layout decision is exercised.
# ---------------------------------------------------------------------------
class _FakeTask:
    def __init__(self, tid: str, arrival: int):
        self.task_id = tid
        self.arrival_ts = arrival


def _scheduler(policy: str) -> DiffusionScheduler:
    return DiffusionScheduler(SimpleNamespace(scheduling_policy=policy, cp_size=4))


def test_pure_policies_are_pool_policies():
    for policy in ("slo_elastic", "pure_dp", "pure_sp"):
        assert _scheduler(policy).is_pool_policy
    assert not _scheduler("fifo").is_pool_policy


def test_pure_sp_single_lane_full_width_and_sequential():
    s = _scheduler("pure_sp")
    pend = [_FakeTask("r0", 10), _FakeTask("r1", 20), _FakeTask("r2", 5)]
    plan = s.plan_pool_round([], pend, 4)
    # one lane, full width, oldest-arrival request chosen; no concurrency
    assert len(plan.lanes) == 1
    assert plan.lanes[0].width == 4
    assert plan.lanes[0].request_id == "r2"
    # an in-flight request keeps the whole pool until it finishes (strictly serial)
    run = [(_FakeTask("rA", 1), 4, [0, 1, 2, 3], True)]
    plan = s.plan_pool_round(run, pend, 4)
    assert [ln.request_id for ln in plan.lanes] == ["rA"]
    assert plan.lanes[0].width == 4


def test_pure_dp_all_width_one_inflight_first_capped_at_total():
    s = _scheduler("pure_dp")
    run = [(_FakeTask("rA", 1), 1, [0], True), (_FakeTask("rB", 2), 1, [1], True)]
    pend = [_FakeTask("p0", 10), _FakeTask("p1", 20), _FakeTask("p2", 5)]
    plan = s.plan_pool_round(run, pend, 4)
    # every lane width 1, in-flight first, then pending by arrival, capped at 4
    assert all(ln.width == 1 for ln in plan.lanes)
    assert [ln.request_id for ln in plan.lanes] == ["rA", "rB", "p2", "p0"]
    # never over-subscribes even with more pending than GPUs
    many = [_FakeTask(f"q{i}", i) for i in range(6)]
    plan = s.plan_pool_round([], many, 4)
    assert len(plan.lanes) == 4
    assert sum(ln.width for ln in plan.lanes) <= 4


# ---------------------------------------------------------------------------
# Dynamic per-lane CFG (CFP) pair-groups. A CFP-2 lane of width w=2*stride runs its
# cond half on the first ``stride`` ranks and uncond half on the next ``stride``,
# reuniting cond-shard-i with uncond-shard-i. These pure-logic tests validate the
# rank-list layout the runtime relies on (no GPUs / no torch.distributed collectives).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stride,world", [(1, 8), (2, 8), (4, 8), (1, 4), (2, 4), (1, 2)])
def test_cfg_pair_rank_lists_partition_and_align(stride, world):
    from chitu_diffusion.core.distributed.parallel_state import _cfg_pair_rank_lists

    pairs = _cfg_pair_rank_lists(stride, world)
    # Every pair is a cond|uncond couple exactly ``stride`` apart within its 2*stride block.
    for pr in pairs:
        assert len(pr) == 2
        cond, uncond = pr
        assert uncond - cond == stride, f"pair {pr} not stride={stride} apart"
        base = (cond // (2 * stride)) * (2 * stride)
        assert base <= cond < base + stride, f"cond {cond} not in cond-half of block {base}"
        assert base + stride <= uncond < base + 2 * stride
    # The pairs partition the whole world exactly once (required by CommGroup: every
    # rank in exactly one subgroup).
    flat = [r for pr in pairs for r in pr]
    assert sorted(flat) == list(range(world))
    assert len(flat) == len(set(flat))


def test_cfg_pairs_stay_within_a_canonical_lane_block():
    # A CFP-2 lane of width w=2*stride occupies an aligned block [off, off+w). Its
    # stride-``stride`` pairs must fall entirely inside that lane (never crossing into a
    # neighbouring lane), so the per-lane all-gather only involves the lane's own ranks.
    from chitu_diffusion.core.distributed.parallel_state import _cfg_pair_rank_lists

    world = 8
    for stride in (1, 2, 4):
        w = 2 * stride
        pairs = _cfg_pair_rank_lists(stride, world)
        for off in range(0, world, w):  # canonical aligned lane offsets
            lane = set(range(off, off + w))
            in_lane = [pr for pr in pairs if pr[0] in lane]
            # exactly ``stride`` pairs live in this lane and both endpoints are inside it
            assert len(in_lane) == stride
            for cond, uncond in in_lane:
                assert cond in lane and uncond in lane


def test_cfg_strides_match_even_lane_widths():
    # The strides we pre-warm are exactly ``w//2`` for each even lane width, so every
    # even-width lane has a CFP realization and odd/width-1 lanes never do.
    widths = [1, 2, 4, 8]
    strides = sorted({w // 2 for w in widths if w >= 2 and w % 2 == 0})
    assert strides == [1, 2, 4]
