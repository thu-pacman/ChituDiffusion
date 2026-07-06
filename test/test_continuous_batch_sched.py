"""CPU-only tests for the M3 continuous-batch scheduler grouping.

Validates that DiffusionScheduler groups same-shape/same-step pending tasks when
continuous_batch is on (capped by max_batch_items), never groups control signals
or flexcache/cp>1 tasks, and falls back to a single decision when off. No GPU or
model needed.

Run: ``python3 test/test_continuous_batch_sched.py``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from chitu_diffusion.runtime.scheduler import DiffusionScheduler  # noqa: E402
from chitu_diffusion.runtime.task import (  # noqa: E402
    DiffusionTask,
    DiffusionTaskPool,
    DiffusionTaskType,
    DiffusionUserParams,
    DiffusionUserRequest,
)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _mk_args(continuous_batch=True, cp_size=1, max_batch_items=8):
    return _Args(
        scheduling_policy="fifo",
        hotswitch_enabled=False,
        switch_allowed_until_step=0,
        continuous_batch=continuous_batch,
        cp_size=cp_size,
        max_batch_items=max_batch_items,
        execution_profiles=[],
    )


def _add_task(task_id, seed=100, size=(1024, 1024), n_sample=1, flexcache=None):
    params = DiffusionUserParams(
        prompt="p", seed=seed, size=size, n_sample=n_sample, num_inference_steps=20, flexcache=flexcache
    )
    req = DiffusionUserRequest(request_id=task_id, params=params)
    task = DiffusionTask(task_id=task_id, task_type=DiffusionTaskType.TextEncode, req=req)
    DiffusionTaskPool.add(task)
    return task


def test_group_same_shape():
    DiffusionTaskPool.reset()
    _add_task("a", seed=1)
    _add_task("b", seed=2)
    _add_task("c", seed=3)
    sched = DiffusionScheduler(_mk_args())
    decisions = sched.schedule_decisions()
    assert [d.task_id for d in decisions] == ["a", "b", "c"], decisions
    assert all(d.reason == "continuous_batch_group" for d in decisions)
    DiffusionTaskPool.reset()


def test_group_excludes_different_shape():
    DiffusionTaskPool.reset()
    _add_task("a", seed=1, size=(1024, 1024))
    _add_task("b", seed=2, size=(512, 512))
    _add_task("c", seed=3, size=(1024, 1024))
    sched = DiffusionScheduler(_mk_args())
    decisions = sched.schedule_decisions()
    # primary 'a' is 1024x1024; only 'c' matches; 'b' is a different shape.
    assert [d.task_id for d in decisions] == ["a", "c"], decisions
    DiffusionTaskPool.reset()


def test_max_batch_items_cap():
    DiffusionTaskPool.reset()
    for i in range(5):
        _add_task(f"t{i}", seed=i)
    sched = DiffusionScheduler(_mk_args(max_batch_items=3))
    decisions = sched.schedule_decisions()
    assert [d.task_id for d in decisions] == ["t0", "t1", "t2"], decisions
    DiffusionTaskPool.reset()


def test_flexcache_not_grouped():
    DiffusionTaskPool.reset()
    _add_task("a", seed=1, flexcache="teacache")
    _add_task("b", seed=2, flexcache="teacache")
    sched = DiffusionScheduler(_mk_args())
    decisions = sched.schedule_decisions()
    # primary is flexcache -> not groupable -> single decision.
    assert [d.task_id for d in decisions] == ["a"], decisions
    DiffusionTaskPool.reset()


def test_cp_size_disables_grouping():
    DiffusionTaskPool.reset()
    _add_task("a", seed=1)
    _add_task("b", seed=2)
    sched = DiffusionScheduler(_mk_args(cp_size=2))
    decisions = sched.schedule_decisions()
    assert [d.task_id for d in decisions] == ["a"], decisions
    DiffusionTaskPool.reset()


def test_off_returns_single():
    DiffusionTaskPool.reset()
    _add_task("a", seed=1)
    _add_task("b", seed=2)
    sched = DiffusionScheduler(_mk_args(continuous_batch=False))
    decisions = sched.schedule_decisions()
    assert [d.task_id for d in decisions] == ["a"], decisions
    DiffusionTaskPool.reset()


# ----------------------------------------------------------------------
# M4 mixed-step continuous batching: dynamic-membership admission planner.
# These exercise DiffusionScheduler.select_cb_admissions / cb_shape_key, the
# stage-independent grouping the engine uses to admit new work into an already
# in-flight (possibly mixed-step) denoise group. No GPU/model needed.
# ----------------------------------------------------------------------


def _mark_inflight(task, size=(1024, 1024), step=0):
    """Simulate a task that already reached Denoise: image_size/seq_len populated
    (so its stage-dependent ``shape_key`` differs from a fresh task's) and sitting
    at an arbitrary current_step."""
    task.task_type = DiffusionTaskType.Denoise
    task.buffer.image_size = size
    task.buffer.seq_len = 1024
    task.buffer.current_step = step
    return task


def test_cb_shape_key_stage_independent():
    DiffusionTaskPool.reset()
    fresh = _add_task("fresh", seed=1, size=(1024, 1024))
    inflight = _add_task("inflight", seed=2, size=(1024, 1024))
    _mark_inflight(inflight, size=(1024, 1024), step=7)
    sched = DiffusionScheduler(_mk_args())
    # Stage-dependent shape_key differs (image_size/seq_len only set once denoising)
    assert fresh.shape_key() != inflight.shape_key()
    # Stage-independent cb_shape_key matches (same request shape/steps/solver)
    assert sched.cb_shape_key(fresh) == sched.cb_shape_key(inflight)
    DiffusionTaskPool.reset()


def test_cb_admissions_admit_all_same_shape():
    DiffusionTaskPool.reset()
    a = _add_task("a", seed=1)
    b = _add_task("b", seed=2)
    c = _add_task("c", seed=3)
    sched = DiffusionScheduler(_mk_args())
    admit = sched.select_cb_admissions([], [a, b, c], capacity=8)
    assert [t.task_id for t in admit] == ["a", "b", "c"], admit
    DiffusionTaskPool.reset()


def test_cb_admissions_capacity():
    DiffusionTaskPool.reset()
    tasks = [_add_task(f"t{i}", seed=i) for i in range(5)]
    sched = DiffusionScheduler(_mk_args())
    admit = sched.select_cb_admissions([], tasks, capacity=2)
    assert [t.task_id for t in admit] == ["t0", "t1"], admit
    # Capacity is max_batch_items - in-flight; zero/negative admits nothing.
    assert sched.select_cb_admissions(tasks[:8], tasks, capacity=0) == []
    DiffusionTaskPool.reset()


def test_cb_admissions_shape_locked_to_inflight():
    DiffusionTaskPool.reset()
    inflight = _mark_inflight(_add_task("inflight", seed=1, size=(1024, 1024)), size=(1024, 1024), step=5)
    b = _add_task("b_512", seed=2, size=(512, 512))
    c = _add_task("c_1024", seed=3, size=(1024, 1024))
    sched = DiffusionScheduler(_mk_args())
    # Even though the 512 task is FIFO-first among pending, the target shape is
    # locked to the in-flight group's shape -> only the matching 1024 is admitted.
    admit = sched.select_cb_admissions([inflight], [b, c], capacity=8)
    assert [t.task_id for t in admit] == ["c_1024"], admit
    DiffusionTaskPool.reset()


def test_cb_admissions_dedup_inflight():
    DiffusionTaskPool.reset()
    a = _mark_inflight(_add_task("a", seed=1), step=3)
    b = _add_task("b", seed=2)
    sched = DiffusionScheduler(_mk_args())
    # 'a' is still Pending (so can_schedule stays alive) but already in-flight;
    # it must not be re-admitted. 'b' (same shape) is admitted at step 0.
    admit = sched.select_cb_admissions([a], [a, b], capacity=8)
    assert [t.task_id for t in admit] == ["b"], admit
    DiffusionTaskPool.reset()


def test_cb_admissions_exclude_flexcache_and_cp():
    DiffusionTaskPool.reset()
    a = _add_task("a", seed=1, flexcache="teacache")
    b = _add_task("b", seed=2)
    sched = DiffusionScheduler(_mk_args())
    admit = sched.select_cb_admissions([], [a, b], capacity=8)
    assert [t.task_id for t in admit] == ["b"], admit  # flexcache excluded
    sched_cp = DiffusionScheduler(_mk_args(cp_size=2))
    assert sched_cp.select_cb_admissions([], [b], capacity=8) == []  # cp>1 excluded
    DiffusionTaskPool.reset()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all continuous-batch scheduler tests passed")
