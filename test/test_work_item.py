"""Unit tests for the M2 work-item abstraction (no GPU/torch model needed).

Verifies that a request with n_sample=N expands into N work-items that share the
parent task's buffer/embeddings by reference, that seeds line up, and that the
pool exposes work-item level counts. Execution numerics are unchanged by M2, so
these are pure data-model checks.

Run: ``python3 test/test_work_item.py``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Avoid importing the heavy runtime backend graph: task.py imports DiffusionBackend,
# which is fine to import (module-level, no CUDA). If torch is unavailable this
# test is skipped by the caller.
from chitu_diffusion.runtime.task import (  # noqa: E402
    DiffusionTask,
    DiffusionTaskType,
    DiffusionUserParams,
    DiffusionUserRequest,
    WorkItem,
)


def _make_task(task_id: str, n_sample: int, seed: int = 100, size=(1024, 1024)) -> DiffusionTask:
    params = DiffusionUserParams(prompt="p", seed=seed, size=size, n_sample=n_sample, num_inference_steps=20)
    req = DiffusionUserRequest(request_id=task_id, params=params)
    return DiffusionTask(task_id=task_id, task_type=DiffusionTaskType.Denoise, req=req)


def test_n_sample_expands_to_work_items():
    task = _make_task("r1", n_sample=4)
    items = task.work_items()
    assert len(items) == 4
    assert [it.sample_index for it in items] == [0, 1, 2, 3]
    assert all(it.n_sample == 4 for it in items)
    assert all(it.request_id == "r1" for it in items)


def test_work_items_share_task_and_buffer_by_reference():
    task = _make_task("r2", n_sample=3)
    items = task.work_items()
    # all work-items point at the SAME task/buffer object (encode-once sharing)
    assert all(it.task is task for it in items)
    task.buffer.current_step = 7
    assert all(it.current_step == 7 for it in items)


def test_seed_alignment_across_samples():
    task = _make_task("r3", n_sample=3, seed=50)
    seeds = task.req.params.sample_seeds(fallback=0)
    assert seeds == [50, 51, 52]  # sample i uses seed base+i (aligns with DP replica s_i)


def test_control_signal_has_no_work_items():
    term = DiffusionTask.create_terminate_signal()
    assert term.n_sample() == 0
    assert term.work_items() == []


def test_pool_work_item_counts():
    from chitu_diffusion.runtime.task import DiffusionTaskPool

    DiffusionTaskPool.reset()
    DiffusionTaskPool.add(_make_task("p1", n_sample=2))
    DiffusionTaskPool.add(_make_task("p2", n_sample=4))
    assert DiffusionTaskPool.work_item_count() == 6
    pending = DiffusionTaskPool.pending_work_items()
    assert len(pending) == 6
    assert isinstance(pending[0], WorkItem)
    # order follows arrival (id_list): p1's 2 then p2's 4
    assert [wi.request_id for wi in pending] == ["p1", "p1", "p2", "p2", "p2", "p2"]
    DiffusionTaskPool.reset()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all work_item tests passed")
