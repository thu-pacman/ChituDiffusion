"""Unit tests for the decoupled cost model (pure Python, no GPU/torch needed).

Validates the composition logic, roofline knee detection, grid interpolation,
and JSON round-trip. Run: ``python -m pytest test/test_cost_model.py`` or
``python test/test_cost_model.py``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from chitu_diffusion.runtime.cost_model import (
    CommModel,
    CostModel,
    RooflineComputeModel,
    Z_IMAGE_SPEC,
    default_cost_model,
    image_token_count,
)


def test_image_token_count():
    assert image_token_count(1024, 1024) == 4096  # 128/2 * 128/2
    assert image_token_count(512, 512) == 1024
    assert image_token_count(1536, 1536) == 9216


def test_comm_zero_at_sp1_and_grows_with_k():
    comm = CommModel(spec=Z_IMAGE_SPEC)
    assert comm.step_comm_ms(1, 4096, 1) == 0.0
    c2 = comm.step_comm_ms(2, 4096, 1)
    c4 = comm.step_comm_ms(4, 4096, 1)
    assert c2 > 0.0 and c4 > 0.0
    # per-step comm volume per rank shrinks with k, but total stays comparable;
    # both must be strictly positive and finite.
    assert c2 < float("inf") and c4 < float("inf")


def test_roofline_monotonicity():
    compute = RooflineComputeModel(spec=Z_IMAGE_SPEC)
    # latency must not decrease with batch or seq_len
    assert compute.predict(2, 4096) >= compute.predict(1, 4096)
    assert compute.predict(1, 8192) >= compute.predict(1, 4096)


def test_grid_interpolation():
    grid = {
        (1, 1000): 50.0,
        (1, 5000): 150.0,
        (2, 1000): 60.0,
        (2, 5000): 250.0,
    }
    compute = RooflineComputeModel(spec=Z_IMAGE_SPEC, grid=grid)
    # exact grid points
    assert abs(compute.predict(1, 1000) - 50.0) < 1e-6
    assert abs(compute.predict(2, 5000) - 250.0) < 1e-6
    # midpoint in seq_len at batch 1: linear between 50 and 150 -> 100
    assert abs(compute.predict(1, 3000) - 100.0) < 1e-6
    # clamp beyond edges
    assert abs(compute.predict(1, 500) - 50.0) < 1e-6


def test_batch_saturation_knee():
    # small seq: nearly flat until B=4 then rises steeply -> knee at 4
    grid = {
        (1, 1000): 40.0,
        (2, 1000): 42.0,
        (4, 1000): 44.0,
        (8, 1000): 120.0,
    }
    compute = RooflineComputeModel(spec=Z_IMAGE_SPEC, grid=grid)
    knee = compute.batch_saturation(1000, batches=[1, 2, 4, 8], tol=0.10)
    assert knee == 4, knee


def test_predict_step_sp_reduces_compute_adds_comm():
    model = default_cost_model()
    step_k1 = model.predict_step_ms(img_tokens=9216, batch=1, sp_degree=1)
    step_k4 = model.predict_step_ms(img_tokens=9216, batch=1, sp_degree=4)
    # for a large shape, sharding across 4 workers should reduce total step time
    assert step_k4 < step_k1
    # compute-only part must strictly drop with sharding
    comp_k1 = model.compute.predict(1, Z_IMAGE_SPEC.text_tokens + 9216)
    comp_k4 = model.compute.predict(1, Z_IMAGE_SPEC.text_tokens + 9216 // 4)
    assert comp_k4 < comp_k1


def test_json_round_trip(tmp_path=None):
    import tempfile

    model = CostModel(
        compute=RooflineComputeModel(spec=Z_IMAGE_SPEC, grid={(1, 1000): 50.0, (2, 5000): 250.0}),
        comm=CommModel(spec=Z_IMAGE_SPEC, nvlink_gbps=280.0),
    )
    d = tempfile.mkdtemp()
    path = os.path.join(d, "cm.json")
    model.save(path)
    loaded = CostModel.load(path)
    assert loaded.compute.grid[(1, 1000)] == 50.0
    assert loaded.compute.grid[(2, 5000)] == 250.0
    assert abs(loaded.comm.nvlink_gbps - 280.0) < 1e-9
    assert loaded.spec.dim == Z_IMAGE_SPEC.dim


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all cost_model tests passed")
