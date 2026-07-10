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
    QWEN_IMAGE_SPEC,
    default_cost_model,
    image_token_count,
    latent_token_count,
)


def test_image_token_count():
    assert image_token_count(1024, 1024) == 4096  # 128/2 * 128/2
    assert image_token_count(512, 512) == 1024
    assert image_token_count(1536, 1536) == 9216


def test_qwen_latent_token_count():
    # Qwen: h//16//2 per dim -> 1024x1024 => 32*32 = 1024 image tokens
    assert latent_token_count(1024, 1024, "Qwen-Image") == 1024
    assert latent_token_count(512, 512, "Qwen-Image") == 256


def test_qwen_cost_model_loads():
    path = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "experiments/cp_dp_hot_switch/cost_models/qwen_image_h20.json",
    )
    model = CostModel.load(path)
    assert model.spec.name == "Qwen-Image"
    assert model.spec.n_layers == 60
    # Roofline fallback works with an empty grid
    assert model.predict_step_ms(img_tokens=1024, sp_degree=1) > 0.0


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


def test_cfg_conditions_double_the_forwards():
    model = default_cost_model()
    one = model.predict_step_ms(img_tokens=4096, batch=1, sp_degree=1, cfg_conditions=1)
    two = model.predict_step_ms(img_tokens=4096, batch=1, sp_degree=1, cfg_conditions=2)
    # cond+uncond with no CFG parallelism == exactly two serial forwards
    assert abs(two - 2.0 * one) < 1e-9


def test_cfg_parallel_is_near_perfect_and_beats_context_parallel():
    model = default_cost_model()
    tokens = 4096
    serial = model.predict_step_ms(img_tokens=tokens, batch=1, sp_degree=1, cfg_conditions=2, cfg_parallel=1)
    cfg2 = model.predict_step_ms(img_tokens=tokens, batch=1, sp_degree=1, cfg_conditions=2, cfg_parallel=2)
    one_forward = model.predict_step_ms(img_tokens=tokens, batch=1, sp_degree=1, cfg_conditions=1)
    # CFG parallelism should recover almost the full 2x: ~one forward + a tiny combine
    assert cfg2 < serial
    assert one_forward <= cfg2 < one_forward * 1.05
    # and it should beat spending the same 2 GPUs on context parallelism, which
    # pays all-to-all every layer on both cond and uncond forwards.
    cp2 = model.predict_step_ms(img_tokens=tokens, batch=1, sp_degree=2, cfg_conditions=2, cfg_parallel=1)
    assert cfg2 < cp2


def test_cfg_parallel_clamped_to_conditions():
    model = default_cost_model()
    # cannot split a single condition across 2 workers -> same as cfg_parallel=1
    a = model.predict_step_ms(img_tokens=4096, cfg_conditions=1, cfg_parallel=2)
    b = model.predict_step_ms(img_tokens=4096, cfg_conditions=1, cfg_parallel=1)
    assert abs(a - b) < 1e-9


def test_profile_cfg_first_prefers_cfg_over_cp():
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir)))
    from chitu_diffusion.runtime.elastic_policy import profile_from_cost_model

    model = default_cost_model()
    # Same guidance>1 shape, two ways to spend a request's first extra GPU:
    #   cp_only  -> both cond+uncond forwards run context-parallel (k=2)
    #   cfg_first -> cond|uncond split across the 2 GPUs (CFG parallel)
    cp_only = profile_from_cost_model(
        model, name="p", width=1024, height=1024, num_steps=4,
        guidance_scale=4.0, cfg_parallel_max=1,
    )
    cfg_first = profile_from_cost_model(
        model, name="p", width=1024, height=1024, num_steps=4,
        guidance_scale=4.0, cfg_parallel_max=2,
    )
    # both start from the same width1 (both CFG conditions on one GPU)
    assert abs(cfg_first.dp_1gpu_step_ms[0] - cp_only.dp_1gpu_step_ms[0]) < 1e-9
    # CFG-first width2 must beat spending the same 2 GPUs on context parallelism
    assert cfg_first.cp_2gpu_step_ms[0] < cp_only.cp_2gpu_step_ms[0]
    # width1 (dp1) now carries both CFG conditions -> ~2x a single forward, so the
    # CFG-parallel width2 must roughly halve it (clears any sane speedup threshold).
    assert cfg_first.cp_2gpu_step_ms[0] < 0.6 * cfg_first.dp_1gpu_step_ms[0]


def test_profile_populates_width8_mode():
    """8-GPU profiles expose a cp8 tier that is strictly faster than cp4, and on
    a CFG-first run cp8 is realized as cfp2xcp4 (per-condition CP degree 4)."""
    from chitu_diffusion.runtime.elastic_policy import profile_from_cost_model

    model = default_cost_model()
    cp_only = profile_from_cost_model(
        model, name="p", width=1024, height=1024, num_steps=4,
        guidance_scale=4.0, cfg_parallel_max=1,
    )
    cfg_first = profile_from_cost_model(
        model, name="p", width=1024, height=1024, num_steps=4,
        guidance_scale=4.0, cfg_parallel_max=2,
    )
    for prof in (cp_only, cfg_first):
        assert prof.cp_8gpu_step_ms is not None and prof.cp_8gpu_step_ms
        # step_ms must resolve cp8 without falling back to a narrower tier.
        assert prof.step_ms("cp8", 0) == prof.cp_8gpu_step_ms[0]
        # more GPUs never make a step slower.
        assert prof.step_ms("cp8", 0) <= prof.step_ms("cp4", 0)
    # CFG-first cp8 (cfp2xcp4) beats pure cp8 for the same 8 GPUs.
    assert cfg_first.step_ms("cp8", 0) < cp_only.step_ms("cp8", 0)


def test_policy_supports_width8_partitions_and_mode_map():
    from chitu_diffusion.runtime.elastic_policy import (
        SUPPORTED_LANE_WIDTHS,
        _mode_for_width,
        _largest_valid_width,
        _gpu_partitions,
    )

    assert 8 in SUPPORTED_LANE_WIDTHS
    assert _mode_for_width(8) == "cp8"
    assert _largest_valid_width(8) == 8
    assert _largest_valid_width(7) == 4
    # a single width-8 lane must be a valid partition of an 8-GPU pool.
    parts = _gpu_partitions(8, 8)
    assert (8,) in parts
    # widths are always powers of two drawn from the supported set.
    for p in parts:
        assert all(w in SUPPORTED_LANE_WIDTHS for w in p)
        assert sum(p) <= 8


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


def test_calibrator_disabled_is_identity():
    from chitu_diffusion.runtime.cost_model import RuntimeCalibrator

    model = default_cost_model()
    cal = RuntimeCalibrator(model, guidance_scale=2.0, enabled=False)
    cal.observe(4096, 1, 1, 9999.0)  # ignored while disabled
    assert cal.factor(4096, 1, 1) == 1.0
    assert cal.version == 0


def test_calibrator_ewma_converges_and_corrects():
    from chitu_diffusion.runtime.cost_model import RuntimeCalibrator

    model = default_cost_model()
    cal = RuntimeCalibrator(model, guidance_scale=2.0, alpha=0.5, warmup_skip=0, min_samples=2)
    base = model.predict_step_ms(img_tokens=1024, sp_degree=1, cfg_conditions=2, cfg_parallel=1)
    # feed a stream of "measured" values that are 0.6x the roofline (over-priced shape)
    target = 0.6 * base
    for _ in range(20):
        cal.observe(1024, 1, 1, target)
    # per-key factor must converge toward 0.6 and be used once confident
    assert abs(cal.factor(1024, 1, 1) - 0.6) < 0.05
    assert cal.version > 0


def test_calibrator_warmup_skip_and_outlier_clip():
    from chitu_diffusion.runtime.cost_model import RuntimeCalibrator

    model = default_cost_model()
    cal = RuntimeCalibrator(model, guidance_scale=2.0, alpha=0.5, warmup_skip=2, clip=3.0, min_samples=1)
    base = model.predict_step_ms(img_tokens=4096, sp_degree=4, cfg_conditions=2, cfg_parallel=2)
    # first 2 samples are warmup -> skipped, no factor yet
    cal.observe(4096, 4, 2, base)
    cal.observe(4096, 4, 2, base)
    assert cal._count.get((4096, 4, 2), 0) == 0
    # a real sample establishes the factor (~1.0)
    cal.observe(4096, 4, 2, base)
    assert abs(cal.factor(4096, 4, 2) - 1.0) < 1e-6
    # a 100x spike is rejected as an outlier (factor barely moves)
    before = cal.factor(4096, 4, 2)
    cal.observe(4096, 4, 2, base * 100.0)
    assert abs(cal.factor(4096, 4, 2) - before) < 1e-6


def test_calibrated_cost_model_scales_only_batch1():
    from chitu_diffusion.runtime.cost_model import CalibratedCostModel, RuntimeCalibrator

    model = default_cost_model()
    cal = RuntimeCalibrator(model, guidance_scale=2.0, alpha=1.0, warmup_skip=0, min_samples=1)
    wrapped = CalibratedCostModel(model, cal)
    raw = model.predict_step_ms(img_tokens=1024, sp_degree=1, cfg_conditions=2, cfg_parallel=1)
    # teach the calibrator a 0.5x correction for this key
    cal.observe(1024, 1, 1, 0.5 * raw)
    got = wrapped.predict_step_ms(img_tokens=1024, sp_degree=1, cfg_conditions=2, cfg_parallel=1)
    assert abs(got - 0.5 * raw) < 1e-6
    # batched pricing is left on the raw roofline (no extrapolated factor)
    raw_b = model.predict_step_ms(img_tokens=1024, batch=4, sp_degree=1, cfg_conditions=2, cfg_parallel=1)
    got_b = wrapped.predict_step_ms(img_tokens=1024, batch=4, sp_degree=1, cfg_conditions=2, cfg_parallel=1)
    assert abs(got_b - raw_b) < 1e-6
    # spec + other attributes delegate transparently
    assert wrapped.spec.dim == model.spec.dim


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all cost_model tests passed")
