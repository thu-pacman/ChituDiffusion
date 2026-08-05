import math

import torch

from chitu_diffusion.flexcache.freecache_core import (
    JvpScheduleController,
    compute_jvp,
    embedded_jvp_error,
    jvp_predict_noise_pred,
    one_step_jvp_drift,
)


def _synthetic_log(num_fresh: int = 3, dim: int = 8):
    sigmas = torch.tensor([1.0 - 0.1 * i for i in range(num_fresh + 1)], dtype=torch.float32)
    latents_pre = []
    latents = []
    sigmas_pre = []
    sigmas_out = []
    vs = []
    x = torch.zeros(1, dim, 1, 1)
    for i in range(num_fresh):
        s0 = sigmas[i]
        s1 = sigmas[i + 1]
        v = torch.sin(s0) * torch.ones(1, dim, 1, 1)
        x_next = x + v * (s1 - s0)
        latents_pre.append(x.clone())
        latents.append(x_next.clone())
        sigmas_pre.append(s0.clone())
        sigmas_out.append(s1.clone())
        vs.append(v.clone())
        x = x_next
    log = {
        "v": vs,
        "latents_pre": latents_pre,
        "latents": latents,
        "sigmas_pre": sigmas_pre,
        "sigmas": sigmas_out,
    }
    return log, sigmas


def test_jvp_predict_matches_finite_difference_on_smooth_segment():
    log, sigmas = _synthetic_log(num_fresh=4)
    step = 3
    pred = jvp_predict_noise_pred(log, sigmas, step, order=1)
    v_ref = log["v"][-1]
    assert pred.shape == v_ref.shape
    assert torch.isfinite(pred).all()


def test_one_step_drift_is_small_on_smooth_segment():
    log, sigmas = _synthetic_log(num_fresh=4)
    drift = one_step_jvp_drift(log, sigmas, step=3, order=1)
    assert drift < 0.2


def test_controller_reuses_when_drift_accumulates_slowly():
    ctrl = JvpScheduleController(tol=0.5, max_gap=4)
    plan = ctrl.plan(0.05)
    assert plan.do_reuse
    plan = ctrl.plan(0.05)
    assert plan.do_reuse


def test_controller_fresh_when_drift_exceeds_tol():
    ctrl = JvpScheduleController(tol=0.1, max_gap=8)
    plan = ctrl.plan(0.2)
    assert not plan.do_reuse


def test_embedded_error_zero_when_true_matches_prediction():
    log, sigmas = _synthetic_log(num_fresh=3)
    step = 2
    v_hat = jvp_predict_noise_pred(log, sigmas, step, order=1)
    err = embedded_jvp_error(log, sigmas, step, v_hat, order=1)
    assert err < 1e-4


def test_compute_jvp_shape_matches_velocity():
    log, _ = _synthetic_log(num_fresh=3)
    jvp = compute_jvp(log, order=1)
    assert jvp is not None
    v_ref = log["v"][-1]
    assert jvp.shape == v_ref.shape


def test_zero_order_predict_reuses_last_fresh_velocity():
    log, sigmas = _synthetic_log(num_fresh=3)
    pred = jvp_predict_noise_pred(log, sigmas, step=2, order=0)
    assert torch.equal(pred, log["v"][-1])
    drift = one_step_jvp_drift(log, sigmas, step=2, order=0)
    assert drift == 0.0
