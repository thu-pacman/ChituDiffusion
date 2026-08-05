"""Train-free JVP scheduling helpers for FreeCache.

FreeCache keeps MeanCache's sigma-JVP velocity extrapolation and only replaces
the hard-coded fresh-step table with an online controller driven by:

* Signal A (predictive): relative one-step JVP correction ``||jvp*Δσ|| / ||v||``,
  accumulated like TeaCache's ``acc_error`` until it exceeds ``tol``.
* Signal B (corrective): at each fresh step, embedded error
  ``||v_true - v_hat|| / ||v_true||`` between the real forward and the JVP
  prediction at the same step (logged for analysis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

STEP_LEVEL_CACHE_TYPES = frozenset({"meancache", "freecache", "steptrace", "traceplanner"})
STEP_LEVEL_CACHE_KEY = "step_level_noise_pred"


def is_step_level_cache_strategy(strategy) -> bool:
    return getattr(strategy, "type", None) in STEP_LEVEL_CACHE_TYPES


def _rel_tensor_norm(num: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    den = float(ref.detach().to(torch.float32).norm())
    n = float(num.detach().to(torch.float32).norm())
    return n / max(den, eps)


def compute_jvp(
    log_dict: Dict[str, List[Any]],
    order: int,
) -> Optional[torch.Tensor]:
    """MeanCache-style JVP from latent displacement and instantaneous velocity."""
    count = len(log_dict["v"])
    if count == 0:
        return None
    if int(order) <= 0:
        return torch.zeros_like(log_dict["v"][-1])
    actual_steps = min(max(1, int(order)), count)
    i1 = count - 1
    i0 = count - actual_steps
    x_start = log_dict["latents_pre"][i0]
    x_end = log_dict["latents"][i1]
    s_start = log_dict["sigmas_pre"][i0]
    s_end = log_dict["sigmas"][i1]
    denom = (s_end - s_start).to(device=x_start.device, dtype=x_start.dtype)
    if denom.dim() == 0:
        denom_b = denom.reshape(*([1] * x_start.ndim))
    else:
        denom_b = denom.view(*denom.shape, *([1] * (x_start.ndim - denom.dim())))
    avg_u = (x_end - x_start) / denom_b
    inst_u = log_dict["v"][i0]
    return (inst_u - avg_u) / denom_b


def jvp_predict_noise_pred(
    log_dict: Dict[str, List[Any]],
    sigmas: torch.Tensor,
    step: int,
    *,
    order: int = 1,
) -> torch.Tensor:
    """One-step sigma-JVP extrapolation (same formula as MeanCache)."""
    if not log_dict["v"]:
        raise RuntimeError("JVP predict requires at least one fresh velocity.")
    v_last = log_dict["v"][-1]
    if int(order) <= 0:
        return v_last.clone()
    jvp = compute_jvp(log_dict, order)
    if jvp is None or step + 1 >= len(sigmas):
        return v_last.clone()
    denom = sigmas[step + 1].to(device=jvp.device, dtype=jvp.dtype) - sigmas[step].to(
        device=jvp.device, dtype=jvp.dtype
    )
    return v_last.to(device=jvp.device, dtype=jvp.dtype) - jvp.to(jvp.dtype) * denom


def one_step_jvp_drift(
    log_dict: Dict[str, List[Any]],
    sigmas: torch.Tensor,
    step: int,
    *,
    order: int = 1,
) -> float:
    """Signal A: relative magnitude of the JVP correction for this step."""
    if not log_dict["v"]:
        return float("inf")
    v_last = log_dict["v"][-1]
    v_hat = jvp_predict_noise_pred(log_dict, sigmas, step, order=order)
    return _rel_tensor_norm(v_hat - v_last, v_last)


def embedded_jvp_error(
    log_dict: Dict[str, List[Any]],
    sigmas: torch.Tensor,
    step: int,
    v_true: torch.Tensor,
    *,
    order: int = 1,
) -> float:
    """Signal B: true vs JVP-predicted velocity at a fresh step."""
    v_hat = jvp_predict_noise_pred(log_dict, sigmas, step, order=order)
    return _rel_tensor_norm(v_true.to(torch.float32) - v_hat.to(torch.float32), v_true)


@dataclass
class JvpStepPlan:
    do_reuse: bool
    pred_drift: float
    accumulated_drift: float
    embedded_error: Optional[float] = None


class JvpScheduleController:
    """TeaCache-style accumulator on JVP drift (signal A)."""

    def __init__(self, tol: float = 0.15, max_gap: int = 8):
        self.tol = float(tol)
        self.max_gap = max(1, int(max_gap))
        self.accumulated_drift = 0.0
        self.steps_since_fresh = 0
        self.last_embedded_error: Optional[float] = None
        self.reset()

    def reset(self) -> None:
        self.accumulated_drift = 0.0
        self.steps_since_fresh = 0
        self.last_embedded_error = None

    def plan(self, step_drift: float, *, consensus_drift: Optional[float] = None) -> JvpStepPlan:
        drift = float(consensus_drift if consensus_drift is not None else step_drift)
        candidate = self.accumulated_drift + drift
        do_reuse = candidate <= self.tol and self.steps_since_fresh < self.max_gap
        if do_reuse:
            self.accumulated_drift = candidate
            self.steps_since_fresh += 1
        return JvpStepPlan(
            do_reuse=do_reuse,
            pred_drift=drift,
            accumulated_drift=candidate if do_reuse else self.accumulated_drift,
        )

    def commit_fresh(self, embedded_error: Optional[float] = None) -> None:
        self.last_embedded_error = embedded_error
        self.accumulated_drift = 0.0
        self.steps_since_fresh = 0
