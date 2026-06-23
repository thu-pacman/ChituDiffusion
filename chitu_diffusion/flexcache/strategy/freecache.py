from __future__ import annotations

import json
import os
from logging import getLogger
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.flexcache.freecache_core import (
    STEP_LEVEL_CACHE_KEY,
    JvpScheduleController,
    JvpStepPlan,
    embedded_jvp_error,
    jvp_predict_noise_pred,
    one_step_jvp_drift,
)
from chitu_diffusion.runtime.backend import DiffusionBackend

logger = getLogger(__name__)


class FreeCacheStrategy(FlexCacheStrategy):
    """Train-free step-level cache: MeanCache JVP extrapolation + adaptive fresh steps.

    Reuse prediction matches MeanCache (sigma-JVP with latent-averaged curvature).
    Fresh-step scheduling replaces magic tables with online JVP-drift accumulation.
    """

    def __init__(
        self,
        task,
        tol: float = 0.15,
        max_gap: int = 8,
        jvp_order: int = 1,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
    ):
        super().__init__()
        self.type = "freecache"
        self.total_steps = int(task.req.params.num_inference_steps)
        self.tol = float(tol)
        self.max_gap = max(1, int(max_gap))
        self.jvp_order = max(1, int(jvp_order))
        self.warmup_steps = int(warmup_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.tradeoff_score = self.tol
        self.controller = JvpScheduleController(tol=self.tol, max_gap=self.max_gap)
        self.log_dict: Dict[str, List[Any]] = {
            "v": [],
            "latents_pre": [],
            "latents": [],
            "sigmas_pre": [],
            "sigmas": [],
        }
        self._step_plans: Dict[int, JvpStepPlan] = {}
        self.step_trace: List[Dict[str, Any]] = []

    def get_reuse_key(self, **kwargs) -> Optional[str]:
        step = int(kwargs["step"])
        plan = self._plan_for_step(step)
        if plan.do_reuse:
            self._record_step_trace(step, plan, decision="reuse")
            return STEP_LEVEL_CACHE_KEY
        return None

    def reuse(self, **kwargs) -> torch.Tensor:
        return self.predict_noise_pred(int(kwargs["step"]))

    def get_store_key(self, **kwargs) -> Optional[str]:
        return None

    def store(self, **kwargs) -> None:
        step = int(kwargs["step"])
        sigmas = self._scheduler_sigmas()
        embedded = None
        if sigmas is not None and self.log_dict["v"]:
            embedded = embedded_jvp_error(
                self.log_dict,
                sigmas,
                step,
                kwargs["noise_pred"],
                order=self.jvp_order,
            )
            embedded = self._consensus_scalar(embedded)
        self.controller.commit_fresh(embedded)
        plan = self._step_plans.get(step)
        if plan is None:
            plan = JvpStepPlan(do_reuse=False, pred_drift=0.0, accumulated_drift=0.0, embedded_error=embedded)
        else:
            plan = JvpStepPlan(
                do_reuse=False,
                pred_drift=plan.pred_drift,
                accumulated_drift=0.0,
                embedded_error=embedded,
            )
        self._record_step_trace(step, plan, decision="compute")
        self.record_fresh_step(
            step=step,
            latents_pre=kwargs["latents_pre"],
            latents=kwargs["latents"],
            sigma_pre=kwargs["sigma_pre"],
            sigma=kwargs["sigma"],
            noise_pred=kwargs["noise_pred"],
        )
        return None

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        logger.info(
            "FreeCache step strategy enabled: tol=%.4f max_gap=%d jvp_order=%d warmup=%d cooldown=%d",
            self.tol,
            self.max_gap,
            self.jvp_order,
            self.warmup_steps,
            self.cooldown_steps,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            self._save_step_trace(run_output_dir)
        logger.info("FreeCache step strategy disabled.")

    def reset_state(self) -> None:
        for values in self.log_dict.values():
            values.clear()
        self._step_plans.clear()
        self.step_trace.clear()
        self.controller.reset()
        DiffusionBackend.flexcache.clear_cache()

    def predict_noise_pred(self, step: int) -> torch.Tensor:
        sigmas = self._scheduler_sigmas()
        if sigmas is None:
            return self.log_dict["v"][-1].clone()
        return jvp_predict_noise_pred(self.log_dict, sigmas, step, order=self.jvp_order)

    def record_fresh_step(
        self,
        *,
        step: int,
        latents_pre: torch.Tensor,
        latents: torch.Tensor,
        sigma_pre: torch.Tensor,
        sigma: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> None:
        self.log_dict["latents_pre"].append(latents_pre.detach().to(torch.float32))
        self.log_dict["latents"].append(latents.detach().to(torch.float32))
        self.log_dict["sigmas_pre"].append(sigma_pre.detach().to(torch.float32))
        self.log_dict["sigmas"].append(sigma.detach().to(torch.float32))
        self.log_dict["v"].append(noise_pred.detach().to(torch.float32))

    def _plan_for_step(self, step: int) -> JvpStepPlan:
        if step in self._step_plans:
            return self._step_plans[step]

        if step < self.warmup_steps or step >= self.total_steps - self.cooldown_steps:
            plan = JvpStepPlan(do_reuse=False, pred_drift=float("inf"), accumulated_drift=0.0)
            self._step_plans[step] = plan
            return plan

        if not self.log_dict["v"]:
            plan = JvpStepPlan(do_reuse=False, pred_drift=float("inf"), accumulated_drift=0.0)
            self._step_plans[step] = plan
            return plan

        sigmas = self._scheduler_sigmas()
        if sigmas is None or step + 1 >= len(sigmas):
            plan = JvpStepPlan(do_reuse=False, pred_drift=float("inf"), accumulated_drift=0.0)
            self._step_plans[step] = plan
            return plan

        local_drift = one_step_jvp_drift(self.log_dict, sigmas, step, order=self.jvp_order)
        consensus_drift = self._consensus_scalar(local_drift)
        plan = self.controller.plan(local_drift, consensus_drift=consensus_drift)
        self._step_plans[step] = plan
        return plan

    def _scheduler_sigmas(self) -> Optional[torch.Tensor]:
        task = DiffusionBackend.generator.current_task
        sampler = getattr(task.buffer, "sampler", None)
        if sampler is not None:
            sigmas = getattr(sampler, "sigmas", None)
            if sigmas is not None:
                return sigmas
        pipe = getattr(task.buffer, "pipe", None)
        if pipe is not None:
            return getattr(getattr(pipe, "scheduler", None), "sigmas", None)
        return None

    def _consensus_scalar(self, value: float) -> float:
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return value
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([value], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def _record_step_trace(self, step: int, plan: JvpStepPlan, *, decision: str) -> None:
        self.step_trace.append(
            {
                "step": int(step),
                "decision": decision,
                "pred_drift": float(plan.pred_drift),
                "accumulated_drift": float(plan.accumulated_drift),
                "embedded_error": None if plan.embedded_error is None else float(plan.embedded_error),
                "tol": self.tol,
                "steps_since_fresh": int(self.controller.steps_since_fresh),
                "fresh_count": len(self.log_dict["v"]),
                "jvp_order": self.jvp_order,
            }
        )

    def _save_step_trace(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "flexcache_freecache_step_trace.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.step_trace, handle, indent=2)
        logger.info("Saved FreeCache step trace to %s", path)
