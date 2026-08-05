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
        anchor_interval: Optional[int] = None,
        anchor_phase: int = 0,
        forced_compute_steps: Optional[List[int]] = None,
        forced_reuse_orders: Optional[Dict[int, int]] = None,
        save_vectors: bool = False,
    ):
        super().__init__()
        self.type = "freecache"
        self.task_id = str(task.task_id)
        self.total_steps = int(task.req.params.num_inference_steps)
        self.tol = float(tol)
        self.max_gap = max(1, int(max_gap))
        self.jvp_order = max(0, int(jvp_order))
        self.warmup_steps = int(warmup_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.anchor_interval = None if anchor_interval is None else max(1, int(anchor_interval))
        self.anchor_phase = int(anchor_phase)
        self.forced_compute_steps = self._normalize_forced_compute_steps(forced_compute_steps)
        self.forced_reuse_orders = self._normalize_forced_reuse_orders(forced_reuse_orders)
        self.save_vectors = bool(save_vectors)
        self.anchor_compute_steps = self._build_anchor_compute_steps()
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
        self.vector_trace: Dict[str, List[Any]] = {
            "guided_v": [],
            "local_v": [],
            "latents_pre": [],
            "latents": [],
            "sigmas_pre": [],
            "sigmas": [],
            "steps": [],
            "decisions": [],
            "reuse_orders": [],
        }

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
        self.record_vector_step(
            step=step,
            decision="compute",
            latents_pre=kwargs["latents_pre"],
            latents=kwargs["latents"],
            sigma_pre=kwargs["sigma_pre"],
            sigma=kwargs["sigma"],
            noise_pred=kwargs.get("guided_noise_pred", kwargs["noise_pred"]),
            local_noise_pred=kwargs["noise_pred"],
            reuse_order=None,
        )
        return None

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        logger.info(
            "FreeCache step strategy enabled: tol=%.4f max_gap=%d jvp_order=%d warmup=%d cooldown=%d anchor_interval=%s anchor_phase=%d forced_compute=%s",
            self.tol,
            self.max_gap,
            self.jvp_order,
            self.warmup_steps,
            self.cooldown_steps,
            self.anchor_interval,
            self.anchor_phase,
            None if self.forced_compute_steps is None else len(self.forced_compute_steps),
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            self._save_step_trace(run_output_dir)
        logger.info("FreeCache step strategy disabled.")

    def reset_state(self) -> None:
        for values in self.log_dict.values():
            values.clear()
        for values in self.vector_trace.values():
            values.clear()
        self._step_plans.clear()
        self.step_trace.clear()
        self.controller.reset()
        DiffusionBackend.flexcache.clear_cache()

    def predict_noise_pred(self, step: int) -> torch.Tensor:
        sigmas = self._scheduler_sigmas()
        if sigmas is None:
            return self.log_dict["v"][-1].clone()
        return jvp_predict_noise_pred(self.log_dict, sigmas, step, order=self._order_for_step(step))

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

    def record_reuse_step(
        self,
        *,
        step: int,
        latents_pre: torch.Tensor,
        latents: torch.Tensor,
        sigma_pre: torch.Tensor,
        sigma: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> None:
        self.record_vector_step(
            step=step,
            decision="reuse",
            latents_pre=latents_pre,
            latents=latents,
            sigma_pre=sigma_pre,
            sigma=sigma,
            noise_pred=noise_pred,
            local_noise_pred=noise_pred,
            reuse_order=self._order_for_step(step),
        )

    def record_vector_step(
        self,
        *,
        step: int,
        decision: str,
        latents_pre: torch.Tensor,
        latents: torch.Tensor,
        sigma_pre: torch.Tensor,
        sigma: torch.Tensor,
        noise_pred: torch.Tensor,
        local_noise_pred: torch.Tensor,
        reuse_order: Optional[int],
    ) -> None:
        if not self.save_vectors:
            return
        self.vector_trace["guided_v"].append(noise_pred.detach().to(torch.float16).cpu())
        self.vector_trace["local_v"].append(local_noise_pred.detach().to(torch.float16).cpu())
        self.vector_trace["latents_pre"].append(latents_pre.detach().to(torch.float16).cpu())
        self.vector_trace["latents"].append(latents.detach().to(torch.float16).cpu())
        self.vector_trace["sigmas_pre"].append(sigma_pre.detach().to(torch.float32).cpu())
        self.vector_trace["sigmas"].append(sigma.detach().to(torch.float32).cpu())
        self.vector_trace["steps"].append(int(step))
        self.vector_trace["decisions"].append(str(decision))
        self.vector_trace["reuse_orders"].append(-1 if reuse_order is None else int(reuse_order))

    def _plan_for_step(self, step: int) -> JvpStepPlan:
        if step in self._step_plans:
            return self._step_plans[step]

        if self.forced_compute_steps is not None:
            plan = self._forced_plan_for_step(step)
            self._step_plans[step] = plan
            return plan

        if self.anchor_compute_steps is not None:
            plan = self._anchor_plan_for_step(step)
            self._step_plans[step] = plan
            return plan

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
                "reuse_order": self._order_for_step(step),
                "forced_compute": self.forced_compute_steps is not None,
                "forced_reuse_orders": self.forced_reuse_orders is not None,
                "anchor_interval": self.anchor_interval,
                "anchor_phase": self.anchor_phase,
                "anchor_compute": self.anchor_compute_steps is not None,
            }
        )

    def _normalize_forced_compute_steps(self, steps: Optional[List[int]]) -> Optional[set[int]]:
        if steps is None:
            return None
        normalized = {int(step) for step in steps if 0 <= int(step) < self.total_steps}
        if self.total_steps > 0:
            normalized.add(0)
        return normalized

    def _normalize_forced_reuse_orders(self, orders: Optional[Dict[int, int]]) -> Optional[Dict[int, int]]:
        if orders is None:
            return None
        return {
            int(step): max(0, int(order))
            for step, order in orders.items()
            if 0 <= int(step) < self.total_steps
        }

    def _order_for_step(self, step: int) -> int:
        if self.forced_reuse_orders is not None and step in self.forced_reuse_orders:
            return self.forced_reuse_orders[step]
        return self.jvp_order

    def _build_anchor_compute_steps(self) -> Optional[set[int]]:
        if self.anchor_interval is None:
            return None

        phase = self.anchor_phase % self.anchor_interval
        cooldown_start = max(0, self.total_steps - max(0, self.cooldown_steps))
        compute_steps = set()

        for step in range(self.total_steps):
            if step < self.warmup_steps or step >= cooldown_start:
                compute_steps.add(step)
                continue
            relative_step = step - self.warmup_steps - phase
            if relative_step >= 0 and relative_step % self.anchor_interval == 0:
                compute_steps.add(step)

        if self.total_steps > 0:
            compute_steps.add(0)
        return compute_steps

    def _forced_plan_for_step(self, step: int) -> JvpStepPlan:
        if not self.log_dict["v"]:
            return JvpStepPlan(do_reuse=False, pred_drift=float("inf"), accumulated_drift=0.0)

        sigmas = self._scheduler_sigmas()
        pred_drift = 0.0
        if sigmas is not None and step + 1 < len(sigmas):
            pred_drift = self._consensus_scalar(one_step_jvp_drift(self.log_dict, sigmas, step, order=self._order_for_step(step)))

        forced_compute = step in (self.forced_compute_steps or set())
        return JvpStepPlan(
            do_reuse=not forced_compute,
            pred_drift=pred_drift,
            accumulated_drift=0.0,
        )

    def _anchor_plan_for_step(self, step: int) -> JvpStepPlan:
        if not self.log_dict["v"]:
            return JvpStepPlan(do_reuse=False, pred_drift=float("inf"), accumulated_drift=0.0)

        sigmas = self._scheduler_sigmas()
        pred_drift = 0.0
        if sigmas is not None and step + 1 < len(sigmas):
            pred_drift = self._consensus_scalar(one_step_jvp_drift(self.log_dict, sigmas, step, order=self._order_for_step(step)))

        anchor_compute = step in (self.anchor_compute_steps or set())
        return JvpStepPlan(
            do_reuse=not anchor_compute,
            pred_drift=pred_drift,
            accumulated_drift=0.0,
        )

    def _save_step_trace(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        safe_task_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in self.task_id)
        path = os.path.join(output_dir, f"flexcache_freecache_step_trace_{safe_task_id}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.step_trace, handle, indent=2)
        latest_path = os.path.join(output_dir, "flexcache_freecache_step_trace.json")
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(self.step_trace, handle, indent=2)
        if self.save_vectors and self.vector_trace["steps"] and self._should_save_vectors():
            rank_suffix = f"rank{self._rank()}"
            vector_path = os.path.join(output_dir, f"flexcache_freecache_vectors_{safe_task_id}_{rank_suffix}.pt")
            payload = {
                key: torch.stack(values) if values and isinstance(values[0], torch.Tensor) else list(values)
                for key, values in self.vector_trace.items()
            }
            payload["rank"] = self._rank()
            payload["world_size"] = self._world_size()
            payload["cfg_rank"] = self._cfg_rank()
            payload["cp_rank"] = self._cp_rank()
            torch.save(payload, vector_path)
            logger.info("Saved FreeCache replay vectors to %s", vector_path)
        logger.info("Saved FreeCache step trace to %s", path)

    def _should_save_vectors(self) -> bool:
        return self._rank() == 0

    def _rank(self) -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def _world_size(self) -> int:
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def _cfg_rank(self) -> int:
        try:
            from chitu_diffusion.core.distributed.parallel_state import get_cfg_group

            return int(get_cfg_group().rank_in_group)
        except Exception:
            return 0

    def _cp_rank(self) -> int:
        try:
            from chitu_diffusion.core.distributed.parallel_state import get_cp_group

            return int(get_cp_group().rank_in_group)
        except Exception:
            return 0
