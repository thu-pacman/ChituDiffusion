from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.flexcache.freecache_core import jvp_predict_noise_pred
from chitu_diffusion.runtime.backend import DiffusionBackend

logger = getLogger(__name__)


@dataclass
class PlannerState:
    x_hat: torch.Tensor
    actions: List[str]
    step_errors: List[float]
    fresh_steps: List[int]
    reuse_orders: Dict[int, int]
    fresh_count: int
    last_fresh_step: int
    gap: int
    err_sum: float
    err_max: float
    err_tail: float
    log_dict: Dict[str, List[Any]] = field(default_factory=dict)

    def score(self, step_count: int, alpha: float, beta: float) -> float:
        denom = max(1, step_count)
        return self.err_sum / denom + alpha * self.err_max + beta * self.err_tail


@dataclass
class PendingFreshCandidate:
    state_id: int
    base: Optional[PlannerState]
    step: int
    x_current: torch.Tensor
    oracle_next: torch.Tensor
    sigma_pre: torch.Tensor
    sigma: torch.Tensor
    runtime_dtype: torch.dtype


class TracePlannerStrategy(FlexCacheStrategy):
    """Full-compute probe that searches fresh/reuse0/reuse1 policies in memory."""

    def __init__(
        self,
        task,
        budgets: Optional[List[int]] = None,
        beam_width: int = 8,
        max_gap: int = 8,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
        hard_tol: float = 1.0e-2,
        alpha: float = 2.0,
        beta: float = 1.0,
        checkpoint_interval: int = 0,
        checkpoint_topk: int = 0,
    ):
        super().__init__()
        self.type = "traceplanner"
        self.task_id = str(task.task_id)
        self.total_steps = int(task.req.params.num_inference_steps)
        default_budget = max(1, self.total_steps // 2)
        self.budgets = sorted({int(b) for b in (budgets or [default_budget]) if int(b) > 0})
        self.max_budget = min(max(self.budgets), self.total_steps)
        self.beam_width = max(1, int(beam_width))
        self.max_gap = max(1, int(max_gap))
        self.warmup_steps = max(0, int(warmup_steps))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.hard_tol = float(hard_tol)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.checkpoint_interval = max(0, int(checkpoint_interval))
        self.checkpoint_topk = max(0, int(checkpoint_topk))
        self.tradeoff_score = 0.0
        self.states: List[PlannerState] = []
        self.policies: Dict[int, Dict[str, Any]] = {}
        self.step_trace: List[Dict[str, Any]] = []
        self._pending_candidates: Optional[List[PlannerState]] = None
        self._pending_fresh: List[PendingFreshCandidate] = []

    def get_reuse_key(self, **kwargs) -> Optional[str]:
        return None

    def reuse(self, **kwargs) -> torch.Tensor:
        raise RuntimeError("TracePlannerStrategy never reuses cached predictions.")

    def get_store_key(self, **kwargs) -> Optional[str]:
        return "traceplanner_noise_pred"

    def store(self, **kwargs) -> None:
        step = int(kwargs["step"])
        latents_pre = kwargs["latents_pre"].detach().to(torch.float32)
        latents = kwargs["latents"].detach().to(torch.float32)
        sigma_pre = kwargs["sigma_pre"].detach().to(torch.float32)
        sigma = kwargs["sigma"].detach().to(torch.float32)
        raw_guided_noise_pred = kwargs.get("guided_noise_pred", kwargs["noise_pred"]).detach()
        runtime_dtype = raw_guided_noise_pred.dtype
        guided_noise_pred = raw_guided_noise_pred.to(torch.float32)

        if self._rank() == 0:
            self._advance_states(
                step=step,
                latents_pre=latents_pre,
                latents=latents,
                sigma_pre=sigma_pre,
                sigma=sigma,
                guided_noise_pred=guided_noise_pred,
                runtime_dtype=runtime_dtype,
            )

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        logger.info(
            "TracePlanner strategy enabled: budgets=%s beam=%d max_gap=%d warmup=%d cooldown=%d hard_tol=%.3e checkpoint_interval=%d checkpoint_topk=%d",
            self.budgets,
            self.beam_width,
            self.max_gap,
            self.warmup_steps,
            self.cooldown_steps,
            self.hard_tol,
            self.checkpoint_interval,
            self.checkpoint_topk,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        if self._rank() == 0:
            self._finalize_policies()
            run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
            if run_output_dir:
                self._save_policies(run_output_dir)
        logger.info("TracePlanner strategy disabled.")

    def reset_state(self) -> None:
        self.states.clear()
        self.policies.clear()
        self.step_trace.clear()
        self._pending_candidates = None
        self._pending_fresh = []
        DiffusionBackend.flexcache.clear_cache()

    def checkpoint_enabled_for_step(self, step: int) -> bool:
        return (
            self.checkpoint_interval > 0
            and self.checkpoint_topk > 0
            and step > 0
            and step >= self.warmup_steps
            and step < self.total_steps - self.cooldown_steps
            and step % self.checkpoint_interval == 0
        )

    def plan_checkpoint_requests(self, **kwargs) -> List[torch.Tensor]:
        step = int(kwargs["step"])
        latents_pre = kwargs["latents_pre"].detach().to(torch.float32)
        latents = kwargs["latents"].detach().to(torch.float32)
        sigma_pre = kwargs["sigma_pre"].detach().to(torch.float32)
        sigma = kwargs["sigma"].detach().to(torch.float32)
        raw_guided_noise_pred = kwargs.get("guided_noise_pred", kwargs["noise_pred"]).detach()
        runtime_dtype = raw_guided_noise_pred.dtype
        guided_noise_pred = raw_guided_noise_pred.to(torch.float32)

        if self._rank() != 0 or not self.checkpoint_enabled_for_step(step):
            return []

        candidates, fresh_candidates = self._build_candidates(
            step=step,
            latents_pre=latents_pre,
            latents=latents,
            sigma_pre=sigma_pre,
            sigma=sigma,
            guided_noise_pred=guided_noise_pred,
            runtime_dtype=runtime_dtype,
        )
        if not fresh_candidates:
            self._pending_candidates = None
            self._pending_fresh = []
            return []

        fresh_candidates.sort(key=lambda item: item[1].score(len(item[1].actions), self.alpha, self.beta))
        selected = fresh_candidates[: self.checkpoint_topk]
        selected_ids = {state_id for state_id, _state, _pending in selected}
        self._pending_candidates = [state for state in candidates if id(state) not in selected_ids]
        self._pending_fresh = [pending for _state_id, _state, pending in selected]
        logger.info(
            "TracePlanner checkpoint step=%d requests=%d candidates=%d states=%d",
            step,
            len(self._pending_fresh),
            len(candidates),
            len(self.states),
        )
        return [pending.x_current.detach() for pending in self._pending_fresh]

    def finish_checkpoint_requests(self, checkpoint_noise_preds: List[torch.Tensor]) -> None:
        if self._rank() != 0 or self._pending_candidates is None:
            return
        if len(checkpoint_noise_preds) != len(self._pending_fresh):
            raise RuntimeError(
                f"TracePlanner expected {len(self._pending_fresh)} checkpoint predictions, got {len(checkpoint_noise_preds)}."
            )

        candidates = list(self._pending_candidates)
        for pending, noise_pred in zip(self._pending_fresh, checkpoint_noise_preds):
            candidates.append(
                self._fresh_state(
                    base=pending.base,
                    step=pending.step,
                    x_current=pending.x_current,
                    oracle_next=pending.oracle_next,
                    sigma_pre=pending.sigma_pre,
                    sigma=pending.sigma,
                    noise_pred=noise_pred.detach().to(torch.float32),
                    runtime_dtype=pending.runtime_dtype,
                )
            )

        self.states = self._prune([state for state in candidates if math.isfinite(state.err_tail)])
        if self._pending_fresh:
            logger.info(
                "TracePlanner checkpoint step=%d finished states=%d",
                self._pending_fresh[0].step,
                len(self.states),
            )
            self._record_planner_row(self._pending_fresh[0].step)
        self._pending_candidates = None
        self._pending_fresh = []

    def _advance_states(
        self,
        *,
        step: int,
        latents_pre: torch.Tensor,
        latents: torch.Tensor,
        sigma_pre: torch.Tensor,
        sigma: torch.Tensor,
        guided_noise_pred: torch.Tensor,
        runtime_dtype: torch.dtype,
    ) -> None:
        candidates, _fresh_candidates = self._build_candidates(
            step=step,
            latents_pre=latents_pre,
            latents=latents,
            sigma_pre=sigma_pre,
            sigma=sigma,
            guided_noise_pred=guided_noise_pred,
            runtime_dtype=runtime_dtype,
        )
        self.states = self._prune([state for state in candidates if math.isfinite(state.err_tail)])
        self._record_planner_row(step)

    def _build_candidates(
        self,
        *,
        step: int,
        latents_pre: torch.Tensor,
        latents: torch.Tensor,
        sigma_pre: torch.Tensor,
        sigma: torch.Tensor,
        guided_noise_pred: torch.Tensor,
        runtime_dtype: torch.dtype,
    ) -> tuple[List[PlannerState], List[tuple[int, PlannerState, PendingFreshCandidate]]]:
        if step == 0 or not self.states:
            state = self._fresh_state(
                base=None,
                step=step,
                x_current=latents_pre,
                oracle_next=latents,
                sigma_pre=sigma_pre,
                sigma=sigma,
                noise_pred=guided_noise_pred,
                runtime_dtype=runtime_dtype,
            )
            return [state], []

        candidates: List[PlannerState] = []
        fresh_candidates: List[tuple[int, PlannerState, PendingFreshCandidate]] = []
        force_fresh = step < self.warmup_steps or step >= self.total_steps - self.cooldown_steps
        for state in self.states:
            if state.fresh_count < self.max_budget:
                fresh_state = self._fresh_state(
                    base=state,
                    step=step,
                    x_current=state.x_hat,
                    oracle_next=latents,
                    sigma_pre=sigma_pre,
                    sigma=sigma,
                    noise_pred=guided_noise_pred,
                    runtime_dtype=runtime_dtype,
                )
                candidates.append(fresh_state)
                fresh_candidates.append(
                    (
                        id(fresh_state),
                        fresh_state,
                        PendingFreshCandidate(
                            state_id=id(fresh_state),
                            base=state,
                            step=step,
                            x_current=state.x_hat.detach(),
                            oracle_next=latents.detach(),
                            sigma_pre=sigma_pre.detach(),
                            sigma=sigma.detach(),
                            runtime_dtype=runtime_dtype,
                        ),
                    )
                )

            if force_fresh or state.gap >= self.max_gap:
                continue
            candidates.append(
                self._reuse_state(
                    base=state,
                    step=step,
                    oracle_next=latents,
                    sigma_pre=sigma_pre,
                    sigma=sigma,
                    order=0,
                    runtime_dtype=runtime_dtype,
                )
            )
            candidates.append(
                self._reuse_state(
                    base=state,
                    step=step,
                    oracle_next=latents,
                    sigma_pre=sigma_pre,
                    sigma=sigma,
                    order=1,
                    runtime_dtype=runtime_dtype,
                )
            )

        return candidates, fresh_candidates

    def _fresh_state(
        self,
        *,
        base: Optional[PlannerState],
        step: int,
        x_current: torch.Tensor,
        oracle_next: torch.Tensor,
        sigma_pre: torch.Tensor,
        sigma: torch.Tensor,
        noise_pred: torch.Tensor,
        runtime_dtype: torch.dtype,
    ) -> PlannerState:
        x_next = self._scheduler_step(x_current, noise_pred, sigma_pre, sigma, runtime_dtype=runtime_dtype)
        err = self._rel_mse(x_next, oracle_next)
        if base is None:
            actions: List[str] = ["F"]
            step_errors = [err]
            fresh_steps = [step]
            reuse_orders: Dict[int, int] = {}
            err_sum = err
            err_max = err
            log_dict = self._empty_log_dict()
        else:
            actions = [*base.actions, "F"]
            step_errors = [*base.step_errors, err]
            fresh_steps = [*base.fresh_steps, step]
            reuse_orders = dict(base.reuse_orders)
            err_sum = base.err_sum + err
            err_max = max(base.err_max, err)
            log_dict = self._clone_log_dict(base.log_dict)

        log_dict["v"] = [noise_pred.detach()]
        log_dict["latents_pre"] = [x_current.detach()]
        log_dict["latents"] = [x_next.detach()]
        log_dict["sigmas_pre"] = [sigma_pre.detach()]
        log_dict["sigmas"] = [sigma.detach()]
        return PlannerState(
            x_hat=x_next.detach(),
            actions=actions,
            step_errors=step_errors,
            fresh_steps=fresh_steps,
            reuse_orders=reuse_orders,
            fresh_count=len(fresh_steps),
            last_fresh_step=step,
            gap=0,
            err_sum=err_sum,
            err_max=err_max,
            err_tail=err,
            log_dict=log_dict,
        )

    def _reuse_state(
        self,
        *,
        base: PlannerState,
        step: int,
        oracle_next: torch.Tensor,
        sigma_pre: torch.Tensor,
        sigma: torch.Tensor,
        order: int,
        runtime_dtype: torch.dtype,
    ) -> PlannerState:
        sigmas = self._sigmas_for_prediction(base.log_dict, step, sigma_pre, sigma)
        v_hat = jvp_predict_noise_pred(base.log_dict, sigmas, step, order=order)
        x_next = self._scheduler_step(base.x_hat, v_hat, sigma_pre, sigma, runtime_dtype=runtime_dtype)
        err = self._rel_mse(x_next, oracle_next)
        return PlannerState(
            x_hat=x_next.detach(),
            actions=[*base.actions, f"R{order}"],
            step_errors=[*base.step_errors, err],
            fresh_steps=list(base.fresh_steps),
            reuse_orders={**base.reuse_orders, step: order},
            fresh_count=base.fresh_count,
            last_fresh_step=base.last_fresh_step,
            gap=base.gap + 1,
            err_sum=base.err_sum + err,
            err_max=max(base.err_max, err),
            err_tail=err,
            log_dict=self._clone_log_dict(base.log_dict),
        )

    def _prune(self, candidates: List[PlannerState]) -> List[PlannerState]:
        buckets: Dict[int, List[PlannerState]] = {}
        for state in candidates:
            if state.fresh_count > self.max_budget:
                continue
            if state.err_tail > self.hard_tol:
                continue
            buckets.setdefault(state.fresh_count, []).append(state)

        kept: List[PlannerState] = []
        for fresh_count, states in buckets.items():
            states.sort(key=lambda state: state.score(len(state.actions), self.alpha, self.beta))
            nondominated: List[PlannerState] = []
            for state in states:
                if any(self._dominates(other, state) for other in nondominated):
                    continue
                nondominated.append(state)
                if len(nondominated) >= self.beam_width:
                    break
            kept.extend(nondominated)
        return kept

    def _dominates(self, a: PlannerState, b: PlannerState) -> bool:
        return (
            a.fresh_count <= b.fresh_count
            and a.gap <= b.gap
            and a.err_sum <= b.err_sum
            and a.err_max <= b.err_max
            and a.err_tail <= b.err_tail
        )

    def _finalize_policies(self) -> None:
        for budget in self.budgets:
            eligible = [state for state in self.states if state.fresh_count <= budget]
            if not eligible:
                eligible = list(self.states)
            if not eligible:
                continue
            best = min(eligible, key=lambda state: state.score(len(state.actions), self.alpha, self.beta))
            self.policies[budget] = self._policy_payload(best, budget)

    def _policy_payload(self, state: PlannerState, budget: int) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "budget": int(budget),
            "fresh_count": int(state.fresh_count),
            "reuse_count": int(len(state.actions) - state.fresh_count),
            "actions": list(state.actions),
            "step_errors": [float(err) for err in state.step_errors],
            "forced_compute_steps": list(state.fresh_steps),
            "forced_reuse_orders": {str(step): int(order) for step, order in sorted(state.reuse_orders.items())},
            "err_sum": float(state.err_sum),
            "err_max": float(state.err_max),
            "err_tail": float(state.err_tail),
            "score": float(state.score(len(state.actions), self.alpha, self.beta)),
            "planner": {
                "beam_width": self.beam_width,
                "max_gap": self.max_gap,
                "hard_tol": self.hard_tol,
                "alpha": self.alpha,
                "beta": self.beta,
                "warmup": self.warmup_steps,
                "cooldown": self.cooldown_steps,
            },
        }

    def _record_planner_row(self, step: int) -> None:
        by_fresh: Dict[int, int] = {}
        for state in self.states:
            by_fresh[state.fresh_count] = by_fresh.get(state.fresh_count, 0) + 1
        best = min(self.states, key=lambda state: state.score(len(state.actions), self.alpha, self.beta)) if self.states else None
        self.step_trace.append(
            {
                "step": int(step),
                "num_states": len(self.states),
                "states_by_fresh": by_fresh,
                "best_fresh_count": None if best is None else int(best.fresh_count),
                "best_err_tail": None if best is None else float(best.err_tail),
                "best_err_max": None if best is None else float(best.err_max),
                "best_score": None if best is None else float(best.score(len(best.actions), self.alpha, self.beta)),
            }
        )

    def _save_policies(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        safe_task_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in self.task_id)
        summary_path = os.path.join(output_dir, f"flexcache_traceplanner_{safe_task_id}.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump({"policies": self.policies, "step_trace": self.step_trace}, handle, indent=2)
        latest_path = os.path.join(output_dir, "flexcache_traceplanner.json")
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump({"policies": self.policies, "step_trace": self.step_trace}, handle, indent=2)
        for budget, payload in self.policies.items():
            policy_path = os.path.join(output_dir, f"flexcache_traceplanner_policy_B{budget}.json")
            with open(policy_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        logger.info("Saved TracePlanner policies to %s", summary_path)

    def _scheduler_step(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        sigma_pre: torch.Tensor,
        sigma: torch.Tensor,
        *,
        runtime_dtype: torch.dtype,
    ) -> torch.Tensor:
        dt = sigma.to(torch.float32) - sigma_pre.to(torch.float32)
        if dt.dim() == 0:
            dt_b = dt.reshape(*([1] * x.ndim))
        else:
            dt_b = dt.view(*dt.shape, *([1] * (x.ndim - dt.dim())))
        return (x.to(torch.float32) + dt_b * v.to(runtime_dtype).to(torch.float32)).to(runtime_dtype).to(torch.float32)

    def _rel_mse(self, current: torch.Tensor, reference: torch.Tensor) -> float:
        diff = current.to(torch.float64) - reference.to(torch.float64)
        ref = reference.to(torch.float64)
        return float(diff.square().mean().item() / max(ref.square().mean().item(), 1e-20))

    def _sigmas_for_prediction(
        self,
        log_dict: Dict[str, List[Any]],
        step: int,
        sigma_pre: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        sigmas = self._scheduler_sigmas()
        if sigmas is not None and step + 1 < len(sigmas):
            return sigmas
        values = [None] * (step + 2)
        values[step] = sigma_pre.detach()
        values[step + 1] = sigma.detach()
        for idx, value in enumerate(values):
            if value is None:
                values[idx] = torch.tensor(0.0, dtype=torch.float32)
        return torch.stack([value.to(torch.float32) for value in values])

    def _scheduler_sigmas(self) -> Optional[torch.Tensor]:
        task = DiffusionBackend.generator.current_task
        sampler = getattr(task.buffer, "sampler", None)
        if sampler is not None:
            sigmas = getattr(sampler, "sigmas", None)
            if sigmas is not None:
                return sigmas.detach()
        pipe = getattr(task.buffer, "pipe", None)
        if pipe is not None:
            sigmas = getattr(getattr(pipe, "scheduler", None), "sigmas", None)
            if sigmas is not None:
                return sigmas.detach()
        return None

    def _empty_log_dict(self) -> Dict[str, List[Any]]:
        return {"v": [], "latents_pre": [], "latents": [], "sigmas_pre": [], "sigmas": []}

    def _clone_log_dict(self, log_dict: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return {key: list(values) for key, values in log_dict.items()}

    def _rank(self) -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
