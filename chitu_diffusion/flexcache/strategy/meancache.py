from __future__ import annotations

from logging import getLogger
from typing import Optional

import torch

from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.runtime.backend import DiffusionBackend

logger = getLogger(__name__)


_FRESH_STEP_TABLE = {
    30: [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        24,
        28,
        32,
        36,
        40,
        43,
        45,
        47,
        48,
        49,
    ],
    25: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32, 40, 43, 45, 47, 48, 49],
    17: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 17, 24, 31, 38, 45, 48, 49],
    10: [0, 1, 3, 7, 14, 21, 28, 35, 42, 49],
}

_EDGE_SOURCE = {
    30: [
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean_jvp1_s5",
        "v_diff_mean_jvp1_s5",
        "v_diff_mean_jvp1_s5",
        "chain",
        "chain",
    ],
    25: [
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean_jvp1_s5",
        "v_diff_mean_jvp1_s5",
        "v_diff_mean_jvp1_s5",
        "chain",
        "chain",
    ],
    17: [
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "chain",
        "v_diff_mean_jvp1_s2",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean_jvp1_s4",
        "chain",
    ],
    10: [
        "chain",
        "v_diff_mean_jvp1_s2",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean",
        "v_diff_mean_jvp1_s4",
    ],
}

_EDGE_ORDER = {
    "v_diff_mean": 1,
    "v_diff_mean_jvp1_s2": 2,
    "v_diff_mean_jvp1_s3": 3,
    "v_diff_mean_jvp1_s4": 4,
    "v_diff_mean_jvp1_s5": 5,
    "v_diff_mean_jvp1_s6": 6,
    "chain": 0,
}


class MeanCacheStrategy(FlexCacheStrategy):
    """Step-level MeanCache/JVP reuse strategy for flow-matching models.

    The current fresh-step schedules and JVP rules were first validated on the
    Qwen-Image reference implementation under refs/MeanCache. The strategy
    itself operates above the transformer block API, so adapters such as
    Qwen-Image and FLUX can reuse the same step-selection and velocity
    prediction logic as long as they provide latents, sigmas, and fresh noise
    predictions in the expected format.
    """

    def __init__(
        self,
        task,
        fresh_steps: int = 25,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
        use_jvp: bool = True,
    ):
        super().__init__()
        self.type = "meancache"
        self.total_steps = int(task.req.params.num_inference_steps)
        self.fresh_steps = int(fresh_steps)
        self.warmup_steps = int(warmup_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.use_jvp = bool(use_jvp)
        self.tradeoff_score = self.total_steps / max(1, self.fresh_steps)
        self.should_calc = self._build_should_calc()
        self.edge_order = self._build_edge_order()
        self.log_dict = {
            "v": [],
            "latents_pre": [],
            "latents": [],
            "sigmas_pre": [],
            "sigmas": [],
        }

    def get_reuse_key(self, **kwargs) -> Optional[str]:
        return None if self.should_compute(int(kwargs["step"])) else "meancache_noise_pred"

    def reuse(self, **kwargs):
        return self.predict_noise_pred(int(kwargs["step"]))

    def get_store_key(self, **kwargs) -> Optional[str]:
        return "meancache_noise_pred" if self.should_compute(int(kwargs["step"])) else None

    def store(self, **kwargs) -> None:
        self.record_fresh_step(
            step=int(kwargs["step"]),
            latents_pre=kwargs["latents_pre"],
            latents=kwargs["latents"],
            sigma_pre=kwargs["sigma_pre"],
            sigma=kwargs["sigma"],
            noise_pred=kwargs["noise_pred"],
        )
        return None

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        logger.info(
            "MeanCache step strategy enabled: fresh_steps=%d use_jvp=%s warmup=%d cooldown=%d",
            self.fresh_steps,
            self.use_jvp,
            self.warmup_steps,
            self.cooldown_steps,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        logger.info("MeanCache step strategy disabled.")

    def reset_state(self):
        for values in self.log_dict.values():
            values.clear()
        DiffusionBackend.flexcache.clear_cache()

    def should_compute(self, step: int) -> bool:
        if step < self.warmup_steps or step >= self.total_steps - self.cooldown_steps:
            return True
        if step < 0 or step >= len(self.should_calc):
            return True
        return self.should_calc[step]

    def predict_noise_pred(self, step: int) -> torch.Tensor:
        if not self.log_dict["v"]:
            raise RuntimeError("MeanCache cannot reuse before the first fresh noise prediction.")
        if not self.use_jvp:
            return self.log_dict["v"][-1].clone()

        order = 1
        if 0 <= step - 1 < len(self.edge_order):
            order = max(1, int(self.edge_order[step - 1]))
        jvp = self._compute_current_jvp(order)
        if jvp is None:
            return self.log_dict["v"][-1].clone()

        sigmas = self._scheduler_sigmas()
        if sigmas is None or step + 1 >= len(sigmas):
            return self.log_dict["v"][-1].clone()
        denom = sigmas[step + 1].to(jvp.device, jvp.dtype) - sigmas[step].to(jvp.device, jvp.dtype)
        return self.log_dict["v"][-1].to(jvp.device, jvp.dtype) - jvp.to(jvp.dtype) * denom

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

    def _build_should_calc(self) -> list[bool]:
        steps = [step for step in _FRESH_STEP_TABLE.get(self.fresh_steps, _FRESH_STEP_TABLE[25]) if step < self.total_steps]
        should = [False] * self.total_steps
        for step in steps:
            should[step] = True
        for step in range(min(self.warmup_steps, self.total_steps)):
            should[step] = True
        cooldown_start = max(0, self.total_steps - self.cooldown_steps)
        for step in range(cooldown_start, self.total_steps):
            should[step] = True
        return should

    def _build_edge_order(self) -> list[int]:
        fresh = _FRESH_STEP_TABLE.get(self.fresh_steps, _FRESH_STEP_TABLE[25])
        edge_source = _EDGE_SOURCE.get(self.fresh_steps, _EDGE_SOURCE[25])
        result = [1] * max(self.total_steps, 1)
        edge_order = [_EDGE_ORDER.get(rule, 1) for rule in edge_source]
        for idx in range(min(len(fresh) - 1, len(edge_order))):
            start = fresh[idx]
            end = fresh[idx + 1]
            for pos in range(start, min(end, self.total_steps)):
                result[pos] = edge_order[idx]
        return result

    def _compute_current_jvp(self, steps: int) -> Optional[torch.Tensor]:
        count = len(self.log_dict["v"])
        if count == 0:
            return None
        actual_steps = min(max(1, steps), count)
        i1 = count - 1
        i0 = count - actual_steps
        x_start = self.log_dict["latents_pre"][i0]
        x_end = self.log_dict["latents"][i1]
        s_start = self.log_dict["sigmas_pre"][i0]
        s_end = self.log_dict["sigmas"][i1]
        denom = (s_end - s_start).to(device=x_start.device, dtype=x_start.dtype)
        denom_b = denom.view(-1, *([1] * (x_start.ndim - denom.dim())))
        avg_u = (x_end - x_start) / denom_b
        inst_u = self.log_dict["v"][i0]
        return (inst_u - avg_u) / denom_b

    def _scheduler_sigmas(self) -> Optional[torch.Tensor]:
        task = DiffusionBackend.generator.current_task
        sampler = getattr(task.buffer, "sampler", None)
        sigmas = getattr(sampler, "sigmas", None)
        return sigmas
