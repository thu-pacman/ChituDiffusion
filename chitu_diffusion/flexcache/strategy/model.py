import functools
import os
from logging import getLogger
from typing import Dict, Optional

import torch

from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend
from chitu_diffusion.runtime.output_layout import debug_output_dir

logger = getLogger(__name__)


class ModelStrategy(FlexCacheStrategy):
    """Model-output residual cache driven by token curvature."""

    def __init__(
        self,
        task,
        cache_ratio: float,
        warmup_steps: int,
        cooldown_steps: int,
        tau_max: int,
        curvature_interval_power: float,
    ):
        super().__init__()
        self.type = "model"
        self.tradeoff_score = cache_ratio
        self.num_steps = task.req.params.num_inference_steps
        self.cache_ratio = float(max(0.0, min(1.0, cache_ratio)))
        self.warmup_steps = max(0, int(warmup_steps))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.tau_max = max(1, int(tau_max))
        self.curvature_interval_power = max(0.0, float(curvature_interval_power))
        self.curvature_thresh = self._threshold_from_ratio(self.cache_ratio)

        self.previous_x: Dict[str, torch.Tensor] = {}
        self.accumulated_curvature: Dict[str, float] = {}
        self.last_compute_step: Dict[str, int] = {}
        self._vis_records: Dict[int, int] = {}
        self._vis_max_step = -1

    @staticmethod
    def _threshold_from_ratio(cache_ratio: float) -> float:
        # 0 => quality first, 1 => more aggressive reuse.
        return 0.03 + float(cache_ratio) * 0.27

    def _branch_key(self) -> str:
        if DiffusionBackend.cfg_type == CFGType.NEG:
            return "neg"
        return "pos"

    def _current_step(self) -> int:
        return int(DiffusionBackend.generator.current_task.buffer.current_step)

    def _in_warmup_or_cooldown(self, step: int) -> bool:
        cooldown_start = max(0, self.num_steps - self.cooldown_steps)
        return step < self.warmup_steps or step >= cooldown_start

    def _record_step_policy(self, step: int, code: int):
        if DiffusionBackend.cfg_type != CFGType.POS:
            return
        self._vis_records[step] = max(self._vis_records.get(step, 0), code)
        self._vis_max_step = max(self._vis_max_step, step)

    def _relative_curvature(self, x: torch.Tensor, previous: torch.Tensor) -> float:
        with torch.no_grad():
            x_float = x.detach().float()
            prev_float = previous.to(x.device).float()
            diff = (x_float - prev_float).square().mean().sqrt()
            denom = prev_float.square().mean().sqrt().clamp_min(1e-8)
            return float((diff / denom).detach().cpu().item())

    def get_reuse_key(
        self,
        x: torch.Tensor = None,
        **kwargs,
    ) -> Optional[str]:
        step = self._current_step()
        branch_key = self._branch_key()

        if x is None:
            self._record_step_policy(step, 1)
            return None

        if self._in_warmup_or_cooldown(step):
            self.accumulated_curvature[branch_key] = 0.0
            self._record_step_policy(step, 0)
            return None

        previous = self.previous_x.get(branch_key)
        if previous is None:
            self._record_step_policy(step, 2)
            return None

        curvature = self._relative_curvature(x, previous)
        scaled = curvature ** self.curvature_interval_power if self.curvature_interval_power > 0 else curvature
        accumulated = self.accumulated_curvature.get(branch_key, 0.0) + scaled

        if accumulated < self.curvature_thresh:
            self.accumulated_curvature[branch_key] = accumulated
            self.previous_x[branch_key] = x.detach().clone()
            self._record_step_policy(step, 3)
            return branch_key

        self.accumulated_curvature[branch_key] = 0.0
        self._record_step_policy(step, 2)
        logger.info(
            "[FlexCache model] branch=%s step=%d curvature=%.6e accumulated=%.6e thresh=%.6e",
            branch_key,
            step,
            curvature,
            accumulated,
            self.curvature_thresh,
        )
        return None

    def reuse(self, cached_feature: torch.Tensor, x: torch.Tensor, **kwargs):
        return x + cached_feature

    def get_store_key(
        self,
        x: torch.Tensor = None,
        **kwargs,
    ) -> str:
        branch_key = self._branch_key()
        if x is not None:
            self.previous_x[branch_key] = x.detach().clone()
        self.last_compute_step[branch_key] = self._current_step()
        return branch_key

    def store(self, fresh_feature: torch.Tensor, x: torch.Tensor, **kwargs):
        return fresh_feature - x

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.model_compute

        original_forward = module._original_forward

        @functools.wraps(original_forward)
        def model_compute_with_model_cache(x: torch.Tensor, **kwargs):
            kwargs.pop("raw_e", None)
            reuse_key = self.get_reuse_key(x=x)
            if reuse_key is not None and reuse_key in DiffusionBackend.flexcache.cache:
                return self.reuse(DiffusionBackend.flexcache.cache[reuse_key], x=x)

            original_x = x.clone()
            output = original_forward(x, **kwargs)
            store_key = self.get_store_key(x=original_x)
            DiffusionBackend.flexcache.cache[store_key] = self.store(output, x=original_x)
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_store",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
                extra={"cache_key": str(store_key)},
            )
            return output

        module.model_compute = model_compute_with_model_cache
        logger.info(
            "Module %s wrapped with curvature model strategy: cache_ratio=%.3f thresh=%.6e",
            module.__class__.__name__,
            self.cache_ratio,
            self.curvature_thresh,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        if hasattr(module, "_original_forward"):
            module.model_compute = module._original_forward
            delattr(module, "_original_forward")
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            self._save_policy_ppm(debug_output_dir(run_output_dir))
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_unwrap",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            )
        logger.info("Module %s unwrapped from curvature model strategy", module.__class__.__name__)

    def _save_policy_ppm(self, output_dir: str):
        if self._vis_max_step < 0:
            return

        os.makedirs(output_dir, exist_ok=True)
        cell = 12
        width = (self._vis_max_step + 1) * cell
        height = cell
        rgb = bytearray(width * height * 3)

        for step in range(self._vis_max_step + 1):
            code = self._vis_records.get(step, 0)
            if code == 3:
                color = (255, 180, 40)  # reuse
            elif code == 2:
                color = (30, 180, 80)   # anchor: compute + decision
            elif code == 1:
                color = (40, 140, 255)  # compute
            else:
                color = (160, 160, 160) # warmup/cooldown

            for yy in range(height):
                for xx in range(step * cell, (step + 1) * cell):
                    idx = (yy * width + xx) * 3
                    rgb[idx] = color[0]
                    rgb[idx + 1] = color[1]
                    rgb[idx + 2] = color[2]

        ppm_path = os.path.join(output_dir, "flexcache_model_policy.ppm")
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            f.write(bytes(rgb))

    def reset_state(self):
        self.previous_x.clear()
        self.accumulated_curvature.clear()
        self.last_compute_step.clear()
        self._vis_records.clear()
        self._vis_max_step = -1
        DiffusionBackend.flexcache.cache.clear()
