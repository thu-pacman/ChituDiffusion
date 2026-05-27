import functools
import os
from logging import getLogger

import torch

from chitu_diffusion.flexcache.core import AnchorCachePlanner
from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.runtime.backend import DiffusionBackend
from chitu_diffusion.runtime.output_layout import debug_output_dir

logger = getLogger(__name__)


class ModelStrategy(FlexCacheStrategy):
    """Model-output residual cache with curvature-driven next-compute scheduling."""

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
        self.cache_ratio = float(max(0.0, min(1.0, cache_ratio)))
        self.warmup_steps = max(0, int(warmup_steps))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.tau_max = max(1, int(tau_max))
        self.curvature_interval_power = max(0.0, float(curvature_interval_power))
        self.planner = AnchorCachePlanner(
            cache_ratio=self.cache_ratio,
            warmup_steps=self.warmup_steps,
            cooldown_steps=self.cooldown_steps,
            total_steps=task.req.params.num_inference_steps,
            tau_max=self.tau_max,
            curvature_interval_power=self.curvature_interval_power,
            mode="model_curvature",
        )

    def get_reuse_key(self, x: torch.Tensor = None, **kwargs):
        key = self.planner.branch_key()
        decision = self.planner.decide(key)
        if decision.should_compute:
            return None
        return key

    def reuse(self, cached_feature: torch.Tensor, x: torch.Tensor, **kwargs):
        return x + cached_feature

    def get_store_key(self, x: torch.Tensor = None, **kwargs):
        return self.planner.branch_key()

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
            residual = self.store(output, x=original_x)
            DiffusionBackend.flexcache.cache[store_key] = residual
            self.planner.mark_computed(store_key, residual, is_anchor=True)
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_store",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
                extra={"cache_key": str(store_key)},
            )
            return output

        module.model_compute = model_compute_with_model_cache
        logger.info(
            "Module %s wrapped with curvature model strategy: cache_ratio=%.3f tau_max=%d",
            module.__class__.__name__,
            self.cache_ratio,
            self.tau_max,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        if hasattr(module, "_original_forward"):
            module.model_compute = module._original_forward
            delattr(module, "_original_forward")
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            output_dir = debug_output_dir(run_output_dir)
            self.planner.save_decision_ppm(output_dir, "flexcache_model_policy.ppm")
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_unwrap",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            )
        logger.info("Module %s unwrapped from curvature model strategy", module.__class__.__name__)

    def reset_state(self):
        self.planner.reset()
        DiffusionBackend.flexcache.clear_cache()
