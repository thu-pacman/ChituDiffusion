import functools
import os
from logging import getLogger

import torch

from chitu_diffusion.flexcache.core import AnchorCachePlanner
from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.runtime.backend import DiffusionBackend
from chitu_diffusion.runtime.output_layout import debug_output_dir

logger = getLogger(__name__)


class LayerStrategy(FlexCacheStrategy):
    """Curvature-driven transformer-block output cache."""

    def __init__(self, task, cache_ratio: float, warmup_steps: int, cooldown_steps: int, tau_max: int, curvature_interval_power: float):
        super().__init__()
        self.type = "layer"
        self.tradeoff_score = cache_ratio
        self.planner = AnchorCachePlanner(
            cache_ratio=cache_ratio,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            total_steps=task.req.params.num_inference_steps,
            tau_max=tau_max,
            curvature_interval_power=curvature_interval_power,
        )

    def _key(self, layer_id: int):
        return (self.planner.branch_key(), "layer", int(layer_id))

    def get_reuse_key(self, layer_id: int, **kwargs):
        decision = self.planner.decide(self._key(layer_id))
        return None if decision.should_compute else decision.key

    def reuse(self, cached_feature: torch.Tensor, **kwargs):
        return cached_feature

    def get_store_key(self, layer_id: int, **kwargs):
        return self._key(layer_id)

    def store(self, fresh_feature: torch.Tensor, **kwargs):
        return fresh_feature.detach()

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        for layer_id, block in enumerate(module.blocks):
            if not hasattr(block, "_original_forward"):
                block._original_forward = block.forward
            original_forward = block._original_forward

            @functools.wraps(original_forward)
            def block_forward_with_layer_cache(*args, layer_id=layer_id, original_fn=original_forward, **kwargs):
                decision = self.planner.decide(self._key(layer_id))
                key = None if decision.should_compute else decision.key
                if key is not None and key in DiffusionBackend.flexcache.cache:
                    return self.reuse(DiffusionBackend.flexcache.cache[key])

                output = original_fn(*args, **kwargs)
                store_key = self.get_store_key(layer_id)
                DiffusionBackend.flexcache.cache[store_key] = self.store(output)
                self.planner.mark_computed(store_key, output, is_anchor=decision.is_anchor)
                DiffusionBackend.flexcache.record_cache_memory(
                    "flexcache_store",
                    task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
                    extra={"cache_key": str(store_key)},
                )
                return output

            block.forward = block_forward_with_layer_cache
        logger.info("Module %s wrapped with curvature layer strategy", module.__class__.__name__)

    def unwrap_module(self, module: torch.nn.Module) -> None:
        for block in module.blocks:
            if hasattr(block, "_original_forward"):
                block.forward = block._original_forward
                delattr(block, "_original_forward")
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            self.planner.save_decision_ppm(debug_output_dir(run_output_dir), "flexcache_layer_policy.ppm")
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_unwrap",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            )
        logger.info("Module %s unwrapped from curvature layer strategy", module.__class__.__name__)

    def reset_state(self):
        self.planner.reset()
        DiffusionBackend.flexcache.clear_cache()
