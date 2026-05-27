import functools
import os
from logging import getLogger

import torch

from chitu_diffusion.flexcache.core import AnchorCachePlanner
from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.runtime.backend import DiffusionBackend
from chitu_diffusion.runtime.output_layout import debug_output_dir

logger = getLogger(__name__)


class AttnStrategy(FlexCacheStrategy):
    """Curvature-driven attention-module output cache."""

    def __init__(self, task, cache_ratio: float, warmup_steps: int, cooldown_steps: int, tau_max: int, curvature_interval_power: float):
        super().__init__()
        self.type = "attn"
        self.tradeoff_score = cache_ratio
        self.planner = AnchorCachePlanner(
            cache_ratio=cache_ratio,
            warmup_steps=warmup_steps,
            cooldown_steps=cooldown_steps,
            total_steps=task.req.params.num_inference_steps,
            tau_max=tau_max,
            curvature_interval_power=curvature_interval_power,
        )

    def _key(self, layer_id: int, attn_kind: str):
        return (self.planner.branch_key(), "attn", int(layer_id), str(attn_kind))

    def get_reuse_key(self, layer_id: int, attn_kind: str = "self", **kwargs):
        decision = self.planner.decide(self._key(layer_id, attn_kind))
        return None if decision.should_compute else decision.key

    def reuse(self, cached_feature: torch.Tensor, **kwargs):
        return cached_feature

    def get_store_key(self, layer_id: int, attn_kind: str = "self", **kwargs):
        return self._key(layer_id, attn_kind)

    def store(self, fresh_feature: torch.Tensor, **kwargs):
        return fresh_feature.detach()

    def _wrap_attn(self, attn_module, layer_id: int, attn_kind: str):
        if not hasattr(attn_module, "_original_forward"):
            attn_module._original_forward = attn_module.forward
        original_forward = attn_module._original_forward

        @functools.wraps(original_forward)
        def forward_with_attn_cache(*args, layer_id=layer_id, attn_kind=attn_kind, original_fn=original_forward, **kwargs):
            decision = self.planner.decide(self._key(layer_id, attn_kind))
            key = None if decision.should_compute else decision.key
            if key is not None and key in DiffusionBackend.flexcache.cache:
                return self.reuse(DiffusionBackend.flexcache.cache[key])

            output = original_fn(*args, **kwargs)
            store_key = self.get_store_key(layer_id, attn_kind=attn_kind)
            DiffusionBackend.flexcache.cache[store_key] = self.store(output)
            self.planner.mark_computed(store_key, output, is_anchor=decision.is_anchor)
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_store",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
                extra={"cache_key": str(store_key)},
            )
            return output

        attn_module.forward = forward_with_attn_cache

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        for layer_id, block in enumerate(module.blocks):
            if hasattr(block, "self_attn"):
                self._wrap_attn(block.self_attn, layer_id, "self")
            if hasattr(block, "cross_attn"):
                self._wrap_attn(block.cross_attn, layer_id, "cross")
        logger.info("Module %s wrapped with curvature attn strategy", module.__class__.__name__)

    def unwrap_module(self, module: torch.nn.Module) -> None:
        for block in module.blocks:
            for name in ("self_attn", "cross_attn"):
                attn_module = getattr(block, name, None)
                if attn_module is not None and hasattr(attn_module, "_original_forward"):
                    attn_module.forward = attn_module._original_forward
                    delattr(attn_module, "_original_forward")
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            self.planner.save_decision_ppm(debug_output_dir(run_output_dir), "flexcache_attn_policy.ppm")
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_unwrap",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            )
        logger.info("Module %s unwrapped from curvature attn strategy", module.__class__.__name__)

    def reset_state(self):
        self.planner.reset()
        DiffusionBackend.flexcache.cache.clear()
