import functools
import math
import os
from logging import getLogger
from typing import Any, Dict, Optional

import torch
import torch.amp as amp

from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend
from chitu_diffusion.runtime.output_layout import debug_output_dir

logger = getLogger(__name__)


def _fp32_autocast():
    return amp.autocast(device_type="cuda", dtype=torch.float32, enabled=torch.cuda.is_available())


def _taylor_formula(derivative_dict: Dict[int, torch.Tensor], distance: int) -> torch.Tensor:
    output = None
    for order in sorted(derivative_dict):
        term = derivative_dict[order] * (distance ** order) / math.factorial(order)
        output = term if output is None else output + term
    if output is None:
        raise ValueError("TaylorSeer cache is empty for the requested module.")
    return output


class TaylorSeerStrategy(FlexCacheStrategy):
    """TaylorSeer-Wan block cache adapted to the FlexCache strategy API."""

    MODULE_SELF_ATTN = "self-attention"
    MODULE_CROSS_ATTN = "cross-attention"
    MODULE_FFN = "ffn"

    def __init__(
        self,
        task,
        fresh_threshold: int = 5,
        max_order: int = 1,
        first_enhance: int = 1,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
    ):
        super().__init__()
        self.type = "taylorseer"
        self.tradeoff_score = 1.0 / max(1, int(fresh_threshold))
        self.num_steps = int(task.req.params.num_inference_steps)
        self.fresh_threshold = max(1, int(fresh_threshold))
        self.max_order = max(0, int(max_order))
        self.first_enhance = max(0, int(first_enhance))
        self.warmup_steps = max(0, int(warmup_steps))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.cache_counter = 0
        self.current_step: Optional[int] = None
        self.current_type = "full"
        self.activated_steps = [0]
        self._wrapped_blocks = []
        self._vis_records: Dict[int, int] = {}
        self._vis_max_step = -1

    def get_reuse_key(self, **kwargs) -> Optional[Any]:
        if self._current_type_for_step() != "Taylor":
            return None
        return self._cache_key(**kwargs)

    def reuse(self, cached_feature: Dict[int, torch.Tensor], distance: int, **kwargs) -> torch.Tensor:
        return _taylor_formula(cached_feature, distance)

    def get_store_key(self, **kwargs) -> Optional[Any]:
        return self._cache_key(**kwargs)

    def store(self, fresh_feature: torch.Tensor, **kwargs) -> Dict[int, torch.Tensor]:
        key = self._cache_key(**kwargs)
        previous = DiffusionBackend.flexcache.cache.get(key, {})
        updated: Dict[int, torch.Tensor] = {0: fresh_feature}
        distance = 1 if len(self.activated_steps) < 2 else max(1, self.activated_steps[-1] - self.activated_steps[-2])
        for order in range(self.max_order):
            if order not in previous or self._current_step_value() <= self.first_enhance - 2:
                break
            updated[order + 1] = (updated[order] - previous[order]) / distance
        return updated

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        if not hasattr(module, "blocks"):
            raise ValueError("TaylorSeer strategy expects a WanModel-like module with blocks.")

        self._wrapped_blocks = []
        for layer_idx, block in enumerate(module.blocks):
            if not hasattr(block, "_taylorseer_original_forward"):
                block._taylorseer_original_forward = block.forward
            block.forward = self._make_block_forward(block, layer_idx)
            self._wrapped_blocks.append(block)

        logger.info(
            "Module %s wrapped with TaylorSeer strategy: fresh_threshold=%d max_order=%d warmup=%d cooldown=%d",
            module.__class__.__name__,
            self.fresh_threshold,
            self.max_order,
            self.warmup_steps,
            self.cooldown_steps,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        blocks = self._wrapped_blocks or list(getattr(module, "blocks", []))
        for block in blocks:
            if hasattr(block, "_taylorseer_original_forward"):
                block.forward = block._taylorseer_original_forward
                delattr(block, "_taylorseer_original_forward")

        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            self._save_policy_ppm(debug_output_dir(run_output_dir))
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_unwrap",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            )
        logger.info("Module %s unwrapped from TaylorSeer strategy", module.__class__.__name__)

    def reset_state(self):
        self.cache_counter = 0
        self.current_step = None
        self.current_type = "full"
        self.activated_steps = [0]
        self._wrapped_blocks = []
        self._vis_records = {}
        self._vis_max_step = -1
        DiffusionBackend.flexcache.clear_cache()

    def _make_block_forward(self, block: torch.nn.Module, layer_idx: int):
        original_forward = block._taylorseer_original_forward

        @functools.wraps(original_forward)
        def forward_with_taylorseer(
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
        ):
            self._ensure_step_decision()
            if self.current_type == "Taylor" and self._has_layer_cache(layer_idx):
                return self._reuse_block(block, x=x, e=e, layer_idx=layer_idx)
            return self._compute_and_store_block(
                block=block,
                x=x,
                e=e,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                context_lens=context_lens,
                layer_idx=layer_idx,
            )

        return forward_with_taylorseer

    def _compute_and_store_block(
        self,
        block,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        layer_idx: int,
    ):
        e_chunks = self._modulation_chunks(block, e)

        y = block.self_attn(
            block.norm1(x).float() * (1 + e_chunks[1]) + e_chunks[0],
            seq_lens,
            grid_sizes,
            freqs,
        )
        self._store_module(layer_idx, self.MODULE_SELF_ATTN, y)
        with _fp32_autocast():
            x = x + y * e_chunks[2]

        y = block.cross_attn(block.norm3(x), context, context_lens)
        self._store_module(layer_idx, self.MODULE_CROSS_ATTN, y)
        x = x + y

        y = block.ffn(block.norm2(x).float() * (1 + e_chunks[4]) + e_chunks[3])
        self._store_module(layer_idx, self.MODULE_FFN, y)
        with _fp32_autocast():
            x = x + y * e_chunks[5]

        self._record_compute("compute", layer_idx)
        self._record_step_policy(self._current_step_value(), 1)
        return x

    def _reuse_block(self, block, x, e, layer_idx: int):
        e_chunks = self._modulation_chunks(block, e)
        distance = self._distance_from_anchor()

        sa = self.reuse(
            DiffusionBackend.flexcache.cache[self._cache_key(layer=layer_idx, module=self.MODULE_SELF_ATTN)],
            distance=distance,
        )
        ca = self.reuse(
            DiffusionBackend.flexcache.cache[self._cache_key(layer=layer_idx, module=self.MODULE_CROSS_ATTN)],
            distance=distance,
        )
        ffn = self.reuse(
            DiffusionBackend.flexcache.cache[self._cache_key(layer=layer_idx, module=self.MODULE_FFN)],
            distance=distance,
        )

        with _fp32_autocast():
            x = x + sa * e_chunks[2]
        x = x + ca
        with _fp32_autocast():
            x = x + ffn * e_chunks[5]

        self._record_compute("reuse", layer_idx)
        self._record_step_policy(self._current_step_value(), 2)
        return x

    def _store_module(self, layer_idx: int, module_name: str, feature: torch.Tensor) -> None:
        key = self.get_store_key(layer=layer_idx, module=module_name)
        DiffusionBackend.flexcache.cache[key] = self.store(
            fresh_feature=feature,
            layer=layer_idx,
            module=module_name,
        )
        if layer_idx == 0 and module_name == self.MODULE_SELF_ATTN:
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_store",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
                extra={"cache_key": "taylorseer"},
            )

    def _modulation_chunks(self, block, e):
        assert e.dtype == torch.float32
        with _fp32_autocast():
            chunks = (block.modulation + e).chunk(6, dim=1)
        assert chunks[0].dtype == torch.float32
        return chunks

    def _ensure_step_decision(self) -> None:
        step = self._current_step_value()
        if self.current_step == step:
            return

        self.current_step = step
        if step < self.warmup_steps or step >= max(0, self.num_steps - self.cooldown_steps):
            self.current_type = "full"
            self.cache_counter = 0
            self._append_anchor_step(step)
            return

        first_step = step < self.first_enhance
        fresh_interval = self.fresh_threshold
        if first_step or self.cache_counter == fresh_interval - 1:
            self.current_type = "full"
            self.cache_counter = 0
            self._append_anchor_step(step)
        else:
            self.current_type = "Taylor"
            self.cache_counter += 1

    def _append_anchor_step(self, step: int) -> None:
        if self.activated_steps[-1] != step:
            self.activated_steps.append(step)

    def _current_type_for_step(self) -> str:
        self._ensure_step_decision()
        return self.current_type

    def _current_step_value(self) -> int:
        task = DiffusionBackend.generator.current_task
        return int(task.buffer.current_step)

    def _branch_key(self) -> str:
        return "pos" if DiffusionBackend.cfg_type == CFGType.POS else "neg"

    def _cache_key(self, layer: int, module: str, **kwargs):
        return ("taylorseer", self._branch_key(), int(layer), str(module))

    def _distance_from_anchor(self) -> int:
        return max(0, self._current_step_value() - self.activated_steps[-1])

    def _has_layer_cache(self, layer_idx: int) -> bool:
        return all(
            self._cache_key(layer=layer_idx, module=module_name) in DiffusionBackend.flexcache.cache
            for module_name in (self.MODULE_SELF_ATTN, self.MODULE_CROSS_ATTN, self.MODULE_FFN)
        )

    def _record_compute(self, decision: str, layer_idx: int) -> None:
        DiffusionBackend.flexcache.record_compute(
            baseline_units=3.0,
            actual_units=0.0 if decision == "reuse" else 3.0,
            task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            scope="taylorseer_block",
            unit="block_modules",
            extra={
                "decision": decision,
                "step": self._current_step_value(),
                "branch": self._branch_key(),
                "layer": layer_idx,
            },
        )

    def _record_step_policy(self, step: int, code: int):
        if DiffusionBackend.cfg_type != CFGType.POS:
            return
        self._vis_records[step] = max(self._vis_records.get(step, 0), code)
        self._vis_max_step = max(self._vis_max_step, step)

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
            if code == 2:
                color = (255, 180, 40)
            elif code == 1:
                color = (40, 140, 255)
            else:
                color = (160, 160, 160)
            for yy in range(height):
                for xx in range(step * cell, (step + 1) * cell):
                    idx = (yy * width + xx) * 3
                    rgb[idx] = color[0]
                    rgb[idx + 1] = color[1]
                    rgb[idx + 2] = color[2]
        path = os.path.join(output_dir, "flexcache_taylorseer_policy.ppm")
        with open(path, "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            f.write(rgb)
