import functools
import os
from logging import getLogger
from typing import Optional

import torch

from chitu_diffusion.core.models.backbone import detach_backbone_value
from chitu_diffusion.core.distributed.parallel_state import get_cp_group
from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend
from chitu_diffusion.runtime.output_layout import debug_output_dir

logger = getLogger(__name__)


class BlockDanceStrategy(FlexCacheStrategy):
    """BlockDance-style train-free layerwise cache.

    The policy runs full DiT forward during the unstable early denoising steps.
    In the active window, every group of N steps starts with a cache step that
    stores the hidden state after a shallow/mid block boundary. The remaining
    group steps reuse that hidden state and run only deeper blocks.
    """

    def __init__(
        self,
        task,
        warmup_steps: int,
        cooldown_steps: int,
        boundary_block: Optional[int] = None,
        group_size: Optional[int] = None,
        start_fraction: float = 0.40,
        end_fraction: float = 0.95,
    ):
        super().__init__()
        self.type = "blockdance"
        self.total_steps = int(task.req.params.num_inference_steps)
        self.boundary_block = 20 if boundary_block is None else max(0, int(boundary_block))
        self.group_size = max(1, int(group_size if group_size is not None else 2))
        self.start_step = max(int(warmup_steps), int(round(self.total_steps * float(start_fraction))))
        self.end_step = min(
            self.total_steps - int(cooldown_steps),
            int(round(self.total_steps * float(end_fraction))),
        )
        self.start_step = max(0, min(self.total_steps, self.start_step))
        self.end_step = max(self.start_step, min(self.total_steps, self.end_step))
        self.tradeoff_score = 1.0 / self.group_size
        self._vis_records = {}
        self._vis_max_step = -1

    def _branch_key(self) -> str:
        cfg_branch = "pos" if DiffusionBackend.cfg_type == CFGType.POS else "neg"
        try:
            cp_group = get_cp_group()
            cp_rank = cp_group.rank_in_group if cp_group.group_size > 1 else 0
        except AssertionError:
            cp_rank = 0
        return f"{cfg_branch}_cp{cp_rank}"

    def _cache_key(self):
        return ("blockdance", self._branch_key(), self.boundary_block)

    def _current_step(self) -> int:
        task = DiffusionBackend.generator.current_task
        return int(task.buffer.current_step)

    def _active_group_offset(self, step: int) -> Optional[int]:
        if step < self.start_step or step >= self.end_step:
            return None
        return (step - self.start_step) % self.group_size

    def get_reuse_key(self, **kwargs):
        offset = self._active_group_offset(self._current_step())
        if offset is None or offset == 0:
            return None
        return self._cache_key()

    def reuse(self, cached_feature: torch.Tensor, **kwargs):
        return cached_feature

    def get_store_key(self, **kwargs):
        offset = self._active_group_offset(self._current_step())
        if offset == 0:
            return self._cache_key()
        return None

    def store(self, fresh_feature: torch.Tensor, **kwargs):
        return detach_backbone_value(fresh_feature)

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        if not hasattr(module, "block_compute"):
            raise ValueError("BlockDance layer strategy expects a module with block_compute.")
        if not hasattr(module, "_blockdance_original_block_compute"):
            module._blockdance_original_block_compute = module.block_compute
        original_block_compute = module._blockdance_original_block_compute
        if not hasattr(module, "backbone_blocks"):
            raise ValueError(f"{module.__class__.__name__} does not implement the backbone block API.")
        blocks = module.backbone_blocks()
        block_count = len(blocks)
        boundary_index = min(self.boundary_block, max(0, block_count - 1))
        reuse_start_index = min(boundary_index + 1, block_count)

        @functools.wraps(original_block_compute)
        def block_compute_with_blockdance(block_info, state):
            step = self._current_step()
            offset = self._active_group_offset(step)
            cache_key = self._cache_key()
            task_id = getattr(DiffusionBackend.generator.current_task, "task_id", None)
            layer_id = int(block_info.index)

            if offset is not None and offset != 0 and cache_key in DiffusionBackend.flexcache.cache:
                if layer_id < reuse_start_index:
                    if layer_id == 0:
                        self._record_compute(
                            baseline_units=float(block_count),
                            actual_units=float(block_count - reuse_start_index),
                            task_id=task_id,
                            step=step,
                            decision="reuse",
                            boundary_index=boundary_index,
                            reuse_start_index=reuse_start_index,
                        )
                        self._record_step_policy(step, 2)
                    if layer_id == boundary_index:
                        return module.backbone_restore_cached_state(
                            state,
                            self.reuse(DiffusionBackend.flexcache.cache[cache_key]),
                        )
                    return state
                return original_block_compute(block_info, state)

            should_store = offset == 0
            state = original_block_compute(block_info, state)
            if should_store and layer_id == boundary_index:
                DiffusionBackend.flexcache.cache[cache_key] = self.store(module.backbone_cache_state(state))
                DiffusionBackend.flexcache.record_cache_memory(
                    "flexcache_store",
                    task_id=task_id,
                    extra={"cache_key": str(cache_key)},
                )
                self._record_step_policy(step, 3)
            elif layer_id == 0:
                self._record_step_policy(step, 1)

            if layer_id == 0:
                self._record_compute(
                    baseline_units=float(block_count),
                    actual_units=float(block_count),
                    task_id=task_id,
                    step=step,
                    decision="cache" if should_store else "compute",
                    boundary_index=boundary_index,
                    reuse_start_index=reuse_start_index,
                )
            return state

        module.block_compute = block_compute_with_blockdance
        logger.info(
            "Module %s wrapped with BlockDance layer strategy: boundary_block=%d group_size=%d active=[%d,%d)",
            module.__class__.__name__,
            boundary_index,
            self.group_size,
            self.start_step,
            self.end_step,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        if hasattr(module, "_blockdance_original_block_compute"):
            module.block_compute = module._blockdance_original_block_compute
            delattr(module, "_blockdance_original_block_compute")
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            self._save_policy_ppm(debug_output_dir(run_output_dir), "flexcache_blockdance_policy.ppm")
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_unwrap",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            )
            DiffusionBackend.flexcache.flush_cache_memory_events()
        logger.info("Module %s unwrapped from BlockDance layer strategy", module.__class__.__name__)

    def reset_state(self):
        self._vis_records = {}
        self._vis_max_step = -1
        DiffusionBackend.flexcache.clear_cache()

    def _record_compute(
        self,
        *,
        baseline_units: float,
        actual_units: float,
        task_id: Optional[str],
        step: int,
        decision: str,
        boundary_index: int,
        reuse_start_index: int,
    ):
        DiffusionBackend.flexcache.record_compute(
            baseline_units=baseline_units,
            actual_units=actual_units,
            task_id=task_id,
            scope="blockdance_layer",
            unit="transformer_block",
            extra={
                "decision": decision,
                "step": int(step),
                "branch": self._branch_key(),
                "boundary_block": int(boundary_index),
                "reuse_start_block": int(reuse_start_index),
                "group_size": int(self.group_size),
            },
        )

    def _record_step_policy(self, step: int, code: int):
        if DiffusionBackend.cfg_type != CFGType.POS:
            return
        self._vis_records[step] = max(self._vis_records.get(step, 0), code)
        self._vis_max_step = max(self._vis_max_step, step)

    def _save_policy_ppm(self, output_dir: str, filename: str):
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
                color = (80, 200, 120)
            elif code == 2:
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
        path = os.path.join(output_dir, filename)
        with open(path, "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            f.write(rgb)
