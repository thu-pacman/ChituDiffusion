import functools
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.flexcache.modules import WanCubicSelectiveForwardEngine
from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend
from chitu_diffusion.runtime.output_layout import debug_output_dir

logger = logging.getLogger(__name__)


def _load_cubic_core():
    cubic_root = Path(__file__).resolve().parents[3] / "third_party" / "Cubic"
    if not cubic_root.exists():
        raise ImportError(f"third_party/Cubic not found at {cubic_root}")
    cubic_root_str = str(cubic_root)
    if cubic_root_str not in sys.path:
        sys.path.insert(0, cubic_root_str)

    from cubic_core import (  # type: ignore
        AdaptivePartitioner,
        CurvatureMonitor,
        Phase,
        SelectiveStepScheduler,
        UpdateFrequencyOptimizer,
    )

    return AdaptivePartitioner, CurvatureMonitor, Phase, SelectiveStepScheduler, UpdateFrequencyOptimizer


@dataclass
class CubicWanConfig:
    target_speedup: float = 2.0
    warmup_fraction: float = 0.15
    cooldown_fraction: float = 0.15
    anchor_interval: int = 8
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    num_layers: int = 30
    partition_mode: str = "wan13_832x480_uniform"
    wan13_image_h: int = 480
    wan13_image_w: int = 832
    wan13_block_image_h: int = 60
    wan13_block_image_w: int = 64
    min_block_size: int = 2
    max_block_size: int = 16
    cv_threshold: float = 0.5
    alpha: float = 1.0
    beta: float = 2.0
    curvature_contrast_gamma: float = 1.0


class CubicStrategy(FlexCacheStrategy):
    """Cubic-WAN as a FlexCache strategy."""

    def __init__(self, task, cache_ratio: float, warmup_steps: int, cooldown_steps: int, tau_max: int, **kwargs):
        super().__init__()
        AdaptivePartitioner, CurvatureMonitor, Phase, SelectiveStepScheduler, UpdateFrequencyOptimizer = _load_cubic_core()
        total_steps = int(task.req.params.num_inference_steps)
        warmup_fraction = float(kwargs.get("warmup_fraction", warmup_steps / max(total_steps, 1)))
        cooldown_fraction = float(kwargs.get("cooldown_fraction", cooldown_steps / max(total_steps, 1)))
        target_speedup = float(kwargs.get("target_speedup", max(1.0, 1.0 / max(1.0 - float(cache_ratio), 1e-6))))
        patch_size = tuple(int(v) for v in kwargs.get("patch_size", (1, 2, 2)))
        self.config = CubicWanConfig(
            target_speedup=target_speedup,
            warmup_fraction=warmup_fraction,
            cooldown_fraction=cooldown_fraction,
            anchor_interval=int(kwargs.get("anchor_interval", tau_max)),
            patch_size=patch_size,
            num_layers=int(kwargs.get("num_layers", 30)),
            partition_mode=str(kwargs.get("partition_mode", "wan13_832x480_uniform")),
            wan13_image_h=int(kwargs.get("wan13_image_h", 480)),
            wan13_image_w=int(kwargs.get("wan13_image_w", 832)),
            wan13_block_image_h=int(kwargs.get("wan13_block_image_h", 60)),
            wan13_block_image_w=int(kwargs.get("wan13_block_image_w", 64)),
            min_block_size=int(kwargs.get("min_block_size", 2)),
            max_block_size=int(kwargs.get("max_block_size", 16)),
            cv_threshold=float(kwargs.get("cv_threshold", 0.5)),
            alpha=float(kwargs.get("alpha", 1.0)),
            beta=float(kwargs.get("beta", 2.0)),
            curvature_contrast_gamma=float(kwargs.get("curvature_contrast_gamma", 1.0)),
        )
        self.type = "cubic"
        self.tradeoff_score = target_speedup
        self.total_steps = total_steps
        self.CubicPhase = Phase
        self.monitor = CurvatureMonitor(self.config)
        self.partitioner = AdaptivePartitioner(self.config)
        self.frequency_optimizer = UpdateFrequencyOptimizer(self.config)
        self.step_scheduler = SelectiveStepScheduler(self.config)
        self.selective_forward = WanCubicSelectiveForwardEngine()
        self.current_partition = None
        self.current_frequency_plan = None
        self.current_step_plan = None
        self.next_anchor_step = None
        self.current_anchor_gap = 0
        self.step_trace = []
        self._phase_step = None
        self._phase_value = None

    def get_reuse_key(self, **kwargs):
        return None

    def reuse(self, **kwargs):
        return None

    def get_store_key(self, **kwargs):
        return None

    def store(self, **kwargs) -> None:
        return None

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        if not hasattr(module, "_original_forward"):
            module._original_forward = module.forward
        original_forward = module._original_forward

        @functools.wraps(original_forward)
        def forward_with_cubic(x, t, context, seq_len, clip_fea=None, y=None):
            step = self._current_step()
            phase = self._phase(step)
            self._prepare_step_plan(step, phase)
            branch_key = self._branch_key()
            output = self.selective_forward.forward(
                module=module,
                original_forward=original_forward,
                x=x,
                t=t,
                context=context,
                seq_len=seq_len,
                branch_key=branch_key,
                step_plan=self.current_step_plan,
                clip_fea=clip_fea,
                y=y,
            )
            self._publish_cache_refs()
            self._record_compute_metric(step, phase, branch_key)
            self._record_step_trace(step, phase)
            return output

        module.forward = forward_with_cubic
        logger.info(
            "Module %s wrapped with Cubic strategy: target_speedup=%.3f warmup_fraction=%.3f cooldown_fraction=%.3f anchor_interval=%d",
            module.__class__.__name__,
            self.config.target_speedup,
            self.config.warmup_fraction,
            self.config.cooldown_fraction,
            self.config.anchor_interval,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            delattr(module, "_original_forward")
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            self._save_step_trace(debug_output_dir(run_output_dir))
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_unwrap",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            )
        logger.info("Module %s unwrapped from Cubic strategy", module.__class__.__name__)

    def reset_state(self):
        self.monitor.reset()
        self.selective_forward.reset()
        self.current_partition = None
        self.current_frequency_plan = None
        self.current_step_plan = None
        self.next_anchor_step = None
        self.current_anchor_gap = 0
        self.step_trace = []
        self._phase_step = None
        self._phase_value = None
        DiffusionBackend.flexcache.clear_cache()
        self._publish_cache_refs()

    def _current_step(self) -> int:
        task = DiffusionBackend.generator.current_task
        return int(task.buffer.current_step)

    def _branch_key(self) -> str:
        return "pos" if DiffusionBackend.cfg_type == CFGType.POS else "neg"

    def _phase(self, step: int):
        if self._phase_step == step and self._phase_value is not None:
            return self._phase_value
        warmup_steps = max(0, int(self.total_steps * self.config.warmup_fraction))
        cooldown_steps = max(0, int(self.total_steps * self.config.cooldown_fraction))
        cooldown_start = max(0, self.total_steps - cooldown_steps)
        if step < warmup_steps:
            phase = self.CubicPhase.WARMUP
        elif step >= cooldown_start:
            phase = self.CubicPhase.COOLDOWN
        else:
            if self.next_anchor_step is None:
                self.next_anchor_step = warmup_steps
            phase = self.CubicPhase.CRUISE_ANCHOR if step == self.next_anchor_step else self.CubicPhase.CRUISE_SELECTIVE
        self._phase_step = step
        self._phase_value = phase
        return phase

    def _prepare_step_plan(self, step: int, phase):
        if self.current_partition is None or self.current_frequency_plan is None:
            self.current_step_plan = None
            return
        warmup_steps = max(0, int(self.total_steps * self.config.warmup_fraction))
        self.current_step_plan = self.step_scheduler.build_step_plan(
            partition=self.current_partition,
            block_intervals=self.current_frequency_plan.block_intervals,
            phase=phase,
            step_idx=step,
            selective_step_idx=max(0, step - warmup_steps),
        )

    def _should_monitor(self, step: int, phase) -> bool:
        warmup_steps = max(0, int(self.total_steps * self.config.warmup_fraction))
        return (warmup_steps > 0 and phase == self.CubicPhase.WARMUP and step == warmup_steps - 1) or phase == self.CubicPhase.CRUISE_ANCHOR

    def observe_guided_output(self, x, output, t, step: int | None = None) -> None:
        step = self._current_step() if step is None else int(step)
        phase = self._phase(step)
        if self._should_monitor(step, phase):
            self._update_anchor_state(x=x, output=output, t=t, step=step)

    def _update_anchor_state(self, x, output, t, step: int):
        x_tensor = x[0] if isinstance(x, list) else x
        t_unit = self._normalize_t_to_unit_interval(t)
        curvature_map = self.monitor.update(
            x=x_tensor.unsqueeze(0),
            v=output.unsqueeze(0),
            t=t_unit,
            step_idx=step,
        )
        if curvature_map is None:
            self.current_anchor_gap = 1
            self.next_anchor_step = step + self.current_anchor_gap
            return

        ft, ht, wt = curvature_map.shape
        self.current_partition = self.partitioner.partition(Ft=ft, Ht=ht, Wt=wt, curvature_map=curvature_map)
        block_curvatures = self.monitor.get_block_curvatures(self.current_partition, curvature_map=curvature_map)
        self.current_frequency_plan = self.frequency_optimizer.optimize(
            partition=self.current_partition,
            block_curvatures=block_curvatures,
        )
        self.current_anchor_gap = max(1, 2 * int(self.current_frequency_plan.max_interval))
        self.next_anchor_step = min(self.total_steps - 1, step + self.current_anchor_gap)
        self.current_step_plan = self.step_scheduler.build_step_plan(
            partition=self.current_partition,
            block_intervals=self.current_frequency_plan.block_intervals,
            phase=self.CubicPhase.CRUISE_ANCHOR,
            step_idx=step,
            selective_step_idx=max(0, step - int(self.total_steps * self.config.warmup_fraction)),
        )
        logger.info(
            "[flexcache-cubic] step=%d curvature=%s blocks=%d tau_min=%d tau_max=%d achieved_speedup=%.3f next_anchor_step=%d",
            step,
            tuple(curvature_map.shape),
            len(self.current_partition.blocks),
            self.current_frequency_plan.min_interval,
            self.current_frequency_plan.max_interval,
            self.current_frequency_plan.achieved_speedup,
            self.next_anchor_step,
        )

    def _record_step_trace(self, step: int, phase):
        if self._branch_key() != "pos":
            return
        active_tokens = total_tokens = 0
        full_compute = True
        if self.current_step_plan is not None:
            active_tokens = int(self.current_step_plan.active_token_indices.numel())
            frozen_tokens = int(self.current_step_plan.frozen_token_indices.numel())
            total_tokens = active_tokens + frozen_tokens
            full_compute = bool(self.current_step_plan.is_full_compute)
        elif self.current_partition is not None:
            total_tokens = int(self.current_partition.total_tokens)
            active_tokens = total_tokens
        forward_mode = self.selective_forward.last_forward_mode
        if forward_mode == "cached_residual":
            actual_tokens = 0
        elif forward_mode == "selective" and not full_compute:
            actual_tokens = active_tokens
        else:
            actual_tokens = total_tokens
        self.step_trace.append(
            {
                "step": int(step),
                "phase": phase.value,
                "active_tokens": int(active_tokens),
                "actual_tokens": int(actual_tokens),
                "total_tokens": int(total_tokens),
                "full_compute": bool(full_compute),
                "forward_mode": forward_mode,
                "anchor_gap": int(self.current_anchor_gap),
                "next_anchor_step": None if self.next_anchor_step is None else int(self.next_anchor_step),
            }
        )

    def _record_compute_metric(self, step: int, phase, branch_key: str):
        total_tokens = 0
        active_tokens = 0
        full_compute = True
        if self.current_step_plan is not None:
            active_tokens = int(self.current_step_plan.active_token_indices.numel())
            frozen_tokens = int(self.current_step_plan.frozen_token_indices.numel())
            total_tokens = active_tokens + frozen_tokens
            full_compute = bool(self.current_step_plan.is_full_compute)
        elif self.current_partition is not None:
            total_tokens = int(self.current_partition.total_tokens)
            active_tokens = total_tokens
        if total_tokens <= 0:
            return
        forward_mode = self.selective_forward.last_forward_mode
        if forward_mode == "cached_residual":
            actual_tokens = 0
            decision = "cached_residual"
        elif forward_mode == "selective" and not full_compute:
            actual_tokens = active_tokens
            decision = "selective"
        else:
            actual_tokens = total_tokens
            decision = "compute"
        DiffusionBackend.flexcache.record_compute(
            baseline_units=float(total_tokens),
            actual_units=float(actual_tokens),
            task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            scope="token_forward",
            unit="tokens",
            extra={
                "decision": decision,
                "step": int(step),
                "phase": phase.value,
                "branch": branch_key,
                "active_tokens": int(actual_tokens),
                "total_tokens": int(total_tokens),
                "forward_mode": forward_mode,
            },
        )

    def _publish_cache_refs(self):
        if DiffusionBackend.flexcache is None:
            return
        DiffusionBackend.flexcache.cache["cubic_k_cache"] = self.selective_forward.k_cache
        DiffusionBackend.flexcache.cache["cubic_v_cache"] = self.selective_forward.v_cache
        DiffusionBackend.flexcache.cache["cubic_residual_cache"] = self.selective_forward.residual_cache

    def _save_step_trace(self, output_dir: str):
        import json

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "flexcache_cubic_step_trace.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.step_trace, f, indent=2)

    @staticmethod
    def _normalize_t_to_unit_interval(t) -> float:
        if isinstance(t, torch.Tensor):
            t_val = float(t.detach().reshape(-1)[0].item())
        else:
            t_val = float(t)
        if t_val > 1.0:
            args = getattr(DiffusionBackend, "args", None)
            sampler = getattr(getattr(args, "models", None), "sampler", None)
            denom = float(getattr(sampler, "num_train_timesteps", 1000) - 1)
            t_val = t_val / max(1.0, denom)
        return max(0.0, min(1.0, t_val))
