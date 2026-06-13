import functools
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.flexcache.modules import FluxCubicSelectiveForwardEngine, WanCubicSelectiveForwardEngine
from chitu_diffusion.modules.utils.flux import unpack_flux1_latents
from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend
from chitu_diffusion.runtime.output_layout import debug_output_dir

logger = logging.getLogger(__name__)


def _load_cubic_core():
    repo_root = Path(__file__).resolve().parents[3]
    cubic_root = repo_root / "third_party" / "Cubic"
    if not cubic_root.exists():
        cubic_root = repo_root / "refs" / "Cubic"
    if not cubic_root.exists():
        raise ImportError(f"Cubic reference package not found under {repo_root}/third_party/Cubic or {repo_root}/refs/Cubic")
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
class CubicModelParams:
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    num_layers: int = 30
    partition_mode: str = "wan13_832x480_uniform"
    uniform_square_min_splits: int = 8
    min_block_tokens: int = 1
    max_block_tokens: int = 1 << 30
    max_blocksize_candidates: int = 36
    uniform_common_factor_blocks: bool = False
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


@dataclass
class CubicFluxModelParams(CubicModelParams):
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    num_layers: int = 57
    partition_mode: str = "uniform_square"
    uniform_square_min_splits: int = 8
    min_block_size: int = 8
    max_block_size: int = 8
    alpha: float = 2.0
    beta: float = 4.0
    curvature_contrast_gamma: float = 1.8


@dataclass
class CubicWanModelParams(CubicModelParams):
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    num_layers: int = 30
    partition_mode: str = "wan13_832x480_uniform"
    wan13_image_h: int = 480
    wan13_image_w: int = 832
    wan13_block_image_h: int = 60
    wan13_block_image_w: int = 64
    min_block_size: int = 2
    max_block_size: int = 16


@dataclass
class CubicConfig(CubicModelParams):
    target_speedup: float = 2.0
    warmup_fraction: float = 0.15
    cooldown_fraction: float = 0.15
    anchor_interval: int = 8


class CubicStrategy(FlexCacheStrategy):
    """Cubic selective token forward as a FlexCache strategy."""

    def __init__(
        self,
        task,
        target_speedup: float,
        warmup_steps: int,
        cooldown_steps: int,
        tau_max: int,
        model_params: CubicModelParams,
    ):
        super().__init__()
        AdaptivePartitioner, CurvatureMonitor, Phase, SelectiveStepScheduler, UpdateFrequencyOptimizer = _load_cubic_core()
        total_steps = int(task.req.params.num_inference_steps)
        warmup_fraction = warmup_steps / max(total_steps, 1)
        cooldown_fraction = cooldown_steps / max(total_steps, 1)
        target_speedup = float(target_speedup)
        self.model_name = getattr(getattr(getattr(DiffusionBackend, "args", None), "models", None), "name", None)
        is_flux1 = self.model_name in {"Flux1-dev", "FLUX.1-dev"}
        self.config = CubicConfig(
            target_speedup=target_speedup,
            warmup_fraction=warmup_fraction,
            cooldown_fraction=cooldown_fraction,
            anchor_interval=int(tau_max),
            patch_size=model_params.patch_size,
            num_layers=model_params.num_layers,
            partition_mode=model_params.partition_mode,
            uniform_square_min_splits=model_params.uniform_square_min_splits,
            min_block_tokens=model_params.min_block_tokens,
            max_block_tokens=model_params.max_block_tokens,
            max_blocksize_candidates=model_params.max_blocksize_candidates,
            uniform_common_factor_blocks=model_params.uniform_common_factor_blocks,
            wan13_image_h=model_params.wan13_image_h,
            wan13_image_w=model_params.wan13_image_w,
            wan13_block_image_h=model_params.wan13_block_image_h,
            wan13_block_image_w=model_params.wan13_block_image_w,
            min_block_size=model_params.min_block_size,
            max_block_size=model_params.max_block_size,
            cv_threshold=model_params.cv_threshold,
            alpha=model_params.alpha,
            beta=model_params.beta,
            curvature_contrast_gamma=model_params.curvature_contrast_gamma,
        )
        self.type = "cubic"
        self.tradeoff_score = target_speedup
        self.total_steps = total_steps
        self.CubicPhase = Phase
        self.monitor = CurvatureMonitor(self.config)
        self.partitioner = AdaptivePartitioner(self.config)
        self.frequency_optimizer = UpdateFrequencyOptimizer(self.config)
        self.step_scheduler = SelectiveStepScheduler(self.config)
        self.selective_forward = (
            FluxCubicSelectiveForwardEngine(self.config) if is_flux1 else WanCubicSelectiveForwardEngine()
        )
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

        if self.model_name in {"Flux1-dev", "FLUX.1-dev"}:
            self._wrap_flux1_module(module, original_forward)
            return

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

    def _wrap_flux1_module(self, module: torch.nn.Module, original_forward) -> None:
        @functools.wraps(original_forward)
        def forward_with_cubic(
            hidden_states,
            encoder_hidden_states=None,
            pooled_projections=None,
            timestep=None,
            img_ids=None,
            txt_ids=None,
            guidance=None,
            joint_attention_kwargs=None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict=True,
            controlnet_blocks_repeat=False,
        ):
            if controlnet_block_samples is not None or controlnet_single_block_samples is not None:
                return original_forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                    joint_attention_kwargs=joint_attention_kwargs,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    return_dict=return_dict,
                    controlnet_blocks_repeat=controlnet_blocks_repeat,
                )

            step = self._current_step()
            phase = self._phase(step)
            self._prepare_step_plan(step, phase)
            branch_key = self._branch_key()
            output = self.selective_forward.forward(
                transformer=module,
                original_forward=original_forward,
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                step_plan=self.current_step_plan,
                cache_key=branch_key,
            )
            if self._should_monitor(step, phase):
                self._update_flux_anchor_state(
                    latents=hidden_states,
                    output=output,
                    img_ids=img_ids,
                    t=timestep,
                    step=step,
                )
            self._publish_cache_refs()
            self._record_compute_metric(step, phase, branch_key)
            self._record_step_trace(step, phase)
            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        module.forward = forward_with_cubic
        logger.info(
            "Module %s wrapped with Flux Cubic strategy: target_speedup=%.3f partition_mode=%s uniform_square_min_splits=%d",
            module.__class__.__name__,
            self.config.target_speedup,
            self.config.partition_mode,
            self.config.uniform_square_min_splits,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        if hasattr(self.selective_forward, "restore_attn_processors"):
            self.selective_forward.restore_attn_processors(module)
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

    def _update_flux_anchor_state(self, latents, output, img_ids, t, step: int):
        latents_unpacked = self._unpack_flux_sequence(latents, img_ids)
        output_unpacked = self._unpack_flux_sequence(output, img_ids)
        t_unit = self._normalize_t_to_unit_interval(t)
        curvature_map = self.monitor.update(
            x=latents_unpacked.unsqueeze(2),
            v=output_unpacked.unsqueeze(2),
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
            "[flexcache-cubic-flux] step=%d curvature=%s blocks=%d tau_min=%d tau_max=%d achieved_speedup=%.3f next_anchor_step=%d",
            step,
            tuple(curvature_map.shape),
            len(self.current_partition.blocks),
            self.current_frequency_plan.min_interval,
            self.current_frequency_plan.max_interval,
            self.current_frequency_plan.achieved_speedup,
            self.next_anchor_step,
        )

    @staticmethod
    def _unpack_flux_sequence(sequence: torch.Tensor, img_ids: torch.Tensor | None) -> torch.Tensor:
        if img_ids is not None:
            ids = img_ids[0] if img_ids.ndim == 3 else img_ids
            if ids.shape[1] >= 3:
                h_tokens = int(ids[:, 1].max().item()) + 1
                w_tokens = int(ids[:, 2].max().item()) + 1
                return unpack_flux1_latents(sequence, height=h_tokens * 16, width=w_tokens * 16, vae_scale_factor=16)

        seq_len = int(sequence.shape[1])
        side = int(seq_len**0.5)
        if side * side != seq_len:
            raise ValueError(f"Flux Cubic expects square token grid when img_ids is missing, got seq_len={seq_len}")
        return unpack_flux1_latents(sequence, height=side * 16, width=side * 16, vae_scale_factor=16)

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
