import torch
import os
import time
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

from chitu_core.distributed.parallel_state import get_cfg_group, get_cp_group, get_up_group
from chitu_diffusion.backend import CFGType, DiffusionBackend
from chitu_diffusion.bench import MagLogger, Timer
from chitu_diffusion.flex_cache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.task import DiffusionTask
from chitu_diffusion.utils.shared_utils import (
    async_ring_p2p_commit,
    async_ring_p2p_wait_and_update,
    squeeze_and_transpose,
    update_out_and_lse,
)


logger = getLogger(__name__)


def get_timestep() -> int:
    return DiffusionBackend.generator.current_task.buffer.current_step


@dataclass
class AttentionState:
    out: Optional[torch.Tensor] = None  # [b,s,n,d]
    lse: Optional[torch.Tensor] = None  # [b,s,n,1]

    def is_empty(self) -> bool:
        return self.lse is None

    def update(self, block_out: torch.Tensor, block_lse: torch.Tensor):
        self.out, self.lse = update_out_and_lse(self.out, self.lse, block_out, block_lse)

    @staticmethod
    def merge(state1: "AttentionState", state2: "AttentionState") -> "AttentionState":
        if state2.is_empty():
            return state1
        input_lse = squeeze_and_transpose(state2.lse)
        state1.update(state2.out, input_lse)
        return state1


class DiTangoV3Strategy(FlexCacheStrategy):
    """
    DiTango v3 strategy:
    - Partition CP ranks into groups.
    - Decide compute/reuse for each group by ASE.
    """

    DEFAULT_ASE_THRESHOLD = 0.02
    DEFAULT_ANCHOR_INTERVAL = 5
    DEFAULT_TAU_MAX = 8
    DEFAULT_INTRA_GROUP_SIZE_LIMIT = 1
    DEFAULT_ENABLE_DYNAMIC_COMPOSE = False
    DEFAULT_WARMUP_STEPS = 5
    DEFAULT_COOLDOWN_STEPS = 5
    DEFAULT_DEBUG_CACHE_IO = True
    DEFAULT_ANCHOR_LOG_LAYERS = (28,)

    def __init__(
        self,
        task: Optional[DiffusionTask] = None,
        cache_ratio: Optional[float] = None,
        ase_threshold: Optional[float] = None,
        anchor_interval: Optional[int] = None,
        tau_max: Optional[int] = None,
        intra_group_size_limit: Optional[int] = None,
        enable_dynamic_compose: Optional[bool] = None,
        warmup_steps: Optional[int] = None,
        cooldown_steps: Optional[int] = None,
    ):
        super().__init__()
        if ase_threshold is None:
            ase_threshold = self.DEFAULT_ASE_THRESHOLD
        if anchor_interval is None:
            anchor_interval = self.DEFAULT_ANCHOR_INTERVAL
        if tau_max is None:
            tau_max = self.DEFAULT_TAU_MAX
        if intra_group_size_limit is None:
            intra_group_size_limit = self.DEFAULT_INTRA_GROUP_SIZE_LIMIT
        if enable_dynamic_compose is None:
            enable_dynamic_compose = self.DEFAULT_ENABLE_DYNAMIC_COMPOSE
        if warmup_steps is None:
            warmup_steps = self.DEFAULT_WARMUP_STEPS
        if cooldown_steps is None:
            cooldown_steps = self.DEFAULT_COOLDOWN_STEPS
        if cache_ratio is None:
            cache_ratio = 0.5

        self.type = "ditango"
        self.tradeoff_score = float(ase_threshold)
        self.task = task
        self.cache_ratio = float(max(0.0, min(1.0, cache_ratio)))

        self.ase_threshold = float(ase_threshold)
        self.anchor_interval = max(1, int(anchor_interval))
        self.tau_max = max(1, int(tau_max))
        self.intra_group_size_limit = max(1, int(intra_group_size_limit))
        self.enable_dynamic_compose = bool(enable_dynamic_compose)
        self.warmup_steps = max(1, int(warmup_steps))
        self.cooldown_steps = max(1, int(cooldown_steps))
        self.debug_cache_io = self.DEFAULT_DEBUG_CACHE_IO
        self.anchor_log_layers = set(self.DEFAULT_ANCHOR_LOG_LAYERS)
        self.anchor_rel_err_threshold = 0.2 + 0.3 * self.cache_ratio
        self.total_layers = None
        self.current_local_as_comp = None

        self.reset_state()

    def _is_main_rank(self) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _get_output_dir(self) -> str:
        # Prefer per-task video output directory so visualization is colocated with mp4.
        task = DiffusionBackend.generator.current_task
        if task is not None and task.req is not None and task.req.params is not None:
            if task.req.params.save_dir:
                return task.req.params.save_dir

        if self.task is not None and self.task.req is not None and self.task.req.params is not None:
            if self.task.req.params.save_dir:
                return self.task.req.params.save_dir

        env_output = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if env_output:
            return env_output

        args = getattr(DiffusionBackend, "args", None)
        if args is not None:
            output_cfg = getattr(args, "output", None)
            if output_cfg is not None:
                root_dir = getattr(output_cfg, "root_dir", None)
                if root_dir:
                    return str(root_dir)

        return "./outputs"

    def record_step_layer_stats(
        self,
        layer_id: int,
        step: int,
        plan: Dict[int, bool],
        is_warmup_or_cooldown: bool,
        is_anchor: bool,
    ):
        branch = self._branch_key()
        ratio = 1.0
        if plan:
            ratio = float(sum(1 for v in plan.values() if v)) / float(len(plan))
        key = (step, layer_id)
        self.compute_ratio_records[branch][key] = ratio
        for group_id, should_compute in plan.items():
            self.group_decision_records[branch][(step, layer_id, int(group_id))] = bool(should_compute)
        if is_warmup_or_cooldown:
            phase = "warmup_cooldown"
        elif is_anchor:
            phase = "anchor"
        else:
            phase = "normal"
        self.phase_records[branch][key] = phase
        self.max_step_seen[branch] = max(self.max_step_seen[branch], step)

    def _save_compute_ratio_heatmap(self, output_dir: str, branch: str):
        if branch != "pos":
            return

        if not self.compute_ratio_records[branch]:
            return

        os.makedirs(output_dir, exist_ok=True)
        max_step = self.max_step_seen[branch]
        max_layer = self.total_layers - 1 if self.total_layers is not None else max(
            layer for (_, layer) in self.compute_ratio_records[branch].keys()
        )

        # 2D grid: rows are layers (y), columns are timesteps (x).
        cell = 12
        width = (max_step + 1) * cell
        height = (max_layer + 1) * cell
        rgb = bytearray(width * height * 3)

        for step in range(max_step + 1):
            for layer in range(max_layer + 1):
                key = (step, layer)
                phase = self.phase_records[branch].get(key, "missing")
                ratio = self.compute_ratio_records[branch].get(key, 0.0)

                if phase == "warmup_cooldown":
                    color = (160, 160, 160)  # gray
                elif phase == "anchor":
                    color = (0, 200, 0)  # green
                elif phase == "missing":
                    color = (230, 230, 230)
                else:
                    # Higher compute ratio => darker blue.
                    v = int(max(0, min(255, (1.0 - ratio) * 255.0)))
                    color = (v, v, 255)

                for yy in range(layer * cell, (layer + 1) * cell):
                    for xx in range(step * cell, (step + 1) * cell):
                        idx = (yy * width + xx) * 3
                        rgb[idx] = color[0]
                        rgb[idx + 1] = color[1]
                        rgb[idx + 2] = color[2]

        ppm_path = os.path.join(output_dir, "ditangov3_compute_ratio_pos.ppm")
        with open(ppm_path, "wb") as f:
            header = f"P6\n{width} {height}\n255\n".encode("ascii")
            f.write(header)
            f.write(bytes(rgb))

    def _save_group_decision_overview(self, output_dir: str, branch: str):
        if branch != "pos":
            return

        records = self.group_decision_records.get(branch, {})
        if not records:
            return

        os.makedirs(output_dir, exist_ok=True)
        max_step = self.max_step_seen.get(branch, 0)

        layers = sorted({layer for (_, layer, _) in records.keys()})
        if not layers:
            return

        cell = 10
        layer_sep = 4

        layer_groups = []
        for layer_id in layers:
            groups = sorted({group for (_, layer, group) in records.keys() if layer == layer_id})
            if groups:
                layer_groups.append((layer_id, groups))

        if not layer_groups:
            return

        width = (max_step + 1) * cell
        total_rows = sum(len(groups) for _, groups in layer_groups)
        sep_count = max(0, len(layer_groups) - 1)
        height = total_rows * cell + sep_count * layer_sep
        rgb = bytearray(width * height * 3)

        def fill_rect(x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]):
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    idx = (yy * width + xx) * 3
                    rgb[idx] = color[0]
                    rgb[idx + 1] = color[1]
                    rgb[idx + 2] = color[2]

        y_offset = 0
        for layer_idx, (layer_id, groups) in enumerate(layer_groups):
            group_to_row = {group_id: row for row, group_id in enumerate(groups)}

            for step in range(max_step + 1):
                phase = self.phase_records[branch].get((step, layer_id), "missing")
                x0 = step * cell
                x1 = (step + 1) * cell

                for group_id in groups:
                    decision_key = (step, layer_id, group_id)
                    has_decision = decision_key in records
                    should_compute = bool(records.get(decision_key, False))

                    if phase == "warmup_cooldown":
                        color = (160, 160, 160)  # gray
                    elif phase == "anchor":
                        color = (0, 200, 0)  # green
                    elif not has_decision:
                        color = (230, 230, 230)  # missing
                    else:
                        # normal phase: compute=orange, reuse=blue
                        color = (255, 140, 0) if should_compute else (70, 130, 255)

                    row = group_to_row[group_id]
                    y0 = y_offset + row * cell
                    y1 = y_offset + (row + 1) * cell
                    fill_rect(x0, y0, x1, y1, color)

                    # Thin horizontal split between groups for readability.
                    if row > 0:
                        split_y = y0
                        fill_rect(x0, split_y, x1, min(split_y + 1, y1), (245, 245, 245))

                # Vertical timeline grid: minor every 5 steps, major every 10.
                if step % 10 == 0:
                    fill_rect(x0, y_offset, min(x0 + 2, x1), y_offset + len(groups) * cell, (25, 25, 25))
                elif step % 5 == 0:
                    fill_rect(x0, y_offset, min(x0 + 1, x1), y_offset + len(groups) * cell, (90, 90, 90))

            y_offset += len(groups) * cell
            if layer_idx + 1 != len(layer_groups):
                # Thick separator between layers.
                fill_rect(0, y_offset, width, y_offset + layer_sep, (35, 35, 35))
                y_offset += layer_sep

        ppm_path = os.path.join(output_dir, f"ditangov3_group_overview_{branch}.ppm")
        with open(ppm_path, "wb") as f:
            header = f"P6\n{width} {height}\n255\n".encode("ascii")
            f.write(header)
            f.write(bytes(rgb))

    def dump_compute_reuse_visualization(self):
        if not self._is_main_rank():
            return

        output_dir = self._get_output_dir()
        self._save_group_decision_overview(output_dir, "pos")
        logger.info(
            "[DiTangoV3] Saved group overview visualization to "
            f"{output_dir}/ditangov3_group_overview_pos.ppm"
        )

    def _should_log_cache_io(self, layer_id: int) -> bool:
        if not self.debug_cache_io:
            return False
        if layer_id != 0:
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _should_log_anchor_stats(self, layer_id: int) -> bool:
        if not self.debug_cache_io:
            return False
        if layer_id not in self.anchor_log_layers:
            return False
        if self._branch_key() != "pos":
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _get_group_local_ref_norm(self, layer_id: int, group_id: int) -> float:
        group_key = self._group_key(layer_id, group_id)
        ref = self.group_ref_local_out_comp.get(group_key, None)
        if ref is None:
            return float("nan")
        return float(torch.norm(ref).item())

    @staticmethod
    def _state_digest(state: AttentionState) -> str:
        if state is None or state.is_empty():
            return "empty"
        out = state.out
        lse = state.lse
        out_norm = float(torch.norm(out.float()).item())
        lse_norm = float(torch.norm(lse.float()).item())
        return (
            f"out={tuple(out.shape)} lse={tuple(lse.shape)} "
            f"out_norm={out_norm:.6f} lse_norm={lse_norm:.6f}"
        )

    def _branch_key(self) -> str:
        return "pos" if DiffusionBackend.cfg_type == CFGType.POS else "neg"

    def _group_key(self, layer_id: int, group_id: int) -> str:
        return f"{self._branch_key()}_{layer_id}_group_{group_id}"

    def _anchor_scope_key(self, layer_id: int) -> str:
        return f"{self._branch_key()}_{layer_id}"

    def _anchor_step_scope_key(self) -> str:
        return f"{self._branch_key()}_{get_timestep()}"

    def _get_local_change_metrics(self, layer_id: int) -> Tuple[float, float]:
        """Return (absolute_change, relative_change) of local compressed state vs last anchor."""
        scope_key = self._anchor_scope_key(layer_id)
        curr = self.latest_local_out_comp.get(scope_key, None)
        anchor_ref = self.anchor_local_out_comp.get(scope_key, None)
        if curr is None or anchor_ref is None:
            return float("inf"), float("inf")

        abs_change = float(torch.norm(curr - anchor_ref).item())
        denom = float(torch.clamp(torch.norm(anchor_ref), min=1e-6).item())
        rel_change = abs_change / denom
        return abs_change, rel_change

    def _get_nonlocal_weight_sum(self, layer_id: int) -> float:
        layer_prefix = f"{self._branch_key()}_{layer_id}_group_"
        weight_except_local = 0.0
        for k, meta in self.group_meta.items():
            if k.startswith(layer_prefix):
                w = float(meta.get("weight", 0.0))
                weight_except_local += w

        return max(0.0, weight_except_local)

    def _build_layer_ase_threshold_candidate(self, layer_id: int) -> Optional[float]:
        """
        Candidate for one layer:
        cumulative_local_ase(abs_change) * nonlocal_weight_sum.
        """
        scope_key = self._anchor_scope_key(layer_id)
        abs_change, _ = self._get_local_change_metrics(layer_id)
        if abs_change != abs_change or abs_change == float("inf"):
            return None

        nonlocal_weight_sum = self._get_nonlocal_weight_sum(layer_id)
        candidate = max(1e-8, abs_change * nonlocal_weight_sum)
        self.layer_ase_thresholds[scope_key] = candidate
        return candidate

    def _update_global_ase_threshold_from_layers(self, step: int, trigger_layer_id: int):
        if self.total_layers is None:
            return

        branch = self._branch_key()
        candidates = []
        for layer_id in range(self.total_layers):
            scope_key = f"{branch}_{layer_id}"
            v = float(self.layer_ase_thresholds.get(scope_key, float("nan")))
            if v == v and v != float("inf") and v > 0.0:
                candidates.append(v)

        if not candidates:
            return

        q = float(max(0.0, min(1.0, self.cache_ratio)))
        candidate_tensor = torch.tensor(candidates, dtype=torch.float32)
        self.ase_threshold = max(1e-8, float(torch.quantile(candidate_tensor, q).item()))

        if self._should_log_anchor_stats(trigger_layer_id):
            joined = " ".join(f"{x:.3e}" for x in candidates)
            logger.info(
                f"[DiTangoV3][Anchor Global ASE] t{step}-{branch} "
                f"q={q:.3f} global={self.ase_threshold:.3e} candidates=[{joined}]"
            )

    def log_layer_threshold_snapshot(self, step: int):
        if not self._is_main_rank():
            return
        if self._branch_key() != "pos":
            return
        if self.total_layers is None:
            return
        if self.last_threshold_log_step == step:
            return

        branch = self._branch_key()
        items = []
        for layer_id in range(self.total_layers):
            scope_key = f"{branch}_{layer_id}"
            thr = float(self.layer_ase_thresholds.get(scope_key, self.ase_threshold))
            if thr != thr or thr == float("inf") or thr <= 0.0:
                thr = max(float(self.ase_threshold), 1e-8)
            items.append(f"L{layer_id}:{thr:.3e}")

        global_thr = float(self.ase_threshold)
        if global_thr != global_thr or global_thr == float("inf") or global_thr <= 0.0:
            global_thr = 1e-8
        logger.info(
            f"[DiTangoV3] Layer ASE thr snapshot t{step} "
            f"| global={global_thr:.3e} | {' '.join(items)}"
        )
        self.last_threshold_log_step = step

    def _is_last_layer(self, layer_id: int) -> bool:
        if self.total_layers is None:
            return False
        return layer_id == (self.total_layers - 1)

    def _compute_anchor_rel_snapshot(self) -> Tuple[bool, float, float, float, str]:
        """
        Compute step-level anchor trigger from all-layer local relative errors.
        Trigger when max rel across layers is above anchor threshold.
        """
        if self.total_layers is None:
            return False, float("nan"), float("nan"), float("nan"), "no_total_layers"

        rels = []
        items = []
        for layer_id in range(self.total_layers):
            _, rel = self._get_local_change_metrics(layer_id)
            if rel != float("inf") and rel == rel:
                rels.append(rel)
                items.append(f"L{layer_id}:{rel:.3e}")
            else:
                items.append(f"L{layer_id}:na")

        if not rels:
            return False, float("nan"), float("nan"), float("nan"), " ".join(items)

        rel_tensor = torch.tensor(rels, dtype=torch.float32)
        rel_min = float(torch.min(rel_tensor).item())
        rel_mean = float(torch.mean(rel_tensor).item())
        rel_max = float(torch.max(rel_tensor).item())
        trigger = rel_max > self.anchor_rel_err_threshold
        return trigger, rel_min, rel_mean, rel_max, " ".join(items)

    def should_anchor_this_step(self) -> bool:
        step = get_timestep()
        if step == 0:
            return True

        cache_key = self._anchor_step_scope_key()
        cached = self.anchor_step_decision_cache.get(cache_key, None)
        if cached is not None:
            return bool(cached)

        trigger, rel_min, rel_mean, rel_max, details = self._compute_anchor_rel_snapshot()
        self.anchor_step_decision_cache[cache_key] = trigger

        if self._is_main_rank() and self._branch_key() == "pos":
            logger.info(
                f"[DiTangoV3][Anchor Gate] t{step} trigger={int(trigger)} "
                f"thr={self.anchor_rel_err_threshold:.3e} "
                f"rel_min={rel_min:.3e} rel_mean={rel_mean:.3e} rel_max={rel_max:.3e} "
                f"| {details}"
            )

        return trigger

    def _is_anchor_step(self, step: int, layer_id: int) -> bool:
        return self.should_anchor_this_step()

    def _get_total_steps(self) -> Optional[int]:
        if self.task is not None and self.task.req is not None and self.task.req.params is not None:
            total = self.task.req.params.num_inference_steps
            if total is not None:
                return int(total)

        task = DiffusionBackend.generator.current_task
        if task is not None and task.req is not None and task.req.params is not None:
            total = task.req.params.num_inference_steps
            if total is not None:
                return int(total)
        return None

    def _is_warmup_or_cooldown_step(self, step: int) -> bool:
        if step < self.warmup_steps:
            return True

        total_steps = self._get_total_steps()
        if total_steps is None:
            return False
        return step >= max(0, total_steps - self.cooldown_steps)

    def _mark_anchor_step(self, step: int, layer_id: int):
        scope_key = self._anchor_scope_key(layer_id)
        self.last_anchor_step[scope_key] = step
        local_comp = self.latest_local_out_comp.get(scope_key, None)
        if local_comp is not None:
            self.anchor_local_out_comp[scope_key] = local_comp.clone()

    def _local_as_key(self, layer_id: int) -> str:
        return f"{self._branch_key()}_{layer_id}_local"


    def update_local_out_comp(self, layer_id: int, local_state: AttentionState, as_anchor: bool):
        # if local_state is None or local_state.is_empty():
        #     return
        # key = self._local_as_key(layer_id)
        comp = self._compress_out_hidden(local_state)
        self.current_local_out_comp = comp
        self.latest_local_out_comp[self._anchor_scope_key(layer_id)] = comp.clone()
        # self.current_local_as[key] = comp
        # if as_anchor:
        #     self.anchor_local_as[key] = comp

    def _update_group_ref_local_out(self, layer_id: int, group_id: int):
        group_key = self._group_key(layer_id, group_id)
        # local_key = self._local_as_key(layer_id)
        curr = self.current_local_out_comp
        if curr is None:
            return
        self.group_ref_local_out_comp[group_key] = curr.clone()

    def _get_local_drift(self, layer_id: int, group_id: int) -> float:
        # local_key = self._local_as_key(layer_id)
        group_key = self._group_key(layer_id, group_id)
        # curr = self.current_local_as.get(local_key, None)
        group_ref = self.group_ref_local_out_comp.get(group_key, None)
        if self.current_local_out_comp is None or group_ref is None:
            return float("inf")
        return float(torch.norm(self.current_local_out_comp - group_ref).item())

    def _estimate_group_ase(self, layer_id: int, group_id: int) -> float:
        key = self._group_key(layer_id, group_id)
        meta = self.group_meta.get(key, None)
        if meta is None:
            return float("inf")

        step = get_timestep()
        stale_steps = max(0, step - int(meta.get("last_compute_step", 0)))
        if stale_steps >= self.tau_max:
            return float("inf")

        w = float(meta.get("weight", 0.0))
        alpha_meta = meta.get("alpha", 1.0)
        if torch.is_tensor(alpha_meta):
            alpha = float(alpha_meta.mean().item())
        else:
            alpha = float(alpha_meta)
        drift = self._get_local_drift(layer_id, group_id)
        ase = w * alpha * drift
        self.group_meta[key]["ase"] = ase
        return ase

    @staticmethod
    def _compress_out_hidden(state: AttentionState) -> torch.Tensor:
        """Compress OUT by averaging hidden dim: [B, S, N, D] -> [B, S, N]."""
        if state is None or state.is_empty():
            return torch.empty(0, dtype=torch.float32)
        return state.out.float().mean(dim=-1)

    def plan_group_update(
        self,
        layer_id: int,
        num_groups: int,
        local_group_id: int,
    ) -> Tuple[Dict[int, bool], Dict[int, float]]:
        step = get_timestep()
        is_anchor = self._is_anchor_step(step, layer_id)
        global_ase_threshold = max(float(self.ase_threshold), 1e-8)
        plan: Dict[int, bool] = {}
        ase_map: Dict[int, float] = {}

        for group_id in range(num_groups):
            key = self._group_key(layer_id, group_id)
            # Local partition is handled separately; local group (g-1 states) is still decided by ASE.
            if is_anchor:
                plan[group_id] = True
                ase_map[group_id] = float("nan")
                continue

            if key not in DiffusionBackend.flexcache.cache:
                plan[group_id] = True
                ase_map[group_id] = float("inf")
                continue

            # Force update if this group has not been updated for num_groups+1 steps.
            meta = self.group_meta.get(key, None)
            if meta is None:
                plan[group_id] = True
                ase_map[group_id] = float("inf")
                continue
            last_compute_step = int(meta.get("last_compute_step", -1))
            if (step - last_compute_step) >= (num_groups + 1):
                plan[group_id] = True
                ase_map[group_id] = float("inf")
                continue

            ase = self._estimate_group_ase(layer_id, group_id)
            ase_map[group_id] = ase
            # Non-anchor step selection compares against global threshold.
            plan[group_id] = ase >= global_ase_threshold

        return plan, ase_map

    def update_anchor_stats(
        self,
        layer_id: int,
        group_states: Dict[int, AttentionState],
        local_state: Optional[AttentionState],
    ):
        """
        Anchor step: compute weights and alpha for each group based on AS norms, and update local AS reference for drift estimation.
        """
        if local_state is None or local_state.is_empty():
            return

        step = get_timestep()

        local_lse_score = float(torch.norm(local_state.lse.float()).item())
        ordered_group_ids = sorted(group_states.keys())
        lse_scores = []
        for group_id in ordered_group_ids:
            state = group_states[group_id]
            lse_scores.append(float(torch.norm(state.lse.float()).item()))

        # logger.info(
        #     f"[DiTangoV3 Anchor Stats] Layer {layer_id} | Local LSE norm: {local_lse_score:.6f} | "
        #     + " | ".join(
        #         f"G{group_id}: {score:.6f}" for group_id, score in zip(ordered_group_ids, lse_scores)
        #     )
        # )

        # Local partition also participates in weight normalization.
        score_tensor = torch.tensor(lse_scores + [local_lse_score], dtype=torch.float32)
        # Use std as temperature to avoid near one-hot weights when raw norms are large.
        temperature = torch.clamp(torch.std(score_tensor, unbiased=False), min=1e-6)
        normalized_scores = (score_tensor - score_tensor.max()) / temperature
        weight_tensor = torch.softmax(normalized_scores, dim=0)

        local_comp = self._compress_out_hidden(local_state)
        local_comp_denom = torch.clamp(local_comp.abs(), min=1e-6)
        # Anchor step stores compressed local AS reference for later drift estimation.
        self.update_local_out_comp(layer_id, local_state, as_anchor=True)
        local_abs_change, local_rel_change = self._get_local_change_metrics(layer_id)
        layer_candidate = self._build_layer_ase_threshold_candidate(layer_id)

        log_items = []

        for idx, group_id in enumerate(ordered_group_ids):
            state = group_states[group_id]
            key = self._group_key(layer_id, group_id)
            weight = float(weight_tensor[idx].item())
            group_comp = self._compress_out_hidden(state)
            alpha = group_comp.abs() / local_comp_denom
            alpha_mean = float(alpha.mean().item()) if alpha.numel() > 0 else float("nan")

            # Anchor ASE uses unified local absolute-change scale.
            if local_abs_change == float("inf"):
                anchor_ase = float("nan")
            else:
                anchor_ase = weight * alpha_mean * local_abs_change

            self.group_meta[key] = {
                "weight": weight,
                "alpha": alpha.detach().clone(),
                "last_compute_step": step,
                "ase": float(anchor_ase) if anchor_ase == anchor_ase else 0.0,
            }

            self._update_group_ref_local_out(layer_id, group_id)
            if self._should_log_anchor_stats(layer_id):
                if anchor_ase == anchor_ase:
                    log_items.append(
                        f"G{group_id}:w={weight:.6f},a_mean={alpha_mean:.6f},ase={anchor_ase:.6f}"
                    )
                else:
                    log_items.append(
                        f"G{group_id}:w={weight:.6f},a_mean={alpha_mean:.6f},ase=na"
                    )

        if self._should_log_anchor_stats(layer_id):
            ref_norm = self._get_group_local_ref_norm(layer_id, group_id)
            if ref_norm == ref_norm:
                ref_norm_str = f"{ref_norm:.6f}"
            else:
                ref_norm_str = "na"
            log_items.append(f"local_out_norm={ref_norm_str}")

        if log_items and self._should_log_anchor_stats(layer_id):
            local_weight = float(weight_tensor[-1].item())
            if layer_candidate is None:
                candidate_str = "na"
            else:
                candidate_str = f"{layer_candidate:.3e}"
            logger.info(
                f"[DiTangoV3 Anchor t{step}l{layer_id}-{self._branch_key()}] "
                f"local:w={local_weight:.6f},abs={local_abs_change:.3e},rel={local_rel_change:.3e},"
                f"candidate={candidate_str} | {' | '.join(log_items)}"
            )

        # Requirement: update global ASE threshold only at anchor step last layer.
        if self._is_last_layer(layer_id):
            self._update_global_ase_threshold_from_layers(step, layer_id)

        # Anchor step refreshes per-group meta/local refs per layer, and final layer
        # performs one-shot global ASE threshold aggregation for this step.

    def get_reuse_key(self, layer_id: int, group_id: int = 0, **kwargs):
        return self._group_key(layer_id, group_id)

    def reuse(self, layer_id: int, group_id: int = 0, **kwargs) -> AttentionState:
        key = self.get_reuse_key(layer_id, group_id=group_id)
        step = get_timestep()
        if key in DiffusionBackend.flexcache.cache:
            state = DiffusionBackend.flexcache.cache[key]
            return state

        return AttentionState()

    def get_store_key(self, layer_id: int, group_id: int = 0, **kwargs):
        return self._group_key(layer_id, group_id)

    def store(self, layer_id: int, group_id: int, group_state: Optional[AttentionState], **kwargs):
        step = get_timestep()
        if self._is_warmup_or_cooldown_step(step):
            return

        if group_state is None or group_state.is_empty():
            return

        key = self.get_store_key(layer_id, group_id=group_id)
        DiffusionBackend.flexcache.cache[key] = group_state
        self.group_meta.setdefault(key, {})["last_compute_step"] = step
        # Drift reference for this group should bind to local AS at this group's last compute step.
        self._update_group_ref_local_out(layer_id, group_id)

    def reset_state(self):
        self.anchor_local_as: Dict[str, torch.Tensor] = {}
        # self.current_local_as: Dict[str, torch.Tensor] = {}
        self.current_local_out_comp: Optional[torch.Tensor] = None
        self.group_ref_local_out_comp: Dict[str, torch.Tensor] = {}
        self.group_meta: Dict[str, Dict[str, Any]] = {}
        self.last_anchor_step: Dict[str, int] = {}
        self.anchor_local_out_comp: Dict[str, torch.Tensor] = {}
        self.latest_local_out_comp: Dict[str, torch.Tensor] = {}
        self.layer_ase_thresholds: Dict[str, float] = {}
        self.last_threshold_log_step: int = -1
        self.anchor_step_decision_cache: Dict[str, bool] = {}
        self.compute_ratio_records: Dict[str, Dict[Tuple[int, int], float]] = {"pos": {}, "neg": {}}
        self.group_decision_records: Dict[str, Dict[Tuple[int, int, int], bool]] = {"pos": {}, "neg": {}}
        self.phase_records: Dict[str, Dict[Tuple[int, int], str]] = {"pos": {}, "neg": {}}
        self.max_step_seen: Dict[str, int] = {"pos": 0, "neg": 0}
        if DiffusionBackend.flexcache is not None:
            DiffusionBackend.flexcache.cache.clear()
        logger.debug("DiTango v3 state reset")

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        if not hasattr(module, "_original_attn"):
            module._original_attn = module.blocks[0].self_attn.attn_func

        self.total_layers = len(module.blocks)

        for layer_id, block in enumerate(module.blocks):
            block.self_attn.attn_func = DitangoV3Attention(layer_id=layer_id)

        logger.info(f"Module {module.__class__.__name__} wrapped with ditango strategy")

    def unwrap_module(self, module: torch.nn.Module) -> None:
        if hasattr(module, "_original_attn"):
            for _, block in enumerate(module.blocks):
                block.self_attn.attn_func = module._original_attn
            delattr(module, "_original_attn")
        self.dump_compute_reuse_visualization()
        logger.info(f"Module {module.__class__.__name__} unwrapped")


class DitangoV3Attention:
    def __init__(self, layer_id: int):
        self.attn_backend = DiffusionBackend.attn
        self.group = get_cp_group()
        self.cp_size = self.group.group_size
        self.global_rank = torch.distributed.get_rank()
        self.rank_in_cp = self.group.rank_in_group
        self.layer_id = layer_id

        strategy: DiTangoV3Strategy = DiffusionBackend.flexcache.strategy
        self.intra_group_size = min(self.cp_size, strategy.intra_group_size_limit)
        self.ulysses_size = min(DiffusionBackend.args.infer.diffusion.up_limit, self.cp_size, self.intra_group_size)
        self.enable_cfg_parallel = DiffusionBackend.args.infer.diffusion.cfg_size > 1

        assert self.cp_size <= self.intra_group_size or self.cp_size % self.intra_group_size == 0

        if self.global_rank == 0:
            logger.info(f"L{layer_id} | Using Ditango Attn v3 (group selective).")

    def _print_layer0(self, message: str):
        if self.global_rank == 0 and self.layer_id == 0:
            logger.info(message)

    def _sync_plan_from_pos(
        self,
        local_plan: Dict[int, bool],
        local_ase_map: Dict[int, float],
        num_groups: int,
    ) -> Tuple[Dict[int, bool], Dict[int, float]]:
        """Broadcast POS branch plan/ASE so NEG reuses exactly the same decision."""
        if not self.enable_cfg_parallel:
            return local_plan, local_ase_map

        cfg_group = get_cfg_group()
        if cfg_group.group_size != 2:
            return local_plan, local_ase_map

        src_rank = cfg_group.rank_list[0]

        if cfg_group.rank_in_group == 0:
            plan_tensor = torch.tensor(
                [1 if local_plan.get(group_id, False) else 0 for group_id in range(num_groups)],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            ase_tensor = torch.tensor(
                [float(local_ase_map.get(group_id, float("nan"))) for group_id in range(num_groups)],
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        else:
            plan_tensor = torch.zeros(num_groups, dtype=torch.int32, device=torch.cuda.current_device())
            ase_tensor = torch.zeros(num_groups, dtype=torch.float32, device=torch.cuda.current_device())

        cfg_group.broadcast(plan_tensor, src=src_rank)
        cfg_group.broadcast(ase_tensor, src=src_rank)

        synced_plan = {group_id: bool(int(plan_tensor[group_id].item())) for group_id in range(num_groups)}
        synced_ase_map = {group_id: float(ase_tensor[group_id].item()) for group_id in range(num_groups)}
        return synced_plan, synced_ase_map

    def _sync_anchor_flag_from_pos(self, local_flag: bool) -> bool:
        """Broadcast POS anchor gate result so POS/NEG always follow the same path."""
        if not self.enable_cfg_parallel:
            return local_flag

        cfg_group = get_cfg_group()
        if cfg_group.group_size != 2:
            return local_flag

        src_rank = cfg_group.rank_list[0]
        if cfg_group.rank_in_group == 0:
            flag_tensor = torch.tensor(
                [1 if local_flag else 0],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
        else:
            flag_tensor = torch.zeros(1, dtype=torch.int32, device=torch.cuda.current_device())

        cfg_group.broadcast(flag_tensor, src=src_rank)
        return bool(int(flag_tensor.item()))

    def _format_decision_log(self, plan: Dict[int, bool], ase_map: Dict[int, float], 
                               strategy_ase_threshold: float, strategy: DiTangoV3Strategy) -> str:
        """Format per-group decision log with ASE values, thresholds, and choices."""
        items = []
        for group_id in sorted(plan.keys()):
            ase = ase_map.get(group_id, float("nan"))
            should_compute = plan[group_id]
            
            # Determine status string
            if ase != ase:  # NaN usually means anchor or local group forced compute
                status = "LOCAL"
                score_str = "  --  "
            elif ase == float("inf"):
                status = "MISS "
                score_str = " INF  "
            else:
                status = "COMP " if should_compute else "REUSE"
                score_str = f"{ase:.4f}"

            items.append(f"G{group_id}:{score_str}({status})")
        
        body = " ".join(items)
        return f"Thr={strategy_ase_threshold:.3e} | {body}"

    def _is_varlen_mode(
        self,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
    ) -> bool:
        return (
            cu_seqlens_q is not None
            and cu_seqlens_k is not None
            and max_seqlen_q is not None
            and max_seqlen_k is not None
        )

    @staticmethod
    def _timer_sync_start() -> Optional[float]:
        if not Timer.is_enabled():
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _timer_elapsed_ms(start_ts: Optional[float]) -> Optional[float]:
        if start_ts is None:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - start_ts) * 1000.0

    @staticmethod
    def _phase_tag(force_full_compute: bool, is_anchor: bool) -> str:
        if force_full_compute:
            return "warmup_cooldown"
        if is_anchor:
            return "anchor"
        return "normal"

    def _record_selective_timer_metrics(
        self,
        strategy: DiTangoV3Strategy,
        elapsed_ms: Optional[float],
        force_full_compute: bool,
        is_anchor: bool,
        plan: Dict[int, bool],
    ) -> None:
        if elapsed_ms is None:
            return

        strategy_name = str(getattr(strategy, "type", "unknown"))
        phase = self._phase_tag(force_full_compute, is_anchor)
        total_groups = max(1, len(plan))
        compute_groups = int(sum(1 for should_compute in plan.values() if should_compute))

        Timer.record(f"xattn.{phase}", elapsed_ms)
        # Warmup/cooldown and anchor always compute all groups by design,
        # so compute_X_of_Y buckets are only meaningful in normal phase.
        if phase == "normal":
            Timer.record(
                f"xattn.{phase}.{compute_groups}/{total_groups}",
                elapsed_ms,
            )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        dropout_p: float = 0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        return_attn_probs: bool = False,
    ) -> Tuple[torch.Tensor, None, None]:
        is_varlen = self._is_varlen_mode(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
        seq_dim, head_dim = (0, 1) if is_varlen else (1, 2)
        strategy: DiTangoV3Strategy = DiffusionBackend.flexcache.strategy

        attn_kwargs = {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": window_size,
            "deterministic": deterministic,
            "return_attn_probs": True,
        }

        if self.ulysses_size > 1:
            ulysses_group = get_up_group(self.ulysses_size)
            q = ulysses_group.all_to_all(q, head_dim, seq_dim)
            k = ulysses_group.all_to_all(k, head_dim, seq_dim)
            v = ulysses_group.all_to_all(v, head_dim, seq_dim)

        curr_step = get_timestep()
        force_full_compute = strategy._is_warmup_or_cooldown_step(curr_step)
        pre_anchor_local = strategy.should_anchor_this_step()
        pre_anchor = self._sync_anchor_flag_from_pos(pre_anchor_local)

        # Keep per-branch cache consistent with synced CFG decision for this step.
        cache_key = strategy._anchor_step_scope_key()
        strategy.anchor_step_decision_cache[cache_key] = pre_anchor

        timer_start = self._timer_sync_start()

        if force_full_compute or pre_anchor:
            final_state, group_states, local_state, plan, ase_map = self._anchor_step_attn(q, k, v, is_varlen, **attn_kwargs)
            fallback_count = 0
            is_anchor = True
        else:
            final_state, group_states, local_state, plan, ase_map, fallback_count = self._selective_group_attn(
                q, k, v, is_varlen, **attn_kwargs
            )
            # Non-anchor path must stay non-anchor; anchor is only decided by gate.
            is_anchor = False

        elapsed_ms = self._timer_elapsed_ms(timer_start)
        if elapsed_ms is not None:
            Timer.record("ditango_attn_v3", elapsed_ms)
        self._record_selective_timer_metrics(strategy, elapsed_ms, force_full_compute, is_anchor, plan)

        if self.ulysses_size > 1:
            out = ulysses_group.all_to_all(final_state.out, seq_dim, head_dim)
        else:
            out = final_state.out

        local_group_id = self.rank_in_cp // self.intra_group_size
        strategy.record_step_layer_stats(
            layer_id=self.layer_id,
            step=curr_step,
            plan=plan,
            is_warmup_or_cooldown=force_full_compute,
            is_anchor=(not force_full_compute) and is_anchor,
        )
 
        if is_anchor:
            if not force_full_compute:
                # Compute anchor stats against previous anchor ref first.
                strategy.update_anchor_stats(self.layer_id, group_states, local_state)
            # Refresh anchor reference after stats so rel/abs are not forced to zero.
            strategy._mark_anchor_step(curr_step, self.layer_id)

        if self.layer_id == 0:
            strategy.log_layer_threshold_snapshot(curr_step)

        # Log per-group decision details only for selected debug layers.
        if strategy._should_log_anchor_stats(self.layer_id):
            decision_text = self._format_decision_log(plan, ase_map, strategy.ase_threshold, strategy)
            phase = "anchor" if (is_anchor and not force_full_compute) else ("warmup/cooldown" if force_full_compute else "normal")
            logger.info(
                f"[DiTango {phase} Decision t{curr_step}l{self.layer_id}-{strategy._branch_key()}] "
                f"| {decision_text} |"
            )

        # Clean one-step anchor decision cache on the last layer to avoid unbounded growth.
        if strategy._is_last_layer(self.layer_id):
            cache_key = strategy._anchor_step_scope_key()
            if cache_key in strategy.anchor_step_decision_cache:
                del strategy.anchor_step_decision_cache[cache_key]
        return out, None, None

    def _anchor_step_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_varlen: bool,
        **kwargs,
    ) -> Tuple[AttentionState, Dict[int, AttentionState], AttentionState, Dict[int, bool], Dict[int, float]]:
        """Anchor step: compute all group attention states and cache them."""
        strategy: DiTangoV3Strategy = DiffusionBackend.flexcache.strategy

        num_groups = self.cp_size // self.intra_group_size
        local_group_id = self.rank_in_cp // self.intra_group_size
        outer_loop_size = num_groups
        outer_loop_next_rank = (self.rank_in_cp - self.intra_group_size) % self.cp_size
        outer_loop_prev_rank = (self.rank_in_cp + self.intra_group_size) % self.cp_size

        inner_offset = self.rank_in_cp // self.intra_group_size * self.intra_group_size
        inner_loop_size = self.intra_group_size // self.ulysses_size
        inner_loop_next_rank = (self.rank_in_cp - self.ulysses_size) % self.intra_group_size + inner_offset
        inner_loop_prev_rank = (self.rank_in_cp + self.ulysses_size) % self.intra_group_size + inner_offset

        final_state = AttentionState()
        computed_group_states: Dict[int, AttentionState] = {}
        local_partition_state = AttentionState()
        plan: Dict[int, bool] = {}
        ase_map: Dict[int, float] = {}

        current_group_id = local_group_id
        for outer_step in range(outer_loop_size):
            plan[current_group_id] = True
            ase_map[current_group_id] = float("nan")
            group_state = AttentionState()

            for inner_step in range(inner_loop_size):
                if is_varlen:
                    cu_seqlens_k = kwargs.get("cu_seqlens_k", None)
                    data_pack = (k, v, cu_seqlens_k)
                else:
                    data_pack = (k, v)

                if inner_step + 1 != inner_loop_size:
                    nxt_data_pack = async_ring_p2p_commit(
                        self.group,
                        data_pack,
                        src_rank=inner_loop_prev_rank,
                        dst_rank=inner_loop_next_rank,
                    )
                elif outer_step + 1 != outer_loop_size:
                    nxt_data_pack = async_ring_p2p_commit(
                        self.group,
                        data_pack,
                        src_rank=outer_loop_prev_rank,
                        dst_rank=outer_loop_next_rank,
                    )

                block_out, block_lse, _ = self.attn_backend(q, k, v, **kwargs)

                is_local_partition = (outer_step == 0) and (inner_step == 0)
                if is_local_partition:
                    # Local partition is always computed and contributes only to final merge.
                    local_partition_state.update(block_out, block_lse)
                    strategy.update_local_out_comp(self.layer_id, local_partition_state, as_anchor=False)
                else:
                    group_state.update(block_out, block_lse)

                if (inner_step + 1 != inner_loop_size) or (outer_step + 1 != outer_loop_size):
                    if is_varlen:
                        k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                    else:
                        k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)

            if not group_state.is_empty():
                computed_group_states[current_group_id] = group_state
                strategy.store(self.layer_id, group_id=current_group_id, group_state=group_state)
            current_group_id = (current_group_id + 1) % num_groups

        if not local_partition_state.is_empty():
            final_state = AttentionState.merge(final_state, local_partition_state)

        # Merge after all group states are ready so anchor stats use complete per-group AS.
        for group_id in sorted(computed_group_states.keys()):
            final_state = AttentionState.merge(final_state, computed_group_states[group_id])

        return final_state, computed_group_states, local_partition_state, plan, ase_map

    def _selective_group_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_varlen: bool,
        **kwargs,
    ) -> Tuple[AttentionState, Dict[int, AttentionState], AttentionState, Dict[int, bool], Dict[int, float], int]:
        strategy: DiTangoV3Strategy = DiffusionBackend.flexcache.strategy

        num_groups = self.cp_size // self.intra_group_size
        local_group_id = self.rank_in_cp // self.intra_group_size
        
        # NOTE: Plan is calculated AFTER the first outer step (local group computation)
        # to ensure we have the up-to-date local Attention State for drift estimation.
        group_plan: Dict[int, bool] = {}
        ase_map: Dict[int, float] = {}

        outer_loop_size = num_groups
        outer_loop_next_rank = (self.rank_in_cp - self.intra_group_size) % self.cp_size
        outer_loop_prev_rank = (self.rank_in_cp + self.intra_group_size) % self.cp_size

        inner_offset = self.rank_in_cp // self.intra_group_size * self.intra_group_size
        inner_loop_size = self.intra_group_size // self.ulysses_size
        inner_loop_next_rank = (self.rank_in_cp - self.ulysses_size) % self.intra_group_size + inner_offset
        inner_loop_prev_rank = (self.rank_in_cp + self.ulysses_size) % self.intra_group_size + inner_offset

        final_state = AttentionState()
        computed_group_states: Dict[int, AttentionState] = {}
        local_partition_state = AttentionState()
        fallback_count = 0

        current_group_id = local_group_id
        for outer_step in range(outer_loop_size):
            # Calculate plan after the local group (outer_step=0) is computed
            if outer_step == 1:
                group_plan, ase_map = strategy.plan_group_update(
                    self.layer_id,
                    num_groups,
                    local_group_id,
                )
                group_plan, ase_map = self._sync_plan_from_pos(group_plan, ase_map, num_groups)

            # Determine whether to compute for current group
            should_compute = True
            if outer_step == 0:
                should_compute = True # Local group always computes
            elif current_group_id in group_plan:
                should_compute = group_plan[current_group_id]
            else:
                # Should not happen if plan logic is correct for outer_step >= 1
                should_compute = True

            reused_state = AttentionState()
            if not should_compute:
                reused_state = strategy.reuse(self.layer_id, group_id=current_group_id)
                if reused_state.is_empty():
                    # Missing cache must fallback to compute to preserve correctness.
                    should_compute = True
                    # Update plan/ase to reflect fallback
                    group_plan[current_group_id] = True
                    ase_map[current_group_id] = float("inf")
                    fallback_count += 1
                    if strategy._should_log_cache_io(self.layer_id):
                        key = strategy.get_reuse_key(self.layer_id, group_id=current_group_id)
                        logger.info(
                            f"[DiTangoV3][Fallback->Compute] step={get_timestep()} "
                            f"group={current_group_id} key={key} reason=cache_miss"
                        )

            group_state = AttentionState()
            for inner_step in range(inner_loop_size):
                if is_varlen:
                    cu_seqlens_k = kwargs.get("cu_seqlens_k", None)
                    data_pack = (k, v, cu_seqlens_k)
                else:
                    data_pack = (k, v)

                if inner_step + 1 != inner_loop_size:
                    nxt_data_pack = async_ring_p2p_commit(
                        self.group,
                        data_pack,
                        src_rank=inner_loop_prev_rank,
                        dst_rank=inner_loop_next_rank,
                    )
                elif outer_step + 1 != outer_loop_size:
                    nxt_data_pack = async_ring_p2p_commit(
                        self.group,
                        data_pack,
                        src_rank=outer_loop_prev_rank,
                        dst_rank=outer_loop_next_rank,
                    )

                is_local_partition = (outer_step == 0) and (inner_step == 0)
                # Local partition must always be computed and merged into final_state only.
                if is_local_partition:
                    block_out, block_lse, _ = self.attn_backend(q, k, v, **kwargs)
                    local_partition_state.update(block_out, block_lse)
                    strategy.update_local_out_comp(self.layer_id, local_partition_state, as_anchor=False)
                elif should_compute:
                    block_out, block_lse, _ = self.attn_backend(q, k, v, **kwargs)
                    group_state.update(block_out, block_lse)

                if (inner_step + 1 != inner_loop_size) or (outer_step + 1 != outer_loop_size):
                    if is_varlen:
                        k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                    else:
                        k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)

            if should_compute:
                if not group_state.is_empty():
                    strategy.store(self.layer_id, group_id=current_group_id, group_state=group_state)
                    computed_group_states[current_group_id] = group_state
                    final_state = AttentionState.merge(final_state, group_state)
            else:
                final_state = AttentionState.merge(final_state, reused_state)

            current_group_id = (current_group_id + 1) % num_groups

        # Ensure plan exists if num_groups == 1 (loop didn't hit outer_step=1)
        if not group_plan:
             group_plan, ase_map = strategy.plan_group_update(
                self.layer_id,
                num_groups,
                local_group_id,
                )
             group_plan, ase_map = self._sync_plan_from_pos(group_plan, ase_map, num_groups)

        if not local_partition_state.is_empty():
            final_state = AttentionState.merge(final_state, local_partition_state)

        return final_state, computed_group_states, local_partition_state, group_plan, ase_map, fallback_count