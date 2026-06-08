import json
import os
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

import torch

from chitu_diffusion.core.distributed.parallel_state import get_cfg_group, get_cp_group, get_world_group
from chitu_diffusion.core.logging_utils import should_log_info_on_rank
from chitu_diffusion.ditango.state import AttentionState
from chitu_diffusion.observability.timer import Timer
from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend
from chitu_diffusion.runtime.output_layout import debug_output_dir, task_logs_dir
from chitu_diffusion.runtime.task import DiffusionTask

logger = getLogger(__name__)


def get_timestep() -> int:
    return DiffusionBackend.generator.current_task.buffer.current_step


def _decision_color(code: int) -> Tuple[int, int, int]:
    if code == DiTangoPlanner.DECISION_CODE_WARMUP_COOLDOWN:
        return (160, 160, 160)
    if code == DiTangoPlanner.DECISION_CODE_ANCHOR:
        return (30, 180, 80)
    if code == DiTangoPlanner.DECISION_CODE_COMPUTE:
        return (40, 140, 255)
    if code == DiTangoPlanner.DECISION_CODE_REUSE:
        return (255, 180, 40)
    return (0, 0, 0)


def save_ditango_decision_ppm(
    records: Dict[Tuple[int, int, int], int],
    max_step: int,
    total_layers: int,
    group_num: int,
    ppm_path: str,
) -> None:
    width = max_step + 1
    height = total_layers * group_num
    pixels = [(0, 0, 0)] * (width * height)
    for (step, layer_id, group_id), code in records.items():
        if step < 0 or step >= width or layer_id < 0 or layer_id >= total_layers:
            continue
        if group_id < 0 or group_id >= group_num:
            continue
        row = layer_id * group_num + group_id
        pixels[row * width + step] = _decision_color(code)

    with open(ppm_path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(bytes(channel for pixel in pixels for channel in pixel))


class DiTangoPlanner:
    """CuBic-style anchor-only planner for CP-group DiTango reuse."""

    DEFAULT_ANCHOR_INTERVAL = 5
    DEFAULT_TAU_MAX = 8
    DEFAULT_INTRA_GROUP_SIZE_LIMIT = 1
    DEFAULT_WARMUP_STEPS = 5
    DEFAULT_COOLDOWN_STEPS = 5
    DEFAULT_CURVATURE_INTERVAL_POWER = 1.0 / 3.0
    DEFAULT_LOCALITY_GROUP_COMPUTE_BOOST = 0.0
    DEFAULT_ANCHOR_LOG_LAYERS = ()
    DECISION_CODE_WARMUP_COOLDOWN = 0
    DECISION_CODE_ANCHOR = 1
    DECISION_CODE_COMPUTE = 2
    DECISION_CODE_REUSE = 3
    SOFTMAX_TEMPERATURE_MIN = 1e-6
    CURVATURE_EPSILON = 1e-8

    def __init__(
        self,
        task: Optional[DiffusionTask] = None,
        cache_ratio: Optional[float] = None,
        anchor_interval: Optional[int] = None,
        tau_max: Optional[int] = None,
        intra_group_size_limit: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        cooldown_steps: Optional[int] = None,
        curvature_interval_power: Optional[float] = None,
        locality_group_compute_boost: Optional[float] = None,
        groupwise_stagger_period: Optional[int] = None,
        groupwise_stagger_fresh_count: Optional[int] = None,
        groupwise_stagger_layer_start: Optional[int] = None,
        groupwise_stagger_layer_end: Optional[int] = None,
        groupwise_keep_local: Optional[bool] = None,
        groupwise_force_tail_full_layers: Optional[int] = None,
        groupwise_reuse_stale_kv: Optional[bool] = None,
        groupwise_local_expand: Optional[int] = None,
        groupwise_fixed_anchor_steps: Optional[str] = None,
        groupwise_topk_mode: Optional[str] = None,
        groupwise_extra_topk: Optional[int] = None,
        groupwise_state_align: Optional[bool] = None,
        groupwise_state_align_mode: Optional[str] = None,
        groupwise_state_align_out_scale: Optional[float] = None,
        groupwise_state_align_lse_scale: Optional[float] = None,
        groupwise_state_align_distance_tau: Optional[float] = None,
    ):
        if cache_ratio is None:
            cache_ratio = 0.5
        if anchor_interval is None:
            anchor_interval = self.DEFAULT_ANCHOR_INTERVAL
        if tau_max is None:
            tau_max = self.DEFAULT_TAU_MAX
        if intra_group_size_limit is None:
            intra_group_size_limit = self.DEFAULT_INTRA_GROUP_SIZE_LIMIT
        if warmup_steps is None:
            warmup_steps = self.DEFAULT_WARMUP_STEPS
        if cooldown_steps is None:
            cooldown_steps = self.DEFAULT_COOLDOWN_STEPS
        if curvature_interval_power is None:
            curvature_interval_power = self.DEFAULT_CURVATURE_INTERVAL_POWER
        if locality_group_compute_boost is None:
            locality_group_compute_boost = self.DEFAULT_LOCALITY_GROUP_COMPUTE_BOOST
        if groupwise_stagger_period is None:
            groupwise_stagger_period = 0
        if groupwise_stagger_fresh_count is None:
            groupwise_stagger_fresh_count = 0
        if groupwise_stagger_layer_start is None:
            groupwise_stagger_layer_start = 0
        if groupwise_stagger_layer_end is None:
            groupwise_stagger_layer_end = -1
        if groupwise_keep_local is None:
            groupwise_keep_local = True
        if groupwise_force_tail_full_layers is None:
            groupwise_force_tail_full_layers = 1
        if groupwise_reuse_stale_kv is None:
            groupwise_reuse_stale_kv = False
        if groupwise_local_expand is None:
            groupwise_local_expand = -1
        if groupwise_fixed_anchor_steps is None:
            groupwise_fixed_anchor_steps = ""
        if groupwise_topk_mode is None:
            groupwise_topk_mode = "none"
        if groupwise_extra_topk is None:
            groupwise_extra_topk = 0
        if groupwise_state_align is None:
            groupwise_state_align = False
        if groupwise_state_align_mode is None:
            groupwise_state_align_mode = "delta"
        if groupwise_state_align_out_scale is None:
            groupwise_state_align_out_scale = 1.0
        if groupwise_state_align_lse_scale is None:
            groupwise_state_align_lse_scale = 1.0
        if groupwise_state_align_distance_tau is None:
            groupwise_state_align_distance_tau = 2.0

        self.type = "ditango"
        self.task = task
        self.task_id = getattr(task, "task_id", None)
        self.cache_ratio = float(max(0.0, min(1.0, cache_ratio)))
        self.anchor_interval = max(1, int(anchor_interval))
        self.tau_max = max(1, int(tau_max))
        self.intra_group_size_limit = max(1, int(intra_group_size_limit))
        self.warmup_steps = max(0, int(warmup_steps))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.curvature_interval_power = max(0.0, float(curvature_interval_power))
        self.locality_group_compute_boost = max(0.0, float(locality_group_compute_boost))
        self.groupwise_stagger_period = max(0, int(groupwise_stagger_period))
        self.groupwise_stagger_fresh_count = max(0, int(groupwise_stagger_fresh_count))
        self.groupwise_stagger_layer_start = max(0, int(groupwise_stagger_layer_start))
        self.groupwise_stagger_layer_end = int(groupwise_stagger_layer_end)
        self.groupwise_keep_local = bool(groupwise_keep_local)
        self.groupwise_force_tail_full_layers = max(0, int(groupwise_force_tail_full_layers))
        self.groupwise_reuse_stale_kv = bool(groupwise_reuse_stale_kv)
        self.groupwise_local_expand = int(groupwise_local_expand)
        self.groupwise_fixed_anchor_steps = str(groupwise_fixed_anchor_steps)
        self.fixed_anchor_steps = self._parse_fixed_anchor_steps(self.groupwise_fixed_anchor_steps)
        self.groupwise_topk_mode = str(groupwise_topk_mode).strip().lower()
        self.groupwise_extra_topk = max(0, int(groupwise_extra_topk))
        self.groupwise_state_align = bool(groupwise_state_align)
        self.groupwise_state_align_mode = str(groupwise_state_align_mode)
        self.groupwise_state_align_out_scale = float(groupwise_state_align_out_scale)
        self.groupwise_state_align_lse_scale = float(groupwise_state_align_lse_scale)
        self.groupwise_state_align_distance_tau = max(1e-6, float(groupwise_state_align_distance_tau))
        self.anchor_log_layers = set(self.DEFAULT_ANCHOR_LOG_LAYERS)
        self.total_layers = None
        cp_size = get_cp_group().group_size
        self.intra_group_size_limit = min(cp_size, self.intra_group_size_limit)
        while cp_size % self.intra_group_size_limit != 0 and self.intra_group_size_limit > 1:
            self.intra_group_size_limit -= 1
        self.effective_group_size = min(cp_size, self.intra_group_size_limit)
        self.group_num = max(1, cp_size // self.intra_group_size_limit)

        self.state_cache: Dict[str, AttentionState] = {}
        self.compressed_state_cache: Dict[str, AttentionState] = {}
        self.kv_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = {}
        self.full_output_compress: Dict[str, torch.Tensor] = {}
        self.prev_anchor_full_output_compress: Dict[str, torch.Tensor] = {}
        self.group_meta: Dict[str, Dict[str, Any]] = {}
        self.group_intervals: Dict[str, torch.Tensor] = {}
        self.group_curvature: Dict[str, torch.Tensor] = {}
        self.layer_curvature: Dict[str, torch.Tensor] = {}
        self.ditango_plan: Dict[str, torch.Tensor] = {}
        self.anchor_step_list: Dict[str, list[int]] = {}
        self.last_anchor_step: Dict[str, int] = {}
        self._anchor_has_plan: Dict[str, bool] = {}
        self.cfg1_pos_anchor_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._policy_trace_records: Dict[str, Dict[Tuple[int, int, int], Dict[str, Any]]] = {}

        self.reset_state()

    def reset_state(self) -> None:
        self.state_cache.clear()
        self.compressed_state_cache.clear()
        self.kv_cache.clear()
        self.full_output_compress.clear()
        self.prev_anchor_full_output_compress.clear()
        self.group_meta = {}
        self.group_intervals = {}
        self.group_curvature = {}
        self.layer_curvature = {}
        self.ditango_plan.clear()
        initial_anchor_steps = self._initial_anchor_steps()
        self.anchor_step_list = {"pos": list(initial_anchor_steps), "neg": list(initial_anchor_steps)}
        self.last_anchor_step = {"pos": -1, "neg": -1}
        self._anchor_has_plan = {"pos": False, "neg": False}
        self.cfg1_pos_anchor_cache.clear()
        self._decision_vis_records: Dict[str, Dict[Tuple[int, int, int], int]] = {"pos": {}, "neg": {}}
        self._decision_vis_max_step = -1
        self._step_fallback_counts: Dict[str, Dict[Tuple[int, int], int]] = {"pos": {}, "neg": {}}
        self._policy_trace_records = {"pos": {}, "neg": {}}
        logger.debug("DiTango state reset")

    def _initial_anchor_steps(self) -> list[int]:
        if self.warmup_steps >= 2:
            steps = [self.warmup_steps - 2, self.warmup_steps - 1]
        else:
            steps = [0, 1]
        steps.extend(self.fixed_anchor_steps)
        return sorted({step for step in steps if step >= 0})

    @staticmethod
    def _parse_fixed_anchor_steps(raw_steps: str) -> list[int]:
        if not raw_steps:
            return []
        steps = []
        for item in str(raw_steps).replace(",", "|").split("|"):
            item = item.strip()
            if not item:
                continue
            steps.append(int(item))
        return sorted(set(steps))

    def is_full_compute_mode(self) -> bool:
        return self.cache_ratio <= 0.0

    def is_max_cache_mode(self) -> bool:
        return self.cache_ratio >= 1.0

    def _branch_key(self) -> str:
        return "pos" if DiffusionBackend.cfg_type == CFGType.POS else "neg"

    def _branch_layer_key(self, layer_id: int) -> str:
        return f"{self._branch_key()}_{layer_id}"

    def _branch_layer_group_key(self, layer_id: int, group_id: int) -> str:
        return f"{self._branch_layer_key(layer_id)}_group_{group_id}"

    def _explicit_branch_layer_key(self, branch_key: str, layer_id: int) -> str:
        return f"{branch_key}_{layer_id}"

    def _explicit_branch_layer_group_key(self, branch_key: str, layer_id: int, group_id: int) -> str:
        return f"{branch_key}_{layer_id}_group_{group_id}"

    def _should_log_summary(self) -> bool:
        return should_log_info_on_rank() and (get_cp_group().rank_in_group == 0)

    def _is_last_layer(self, layer_id: int) -> bool:
        return self.total_layers is not None and layer_id == (self.total_layers - 1)

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

    def is_warmup_or_cooldown_step(self) -> bool:
        step = get_timestep()
        if step < self.warmup_steps:
            return True
        total_steps = self._get_total_steps()
        if total_steps is None:
            return False
        return step >= max(0, total_steps - self.cooldown_steps)

    def is_anchor_step(self) -> bool:
        step = get_timestep()
        return step in self.anchor_step_list[self._branch_key()]

    def is_anchor_next_step(self) -> bool:
        step = get_timestep()
        return (step + 1) in self.anchor_step_list[self._branch_key()]

    def _get_output_dir(self) -> str:
        env_output = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if env_output:
            if self.task_id:
                return task_logs_dir(env_output, self.task_id)
            return debug_output_dir(env_output)
        task = DiffusionBackend.generator.current_task
        if task is not None and task.req is not None and task.req.params is not None:
            if task.req.params.save_dir:
                return os.path.join(task.req.params.save_dir, "..", "logs")
        args = getattr(DiffusionBackend, "args", None)
        output_cfg = getattr(args, "output", None) if args is not None else None
        root_dir = getattr(output_cfg, "root_dir", None) if output_cfg is not None else None
        return str(root_dir) if root_dir else "./outputs"

    @staticmethod
    def _merge_decision_code(prev: int, curr: int) -> int:
        if prev == curr:
            return prev
        if prev == DiTangoPlanner.DECISION_CODE_ANCHOR or curr == DiTangoPlanner.DECISION_CODE_ANCHOR:
            return DiTangoPlanner.DECISION_CODE_ANCHOR
        if prev == DiTangoPlanner.DECISION_CODE_WARMUP_COOLDOWN or curr == DiTangoPlanner.DECISION_CODE_WARMUP_COOLDOWN:
            return DiTangoPlanner.DECISION_CODE_WARMUP_COOLDOWN
        if prev == DiTangoPlanner.DECISION_CODE_COMPUTE or curr == DiTangoPlanner.DECISION_CODE_COMPUTE:
            return DiTangoPlanner.DECISION_CODE_COMPUTE
        return DiTangoPlanner.DECISION_CODE_REUSE

    def record_group_decision(self, layer_id: int, group_id: int, decision_code: int) -> None:
        if self.total_layers is None:
            return
        step = get_timestep()
        branch_key = self._branch_key()
        key = (step, layer_id, group_id)
        branch_records = self._decision_vis_records.setdefault(branch_key, {})
        prev = branch_records.get(key)
        branch_records[key] = decision_code if prev is None else self._merge_decision_code(prev, decision_code)
        self._record_policy_trace(branch_key, step, layer_id, group_id, branch_records[key])
        self._decision_vis_max_step = max(self._decision_vis_max_step, step)

    def _record_policy_trace(self, branch_key: str, step: int, layer_id: int, group_id: int, decision_code: int) -> None:
        interval = None
        intervals = self.group_intervals.get(branch_key)
        if intervals is not None and layer_id < intervals.shape[0] and group_id < intervals.shape[1]:
            interval = int(intervals[layer_id, group_id].item())

        curvature = None
        curvature_mat = self.group_curvature.get(branch_key)
        if curvature_mat is not None and layer_id < curvature_mat.shape[0] and group_id < curvature_mat.shape[1]:
            curvature = float(curvature_mat[layer_id, group_id].item())

        anchor_step = self.last_anchor_step.get(branch_key, -1)
        age = None if anchor_step < 0 else max(0, int(step) - int(anchor_step))
        self._policy_trace_records.setdefault(branch_key, {})[(step, layer_id, group_id)] = {
            "branch": branch_key,
            "step": int(step),
            "layer": int(layer_id),
            "group": int(group_id),
            "decision": self._decision_name(decision_code),
            "decision_code": int(decision_code),
            "curvature": curvature,
            "interval": interval,
            "age": age,
        }

    @staticmethod
    def _decision_name(decision_code: int) -> str:
        if decision_code == DiTangoPlanner.DECISION_CODE_WARMUP_COOLDOWN:
            return "warmup_cooldown"
        if decision_code == DiTangoPlanner.DECISION_CODE_ANCHOR:
            return "anchor"
        if decision_code == DiTangoPlanner.DECISION_CODE_COMPUTE:
            return "compute"
        if decision_code == DiTangoPlanner.DECISION_CODE_REUSE:
            return "reuse"
        return "unknown"

    def record_step_fallback(self, layer_id: int, fallback_count: int) -> None:
        if fallback_count <= 0:
            return
        branch_key = self._branch_key()
        step = get_timestep()
        key = (step, layer_id)
        branch_fallback = self._step_fallback_counts.setdefault(branch_key, {})
        branch_fallback[key] = branch_fallback.get(key, 0) + int(fallback_count)

    def dump_compute_reuse_visualization(self) -> None:
        if not self._should_log_summary() or self._decision_vis_max_step < 0:
            return
        if self.total_layers is None or self.total_layers <= 0 or self.group_num <= 0:
            return
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        merged_records: Dict[Tuple[int, int, int], int] = {}
        for branch_key in ("pos", "neg"):
            for key, code in self._decision_vis_records.get(branch_key, {}).items():
                prev = merged_records.get(key)
                merged_records[key] = code if prev is None else self._merge_decision_code(prev, code)
        if not merged_records:
            return
        merged_path = os.path.join(output_dir, "ditango_policy_step_layer_group.ppm")
        save_ditango_decision_ppm(
            records=merged_records,
            max_step=self._decision_vis_max_step,
            total_layers=self.total_layers,
            group_num=self.group_num,
            ppm_path=merged_path,
        )
        logger.info("[DiTango] Saved decision PPM to %s", merged_path)
        trace_path = os.path.join(output_dir, "ditango_policy_trace.json")
        self._dump_policy_trace(trace_path)
        logger.info("[DiTango] Saved policy trace to %s", trace_path)

    def _dump_policy_trace(self, trace_path: str) -> None:
        records = []
        for branch_key in ("pos", "neg"):
            records.extend(self._policy_trace_records.get(branch_key, {}).values())
        records.sort(key=lambda item: (item["step"], item["branch"], item["layer"], item["group"]))
        payload = {
            "strategy": self.type,
            "cache_ratio": self.cache_ratio,
            "selective_cache_ratio_target": self._selective_cache_ratio_target(),
            "tau_max": self.tau_max,
            "curvature_interval_power": self.curvature_interval_power,
            "locality_group_compute_boost": self.locality_group_compute_boost,
            "intra_group_size_limit": self.intra_group_size_limit,
            "anchor_interval": self.anchor_interval,
            "groupwise_stagger_period": self.groupwise_stagger_period,
            "groupwise_stagger_fresh_count": self.groupwise_stagger_fresh_count,
            "groupwise_stagger_layer_start": self.groupwise_stagger_layer_start,
            "groupwise_stagger_layer_end": self.groupwise_stagger_layer_end,
            "groupwise_keep_local": self.groupwise_keep_local,
            "groupwise_force_tail_full_layers": self.groupwise_force_tail_full_layers,
            "groupwise_reuse_stale_kv": self.groupwise_reuse_stale_kv,
            "groupwise_local_expand": self.groupwise_local_expand,
            "groupwise_fixed_anchor_steps": self.groupwise_fixed_anchor_steps,
            "groupwise_topk_mode": self.groupwise_topk_mode,
            "groupwise_extra_topk": self.groupwise_extra_topk,
            "groupwise_state_align": self.groupwise_state_align,
            "groupwise_state_align_mode": self.groupwise_state_align_mode,
            "groupwise_state_align_out_scale": self.groupwise_state_align_out_scale,
            "groupwise_state_align_lse_scale": self.groupwise_state_align_lse_scale,
            "groupwise_state_align_distance_tau": self.groupwise_state_align_distance_tau,
            "total_layers": self.total_layers,
            "group_num": self.group_num,
            "records": records,
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _all_reduce_mean(self, tensor: torch.Tensor, group) -> torch.Tensor:
        if group.group_size <= 1:
            return tensor
        group.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(float(group.group_size))
        return tensor

    @Timer.get_timer("ditango_plan_sync")
    def _sync_anchor_matrices(self, curvature_mat: torch.Tensor, weight_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        group = get_world_group() if get_cfg_group().group_size == 2 else get_cp_group()
        curvature_mat = self._all_reduce_mean(curvature_mat, group)
        weight_mat = self._all_reduce_mean(weight_mat, group)
        return curvature_mat, weight_mat

    def _target_branches_for_step(self, branch_key: str) -> Tuple[str, ...]:
        if get_cfg_group().group_size == 1 and branch_key == "neg":
            return ("pos", "neg")
        return (branch_key,)

    def update_anchor_stats(
        self,
        layer_id: int,
        group_states: Dict[int, AttentionState],
        local_state: Optional[AttentionState],
        full_output: Optional[torch.Tensor] = None,
    ) -> None:
        if local_state is None or local_state.is_empty():
            return
        if full_output is not None:
            self.update_full_output_compress(layer_id, full_output)

        ordered_group_ids = list(range(self.group_num))
        lse_scores = []
        for group_id in ordered_group_ids:
            state = local_state if group_id == 0 else group_states.get(group_id, None)
            if state is not None and not state.is_empty():
                lse_scores.append(float(torch.norm(state.lse.float()).item()))
            else:
                lse_scores.append(0.0)

        score_tensor = torch.tensor(lse_scores, dtype=torch.float32)
        if score_tensor.numel() == 0:
            return
        temperature = torch.clamp(torch.std(score_tensor, unbiased=False), min=self.SOFTMAX_TEMPERATURE_MIN)
        normalized_scores = (score_tensor - score_tensor.max()) / temperature
        weight_tensor = torch.softmax(normalized_scores, dim=0)

        for idx, group_id in enumerate(ordered_group_ids):
            key = self._branch_layer_group_key(layer_id, group_id)
            self.group_meta.setdefault(key, {})["weight"] = float(weight_tensor[idx].item())

    def update_full_output_compress(self, layer_id: int, full_output: torch.Tensor) -> None:
        self.full_output_compress[self._branch_layer_key(layer_id)] = full_output.detach().float().mean(dim=-1).clone()

    def _build_anchor_matrices(self, branch_key: str, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self._anchor_matrix_device(branch_key)
        curvature_mat = torch.zeros((self.total_layers, self.group_num), dtype=torch.float32, device=device)
        weight_mat = torch.zeros((self.total_layers, self.group_num), dtype=torch.float32, device=device)
        prev_anchor_step = int(self.last_anchor_step.get(branch_key, -1))
        anchor_gap = max(1, step - prev_anchor_step) if prev_anchor_step >= 0 else max(1, self.anchor_interval)

        for layer_id in range(self.total_layers):
            layer_key = self._explicit_branch_layer_key(branch_key, layer_id)
            curr_out = self.full_output_compress.get(layer_key)
            prev_out = self.prev_anchor_full_output_compress.get(layer_key)
            layer_curv = 0.0
            if curr_out is not None and prev_out is not None and curr_out.shape == prev_out.shape:
                layer_curv = float(torch.norm((curr_out.float() - prev_out.float()).reshape(-1)).item()) / float(anchor_gap)
            for group_id in range(self.group_num):
                meta_key = self._explicit_branch_layer_group_key(branch_key, layer_id, group_id)
                weight = float(self.group_meta.get(meta_key, {}).get("weight", 0.0))
                weight_mat[layer_id, group_id] = weight
                curvature_mat[layer_id, group_id] = layer_curv * weight
            if curr_out is not None:
                self.prev_anchor_full_output_compress[layer_key] = curr_out.detach().clone()

        self.last_anchor_step[branch_key] = step
        return curvature_mat, weight_mat

    def _anchor_matrix_device(self, branch_key: str):
        for layer_id in range(self.total_layers or 0):
            cached = self.full_output_compress.get(self._explicit_branch_layer_key(branch_key, layer_id))
            if cached is not None:
                return cached.device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _merge_cfg1_anchor_matrices(
        self,
        step: int,
        branch_key: str,
        curvature_mat: torch.Tensor,
        weight_mat: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if get_cfg_group().group_size != 1:
            return curvature_mat, weight_mat
        if branch_key == "pos":
            self.cfg1_pos_anchor_cache[step] = (curvature_mat.clone(), weight_mat.clone())
            return None, None
        pos_cached = self.cfg1_pos_anchor_cache.pop(step, None)
        if pos_cached is None:
            return curvature_mat, weight_mat
        pos_curv, pos_weight = pos_cached
        return (curvature_mat + pos_curv) / 2.0, (weight_mat + pos_weight) / 2.0

    def _compute_interval_plan(self, group_curvature: torch.Tensor) -> torch.Tensor:
        if group_curvature.numel() == 0:
            return torch.ones_like(group_curvature, dtype=torch.long)
        curv = torch.clamp(group_curvature.detach().float(), min=self.CURVATURE_EPSILON)
        power = float(self.curvature_interval_power)
        raw_tau = torch.pow(curv, -power) if power > 0.0 else torch.ones_like(curv)
        raw_tau = self._apply_locality_group_compute_boost(raw_tau)
        target_update_rate = max(1.0 - self._selective_cache_ratio_target(), 1.0 / float(self.tau_max))
        inverse_raw_rate = torch.mean(1.0 / torch.clamp(raw_tau, min=self.CURVATURE_EPSILON))
        scale = float(inverse_raw_rate.item()) / max(target_update_rate, self.CURVATURE_EPSILON)
        tau = torch.round(raw_tau * scale).clamp(min=1, max=self.tau_max).to(dtype=torch.long)
        return tau

    def _apply_locality_group_compute_boost(self, raw_tau: torch.Tensor) -> torch.Tensor:
        boost = float(self.locality_group_compute_boost)
        if boost <= 0.0 or raw_tau.ndim != 2 or raw_tau.shape[1] <= 1:
            return raw_tau
        multipliers = torch.ones(raw_tau.shape[1], dtype=raw_tau.dtype, device=raw_tau.device)
        hot_group_count = min(2, raw_tau.shape[1])
        multipliers[:hot_group_count] = 1.0 / (1.0 + boost)
        return raw_tau * multipliers.unsqueeze(0)

    def _selective_cache_ratio_target(self) -> float:
        total_steps = self._get_total_steps()
        if total_steps is None or total_steps <= 0:
            return self.cache_ratio
        cooldown_start = max(0, total_steps - self.cooldown_steps)
        forced_step_set = {
            step
            for step in range(total_steps)
            if step < self.warmup_steps or step >= cooldown_start
        }
        for branch_steps in self.anchor_step_list.values():
            forced_step_set.update(
                step
                for step in branch_steps
                if 0 <= step < total_steps and step >= self.warmup_steps and step < cooldown_start
            )
        forced_steps = len(forced_step_set)
        selective_steps = max(1, total_steps - forced_steps)
        target_reuse_steps = min(float(total_steps), max(0.0, float(self.cache_ratio) * float(total_steps)))
        return max(0.0, min(1.0, target_reuse_steps / float(selective_steps)))

    def _build_plan_from_intervals(self, step: int, branch_key: str) -> None:
        intervals = self.group_intervals.get(branch_key)
        if intervals is None:
            plan = torch.ones((self.total_layers, self.group_num), dtype=torch.bool)
            plan = self._apply_groupwise_stagger_plan(plan, step)
            plan = self._apply_groupwise_topk_plan(plan, step, branch_key)
            self.ditango_plan[branch_key] = plan
            return
        last_anchor = int(self.last_anchor_step.get(branch_key, step))
        age = max(1, int(step) - last_anchor)
        interval_cpu = intervals.detach().cpu().clamp(min=1)
        plan = (age % interval_cpu) == 0
        plan = self._apply_groupwise_stagger_plan(plan, step)
        plan = self._apply_groupwise_topk_plan(plan, step, branch_key)
        self.ditango_plan[branch_key] = plan.to(dtype=torch.bool)

    def _groupwise_enabled(self) -> bool:
        return (
            self.groupwise_local_expand >= 0
            or (self.groupwise_stagger_period > 0 and self.groupwise_stagger_fresh_count > 0)
        ) and self.group_num > 1

    def _groupwise_layer_bounds(self) -> tuple[int, int]:
        if self.total_layers is None or self.total_layers <= 0:
            return (0, -1)
        start = min(max(0, self.groupwise_stagger_layer_start), self.total_layers - 1)
        if self.groupwise_stagger_layer_end < 0:
            end = self.total_layers - 1
        else:
            end = min(max(start, self.groupwise_stagger_layer_end), self.total_layers - 1)
        if self.groupwise_force_tail_full_layers > 0:
            end = min(end, max(start - 1, self.total_layers - self.groupwise_force_tail_full_layers - 1))
        return start, end

    def _groupwise_fresh_groups(self, step: int) -> set[int]:
        if not self._groupwise_enabled():
            return set()
        if self.groupwise_local_expand >= 0:
            expand = min(max(0, self.groupwise_local_expand), max(0, self.group_num - 1))
            groups = {0}
            for distance in range(1, expand + 1):
                groups.add(distance % self.group_num)
                groups.add((-distance) % self.group_num)
            return groups
        fresh_count = min(self.group_num, self.groupwise_stagger_fresh_count)
        phase = max(0, int(step) - self.warmup_steps)
        start_group = (phase * fresh_count) % self.group_num
        groups = {(start_group + offset) % self.group_num for offset in range(fresh_count)}
        if self.groupwise_keep_local:
            groups.add(0)
        return groups

    def _apply_groupwise_stagger_plan(self, plan: torch.Tensor, step: int) -> torch.Tensor:
        if not self._groupwise_enabled() or self.total_layers is None:
            return plan
        stagger_start, stagger_end = self._groupwise_layer_bounds()
        if stagger_end < stagger_start:
            return plan
        result = plan.clone().to(dtype=torch.bool)
        if self.groupwise_force_tail_full_layers > 0:
            tail_start = max(0, self.total_layers - self.groupwise_force_tail_full_layers)
            result[tail_start:, :] = True
        fresh_groups = self._groupwise_fresh_groups(step)
        for layer_id in range(stagger_start, stagger_end + 1):
            result[layer_id, :] = False
            for group_id in fresh_groups:
                if 0 <= group_id < self.group_num:
                    result[layer_id, group_id] = True
        return result

    def _groupwise_topk_enabled(self) -> bool:
        return self.group_num > 1 and self.groupwise_topk_mode not in {"", "none", "off"} and self.groupwise_extra_topk > 0

    def _apply_groupwise_topk_plan(self, plan: torch.Tensor, step: int, branch_key: str) -> torch.Tensor:
        if not self._groupwise_topk_enabled() or self.total_layers is None:
            return plan
        topk_start, topk_end = self._groupwise_layer_bounds()
        if topk_end < topk_start:
            return plan
        result = plan.clone().to(dtype=torch.bool)
        if self.groupwise_force_tail_full_layers > 0:
            tail_start = max(0, self.total_layers - self.groupwise_force_tail_full_layers)
            result[tail_start:, :] = True

        k = min(max(0, self.groupwise_extra_topk), max(0, self.group_num - 1))
        score_matrix = self._cached_group_score_matrix(branch_key, step)
        for layer_id in range(topk_start, topk_end + 1):
            result[layer_id, :] = False
            result[layer_id, 0] = True
            for group_id in self._rank_cached_groups(score_matrix, layer_id)[:k]:
                result[layer_id, group_id] = True
        return result

    def _rank_cached_groups(self, score_matrix: torch.Tensor, layer_id: int) -> list[int]:
        candidates = []
        for group_id in range(1, self.group_num):
            score = float(score_matrix[layer_id, group_id].item())
            candidates.append((score, -group_id, group_id))
        candidates.sort(reverse=True)
        return [group_id for _, _, group_id in candidates]

    def _cached_group_score_matrix(self, branch_key: str, step: int) -> torch.Tensor:
        scores = torch.full((self.total_layers, self.group_num), float("-inf"), dtype=torch.float32)
        for layer_id in range(self.total_layers):
            for group_id in range(1, self.group_num):
                scores[layer_id, group_id] = self._cached_group_score(branch_key, layer_id, group_id, step)
        if self.groupwise_topk_mode in {"cached_lse", "cached_lse_abs"} and get_cp_group().group_size > 1:
            device = self._anchor_matrix_device(branch_key)
            sync_scores = scores.to(device=device)
            missing_mask = torch.isneginf(sync_scores)
            sync_scores = torch.where(missing_mask, torch.zeros_like(sync_scores), sync_scores)
            counts = (~missing_mask).to(dtype=torch.float32)
            get_cp_group().all_reduce(sync_scores, op=torch.distributed.ReduceOp.SUM)
            get_cp_group().all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            sync_scores = torch.where(counts > 0, sync_scores / torch.clamp(counts, min=1.0), torch.full_like(sync_scores, float("-inf")))
            scores = sync_scores.cpu()
        return scores

    def _cached_group_score(self, branch_key: str, layer_id: int, group_id: int, step: int) -> float:
        mode = self.groupwise_topk_mode
        if mode in {"cached_lse", "cached_lse_abs"}:
            key = self._explicit_branch_layer_group_key(branch_key, layer_id, group_id)
            state = self.state_cache.get(key)
            if state is None or state.is_empty():
                return float("-inf")
            lse_mean = float(state.lse.detach().float().mean().item())
            return abs(lse_mean) if mode == "cached_lse_abs" else lse_mean
        if mode == "nearest":
            return float(-min(group_id, self.group_num - group_id))
        if mode == "farthest":
            return float(min(group_id, self.group_num - group_id))
        if mode == "round_robin":
            return 1.0 if group_id == (max(0, int(step) - self.warmup_steps) % max(1, self.group_num)) else 0.0
        return float("-inf")

    def _schedule_next_anchor(self, step: int, target_branches: Tuple[str, ...]) -> None:
        if self.fixed_anchor_steps:
            return
        total_steps = self._get_total_steps()
        next_step = step + max(1, self.anchor_interval)
        if total_steps is not None and next_step >= max(0, total_steps - self.cooldown_steps):
            return
        for branch_key in target_branches:
            if next_step not in self.anchor_step_list[branch_key]:
                self.anchor_step_list[branch_key].append(next_step)
                self.anchor_step_list[branch_key].sort()

    def _log_anchor_summary(self, step: int, branch_key: str, target_branches: Tuple[str, ...]) -> None:
        if not self._should_log_summary():
            return
        curv = self.group_curvature.get(branch_key)
        intervals = self.group_intervals.get(branch_key)
        if curv is None or intervals is None:
            logger.info(
                "[DiTango Anchor t%d-%s] bootstrap curvature_interval_power=%.4f",
                step,
                branch_key,
                self.curvature_interval_power,
            )
            return
        logger.info(
            "[DiTango Anchor t%d-%s] targets=%s curvature(mean=%.6e,max=%.6e) "
            "interval(min=%d,max=%d,mean=%.3f) cache_ratio=%.3f selective_cache_ratio=%.3f "
            "curvature_interval_power=%.4f locality_group_compute_boost=%.3f "
            "groupwise(period=%d,fresh=%d,layers=%d..%d,keep_local=%s,tail_full=%d,local_expand=%d,"
            "fixed_anchors=%s,topk_mode=%s,extra_topk=%d,state_align=%s,mode=%s,"
            "out_scale=%.3f,lse_scale=%.3f,tau=%.3f) "
            "stale_kv=%s group_num=%d intra_group_size_limit=%d",
            step,
            branch_key,
            target_branches,
            float(curv.float().mean().item()),
            float(curv.float().max().item()),
            int(intervals.min().item()),
            int(intervals.max().item()),
            float(intervals.float().mean().item()),
            self.cache_ratio,
            self._selective_cache_ratio_target(),
            self.curvature_interval_power,
            self.locality_group_compute_boost,
            self.groupwise_stagger_period,
            self.groupwise_stagger_fresh_count,
            self.groupwise_stagger_layer_start,
            self.groupwise_stagger_layer_end,
            self.groupwise_keep_local,
            self.groupwise_force_tail_full_layers,
            self.groupwise_local_expand,
            self.groupwise_fixed_anchor_steps,
            self.groupwise_topk_mode,
            self.groupwise_extra_topk,
            self.groupwise_state_align,
            self.groupwise_state_align_mode,
            self.groupwise_state_align_out_scale,
            self.groupwise_state_align_lse_scale,
            self.groupwise_state_align_distance_tau,
            self.groupwise_reuse_stale_kv,
            self.group_num,
            self.intra_group_size_limit,
        )

    def run_ditango_planner(self, layer_id: int) -> None:
        if self.is_warmup_or_cooldown_step() and not self.is_anchor_step():
            return
        if not self._is_last_layer(layer_id):
            return

        step = get_timestep()
        branch_key = self._branch_key()
        target_branches = self._target_branches_for_step(branch_key)

        if self.is_anchor_step():
            curvature_mat, weight_mat = self._build_anchor_matrices(branch_key, step)
            curvature_mat, weight_mat = self._merge_cfg1_anchor_matrices(step, branch_key, curvature_mat, weight_mat)
            if curvature_mat is None or weight_mat is None:
                return
            curvature_mat, weight_mat = self._sync_anchor_matrices(curvature_mat, weight_mat)

            has_signal = bool(torch.any(curvature_mat > self.CURVATURE_EPSILON).item())
            for target_branch in target_branches:
                self.group_curvature[target_branch] = curvature_mat.detach().cpu()
                self.layer_curvature[target_branch] = curvature_mat.sum(dim=1).detach().cpu()
                if has_signal:
                    intervals = self._compute_interval_plan(curvature_mat).detach().cpu()
                    self.group_intervals[target_branch] = intervals
                    self._anchor_has_plan[target_branch] = True
                    self._build_plan_from_intervals(step + 1, target_branch)
                else:
                    self.ditango_plan[target_branch] = torch.ones((self.total_layers, self.group_num), dtype=torch.bool)
                self._refresh_policy_trace_metadata(step, target_branch)

            if has_signal:
                self._schedule_next_anchor(step, target_branches)
            elif not self.fixed_anchor_steps:
                for target_branch in target_branches:
                    next_step = step + 1
                    if next_step not in self.anchor_step_list[target_branch]:
                        self.anchor_step_list[target_branch].append(next_step)
                        self.anchor_step_list[target_branch].sort()
            self._log_anchor_summary(step, branch_key, target_branches)
            return

        for target_branch in target_branches:
            self._build_plan_from_intervals(step + 1, target_branch)

    def _refresh_policy_trace_metadata(self, step: int, branch_key: str) -> None:
        branch_records = self._decision_vis_records.get(branch_key, {})
        for layer_id in range(self.total_layers):
            for group_id in range(self.group_num):
                code = branch_records.get((step, layer_id, group_id))
                if code is not None:
                    self._record_policy_trace(branch_key, step, layer_id, group_id, code)

    def get_reuse_key(self, layer_id: int, group_id: int = 0, **kwargs) -> str:
        return self._branch_layer_group_key(layer_id, group_id)

    def get_compressed_reuse_key(self, layer_id: int, tag: str) -> str:
        return f"{self._branch_layer_key(layer_id)}_compressed_{tag}"

    def reuse(self, layer_id: int, group_id: int = 0, **kwargs) -> AttentionState:
        key = self.get_reuse_key(layer_id, group_id=group_id)
        return self.state_cache.get(key, AttentionState())

    def reuse_compressed(self, layer_id: int, tag: str) -> AttentionState:
        key = self.get_compressed_reuse_key(layer_id, tag)
        return self.compressed_state_cache.get(key, AttentionState())

    def reuse_kv(
        self,
        layer_id: int,
        group_id: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        key = self.get_reuse_key(layer_id, group_id=group_id)
        return self.kv_cache.get(key)

    def get_store_key(self, layer_id: int, group_id: int = 0, **kwargs) -> str:
        return self._branch_layer_group_key(layer_id, group_id)

    def store(self, layer_id: int, group_id: int, group_state: Optional[AttentionState], **kwargs) -> None:
        if (self.is_warmup_or_cooldown_step() and not self.is_anchor_step()) or group_state is None or group_state.is_empty():
            return
        key = self.get_store_key(layer_id, group_id=group_id)
        self.state_cache[key] = group_state
        self.group_meta.setdefault(key, {})["last_compute_step"] = get_timestep()

    def store_compressed(self, layer_id: int, tag: str, state: Optional[AttentionState]) -> None:
        key = self.get_compressed_reuse_key(layer_id, tag)
        if state is None or state.is_empty():
            self.compressed_state_cache.pop(key, None)
            return
        self.compressed_state_cache[key] = state

    def discard_compressed_state(self, layer_id: int, tag: str) -> None:
        key = self.get_compressed_reuse_key(layer_id, tag)
        self.compressed_state_cache.pop(key, None)

    def store_kv(
        self,
        layer_id: int,
        group_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_k: Optional[torch.Tensor] = None,
    ) -> None:
        if self.is_warmup_or_cooldown_step() and not self.is_anchor_step():
            return
        key = self.get_store_key(layer_id, group_id=group_id)
        cached_cu = None if cu_seqlens_k is None else cu_seqlens_k.detach().clone()
        self.kv_cache[key] = (k.detach().clone(), v.detach().clone(), cached_cu)
        self.group_meta.setdefault(key, {})["last_kv_step"] = get_timestep()

    def cached_kv_step(self, layer_id: int, group_id: int) -> Optional[int]:
        key = self.get_reuse_key(layer_id, group_id=group_id)
        step = self.group_meta.get(key, {}).get("last_kv_step")
        return None if step is None else int(step)

    def discard_group_state(self, layer_id: int, group_id: int) -> None:
        key = self.get_store_key(layer_id, group_id=group_id)
        self.state_cache.pop(key, None)
        self.kv_cache.pop(key, None)

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        from chitu_diffusion.ditango.runtime import DitangoAttention

        if not hasattr(module, "_original_attn"):
            module._original_attn = module.blocks[0].self_attn.attn_func

        self.total_layers = len(module.blocks)
        for layer_id, block in enumerate(module.blocks):
            block.self_attn.attn_func = DitangoAttention(layer_id=layer_id)
        logger.info("Module %s wrapped with ditango strategy", module.__class__.__name__)

    def unwrap_module(self, module: torch.nn.Module) -> None:
        if hasattr(module, "_original_attn"):
            for _, block in enumerate(module.blocks):
                block.self_attn.attn_func = module._original_attn
            delattr(module, "_original_attn")
        self.dump_compute_reuse_visualization()
        logger.info("Module %s unwrapped from ditango strategy", module.__class__.__name__)
