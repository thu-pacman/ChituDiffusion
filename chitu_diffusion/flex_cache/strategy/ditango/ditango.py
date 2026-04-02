import torch
import os
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

from chitu_core.distributed.parallel_state import get_cfg_group, get_cp_group, get_up_group, get_world_group
from chitu_core.logging_utils import should_log_info_on_rank
from chitu_diffusion.backend import CFGType, DiffusionBackend
from chitu_diffusion.flex_cache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.flex_cache.strategy.ditango.ppm_visualizer import save_ditango_decision_ppm
from chitu_diffusion.task import DiffusionTask
from chitu_diffusion.utils.shared_utils import (
    async_ring_p2p_commit,
    async_ring_p2p_wait_and_update,
    squeeze_and_transpose,
    update_out_and_lse,
)
from chitu_diffusion.bench.timer import Timer

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
    DEFAULT_INTRA_GROUP_SIZE_LIMIT = 2
    DEFAULT_ENABLE_DYNAMIC_COMPOSE = False
    DEFAULT_WARMUP_STEPS = 5
    DEFAULT_COOLDOWN_STEPS = 5
    DEFAULT_ANCHOR_LOG_LAYERS = (28,)
    DECISION_CODE_WARMUP_COOLDOWN = 0
    DECISION_CODE_ANCHOR = 1
    DECISION_CODE_COMPUTE = 2
    DECISION_CODE_REUSE = 3
    REL_ERR_EPSILON = 1e-8
    SOFTMAX_TEMPERATURE_MIN = 1e-6

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

        self.anchor_interval = max(1, int(anchor_interval))
        self.tau_max = max(1, int(tau_max))
        self.intra_group_size_limit = max(1, int(intra_group_size_limit))
        self.enable_dynamic_compose = bool(enable_dynamic_compose)
        self.warmup_steps = max(1, int(warmup_steps))
        self.cooldown_steps = max(1, int(cooldown_steps))
        self.anchor_log_layers = set(self.DEFAULT_ANCHOR_LOG_LAYERS)
        self.total_layers = None
        self.effective_group_size = min(get_cp_group().group_size, self.intra_group_size_limit)

        self.ref_local_state_compress: Dict[str, torch.Tensor] = {}
        self.anchor_local_state_compress: Dict[str, torch.Tensor] = {}
        self.curr_local_state_compress: Dict[str, torch.Tensor] = {}
        self.group_num = get_cp_group().group_size // self.intra_group_size_limit
        self.ditango_plan: Dict[str, torch.Tensor] = {} # {branch_key, tensor[layer_num, group_num]: bool}

        self.anchor_step_list: Dict[str, list] = {}
        # anchor step decision
        self.anchor_rel_err_threshold = 0.6 * self.cache_ratio
        # ase threshold decision
        self.ase_threshold = 0.0
        

        self.reset_state()

    def reset_state(self) -> None:
        self.curr_local_state_compress = {}
        self.anchor_local_state_compress = {}
        self.ref_local_state_compress = {}
        self.ditango_plan.clear()
        self.anchor_step_list = {"pos": [self.warmup_steps], "neg": [self.warmup_steps]}
        self.cfg1_pos_error_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        self.group_meta: Dict[str, Dict[str, Any]] = {}
        self.last_anchor_step: Dict[str, int] = {}

        self.layerwise_ase_thresh: Dict[str, float] = {}

        self.last_threshold_log_step: int = -1

        self.anchor_step_decision_pos_cache: Dict[int, bool] = {}
        self.step_layer_plan_pos_cache: Dict[Tuple[int, int], Tuple[Dict[int, bool], Dict[int, float]]] = {}

        self._decision_vis_records: Dict[str, Dict[Tuple[int, int, int], int]] = {"pos": {}, "neg": {}}
        self._decision_vis_max_step: int = -1
        self._step_fallback_counts: Dict[str, Dict[Tuple[int, int], int]] = {"pos": {}, "neg": {}}


        if DiffusionBackend.flexcache is not None:
            DiffusionBackend.flexcache.cache.clear()
        logger.debug("DiTango v3 state reset")


    def _should_log(self, step: int, layer_id: int) -> bool:
        if self._branch_key() == "pos" and (layer_id in self.anchor_log_layers):
            return True
        return False

    def _should_log_summary(self) -> bool:
        return should_log_info_on_rank() and (get_cp_group().rank_in_group == 0)

    def _get_output_dir(self) -> str:
        task = DiffusionBackend.generator.current_task
        if task is not None and task.req is not None and task.req.params is not None:
            if task.req.params.save_dir:
                return task.req.params.save_dir

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

    @staticmethod
    def _merge_decision_code(prev: int, curr: int) -> int:
        if prev == curr:
            return prev
        if prev == DiTangoV3Strategy.DECISION_CODE_ANCHOR or curr == DiTangoV3Strategy.DECISION_CODE_ANCHOR:
            return DiTangoV3Strategy.DECISION_CODE_ANCHOR
        if prev == DiTangoV3Strategy.DECISION_CODE_WARMUP_COOLDOWN or curr == DiTangoV3Strategy.DECISION_CODE_WARMUP_COOLDOWN:
            return DiTangoV3Strategy.DECISION_CODE_WARMUP_COOLDOWN
        if prev == DiTangoV3Strategy.DECISION_CODE_COMPUTE or curr == DiTangoV3Strategy.DECISION_CODE_COMPUTE:
            return DiTangoV3Strategy.DECISION_CODE_COMPUTE
        return DiTangoV3Strategy.DECISION_CODE_REUSE

    def record_group_decision(self, layer_id: int, group_id: int, decision_code: int) -> None:
        if self.total_layers is None:
            return
        step = get_timestep()
        branch_key = self._branch_key()
        key = (step, layer_id, group_id)
        branch_records = self._decision_vis_records.setdefault(branch_key, {})
        prev = branch_records.get(key)
        if prev is None:
            branch_records[key] = decision_code
        else:
            branch_records[key] = self._merge_decision_code(prev, decision_code)
        self._decision_vis_max_step = max(self._decision_vis_max_step, step)

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

        # POS/NEG decisions are synchronized; emit a single merged overview.
        merged_records: Dict[Tuple[int, int, int], int] = {}
        for branch_key in ("pos", "neg"):
            branch_records = self._decision_vis_records.get(branch_key, {})
            for key, code in branch_records.items():
                prev = merged_records.get(key)
                merged_records[key] = code if prev is None else self._merge_decision_code(prev, code)

        if merged_records:
            merged_path = os.path.join(output_dir, "ditango_policy_step_layer_group.ppm")
            save_ditango_decision_ppm(
                records=merged_records,
                max_step=self._decision_vis_max_step,
                total_layers=self.total_layers,
                group_num=self.group_num,
                ppm_path=merged_path,
            )
            logger.info(
                "[DiTango] Saved decision PPM (merged) to %s | "
                "color map: warmup/cooldown=gray, anchor=green, compute=blue, reuse=yellow",
                merged_path,
            )

    def _log_step_summary(
        self,
        step: int,
        source_branch: str,
        ase_mat: torch.Tensor,
        rel_vec: torch.Tensor,
        target_branches: Tuple[str, ...],
    ) -> None:
        if not self._should_log_summary():
            return

        mode = "warmup/cooldown" if self.is_warmup_or_cooldown_step() else ("anchor" if self.is_anchor_step() else "selective")
        rel_items = [f"L{idx}:{float(val):.3e}" for idx, val in enumerate(rel_vec.float().tolist())]

        group_items = []
        for group_id in range(self.group_num):
            if self._is_local_partition(group_id):
                group_items.append(f"g{group_id}:local")
                continue
            group_values = ase_mat[:, group_id].float()
            mean_ase = float(group_values.mean().item())
            compute_ratio = 0.0
            plan = self.ditango_plan.get(source_branch)
            if plan is not None and plan.numel() > 0:
                compute_ratio = float(plan[:, group_id].float().mean().item())
            group_items.append(f"g{group_id}:ase_mean={mean_ase:.3e},compute={compute_ratio:.2f}")

        fallback_total = 0
        for branch_key in target_branches:
            branch_fallback = self._step_fallback_counts.get(branch_key, {})
            for layer_id in range(self.total_layers):
                fallback_total += int(branch_fallback.pop((step, layer_id), 0))

        logger.info(
            f"[DiTango Step t{step}] mode={mode} branch={source_branch} targets={target_branches} "
            f"ase_threshold={self.ase_threshold:.3e} anchor_next={self.is_anchor_next_step()} fallback={fallback_total}"
        )
        logger.info("[DiTango Layerwise] " + " | ".join(rel_items))
        logger.info("[DiTango Groupwise] " + " | ".join(group_items))

    def _get_group_local_ref_compress(self, layer_id: int, group_id: int) -> Optional[torch.Tensor]:
        group_key = self._branch_layer_group_key(layer_id, group_id)
        ref = self.ref_local_state_compress.get(group_key, None)
        if ref is None:
            return None
        return ref

    def _branch_local_key(self, layer_id: int) -> str:
        return f"{self._branch_key()}_{layer_id}_local"
    

    def set_curr_local_state_compress(self, layer_id: int, local_state: AttentionState) -> None:
        """Update compressed local state used by planner drift estimation."""
        assert local_state is not None and not local_state.is_empty()
        local_key = self._branch_local_key(layer_id)
        comp = local_state.out.float().mean(dim=-1)
        self.curr_local_state_compress[local_key] = comp.clone()


    def _update_group_local_ref_compress(self, layer_id: int, group_id: int) -> None:
        local_key = self._branch_local_key(layer_id)
        group_key = self._branch_layer_group_key(layer_id, group_id)
        if self.curr_local_state_compress[local_key] is not None:
            self.ref_local_state_compress[group_key] = self.curr_local_state_compress[local_key].clone()

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

    def _branch_layer_key(self, layer_id: int) -> str:
        return f"{self._branch_key()}_{layer_id}"

    def _branch_layer_group_key(self, layer_id: int, group_id: int) -> str:
        return f"{self._branch_layer_key(layer_id)}_group_{group_id}"

    def _is_last_layer(self, layer_id: int) -> bool:
        if self.total_layers is None:
            return False
        return layer_id == (self.total_layers - 1)

    def _is_local_partition(self, group_id: int) -> bool:
        # In per-rank rotated view, group 0 contains local state. If group size is 1,
        # group 0 is local-only and must not participate in planner decisions.
        if group_id == 0 and self.effective_group_size == 1:
            return True
        return False


    def is_anchor_step(self) -> bool:
        curr_step = get_timestep()
        branch_key = self._branch_key()
        if curr_step in self.anchor_step_list[branch_key]:
            return True
        else:
            return False

    def is_anchor_next_step(self) -> bool:
        curr_step = get_timestep()
        branch_key = self._branch_key()
        if (curr_step - 1) in self.anchor_step_list[branch_key]:
            return True
        else:
            return False


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

    

    def estimate_group_error(self, layer_id: int) -> None:
        local_key = self._branch_local_key(layer_id)
        for group_id in range(self.group_num):
            key = self._branch_layer_group_key(layer_id, group_id)
            if self._is_local_partition(group_id):
                self.group_meta.setdefault(key, {})["ase"] = 0.0
                continue

            w = float(self.group_meta.get(key, {}).get("weight", 0.0))
            group_ref = self._get_group_local_ref_compress(layer_id, group_id)
            if self.curr_local_state_compress.get(local_key) is None or group_ref is None:
                drift = float("inf")
            else:
                drift = float(torch.norm(self.curr_local_state_compress[local_key] - group_ref).item())

            ase = w * drift
            self.group_meta[key]["ase"] = ase
        

    def _decide_ase_threshold(self) -> None:
        ase_list = []
        for layer_id in range(self.total_layers):
            for group_id in range(self.group_num):
                if self._is_local_partition(group_id):
                    continue
                key = self._branch_layer_group_key(layer_id, group_id)
                ase = self.group_meta.get(key, {}).get("ase", None)
                if ase is not None and not torch.isinf(torch.tensor(ase)):
                    ase_list.append(ase)
        ase_sort = sorted(ase_list)
        # 取 cache ratio 分位
        if ase_sort:
            idx = min(len(ase_sort) - 1, int(len(ase_sort) * self.cache_ratio))
            self.ase_threshold = ase_sort[idx]
        else:
            self.ase_threshold = float("inf")
        logger.info(f"\n[ASE Threshold t{get_timestep()}] ase thresh: {self.ase_threshold:.3f}, {ase_list=}")

    def _all_reduce_mean(self, tensor: torch.Tensor, group) -> torch.Tensor:
        if group.group_size <= 1:
            return tensor
        group.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(float(group.group_size))
        return tensor

    def _build_error_matrices(self, branch_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        device = torch.cuda.current_device()
        ase_mat = torch.zeros((self.total_layers, self.group_num), dtype=torch.float16, device=device)
        rel_vec = torch.zeros((self.total_layers,), dtype=torch.float16, device=device)

        for layer_id in range(self.total_layers):
            # 1. group wise ase error
            for group_id in range(self.group_num):
                key = f"{branch_key}_{layer_id}_group_{group_id}"
                ase = float(self.group_meta.get(key, {}).get("ase", 0.0))
                if not torch.isfinite(torch.tensor(ase)):
                    ase = 0.0
                ase_mat[layer_id, group_id] = ase

                if self._is_local_partition(group_id):
                    continue
            # 2. layer wise rel error
            anchor_state = self.anchor_local_state_compress.get(self._branch_local_key(layer_id), None)
            curr_state = self.curr_local_state_compress.get(self._branch_local_key(layer_id), None)
            if anchor_state is not None and curr_state is not None:
                rel_err = float(
                    torch.norm(curr_state - anchor_state).item()
                    / (torch.norm(anchor_state).item() + self.REL_ERR_EPSILON)
                )
                rel_vec[layer_id] = rel_err
            else:
                rel_vec[layer_id] = 0.0
        return ase_mat, rel_vec

    @Timer.get_timer("ditango_plan_sync")
    def _sync_error_matrices(self, ase_mat: torch.Tensor, rel_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if get_cfg_group().group_size == 2:
            group = get_world_group()
        else:
            group = get_cp_group()
        ase_mat = self._all_reduce_mean(ase_mat, group)
        rel_vec = self._all_reduce_mean(rel_vec, group)
        return ase_mat, rel_vec

    def _decide_ase_threshold_from_matrix(self, ase_mat: torch.Tensor, step: int, branch_key: str) -> float:
        ase_list = []
        for group_id in range(self.group_num):
            if self._is_local_partition(group_id):
                continue
            values = ase_mat[:, group_id].float().tolist()
            ase_list.extend(values)

        ase_sort = sorted(ase_list)
        if ase_sort:
            idx = min(len(ase_sort) - 1, int(len(ase_sort) * self.cache_ratio))
            ase_threshold = float(ase_sort[idx])
        else:
            ase_threshold = float("inf")

        if get_cp_group().rank_in_group == 0:
            logger.info(f"\n[ASE Threshold setup!! t{step}-{branch_key}] ase thresh: {ase_threshold:.3f}")
        return ase_threshold

    def _apply_anchor_decision_from_rel(self, rel_vec: torch.Tensor, step: int, target_branches: Tuple[str, ...]) -> None:
        if self.is_anchor_step() or self.is_warmup_or_cooldown_step():
            return

        mean_rel_err = float(rel_vec.float().mean().item())
        should_anchor_next = mean_rel_err >= self.anchor_rel_err_threshold
        for branch_key in target_branches:
            if should_anchor_next and (step + 1) not in self.anchor_step_list[branch_key]:
                self.anchor_step_list[branch_key].append(step + 1)
                self.anchor_step_list[branch_key].sort()

        # logger.info(
        #     f"\n[Stepwise Anchor Plan t{step}-{self._branch_key()}] mean_rel_err={mean_rel_err:.3f} "
        #     f"th={self.anchor_rel_err_threshold:.3f} anchor_next={should_anchor_next}"
        # )

    def _build_plan_from_errors(
        self,
        ase_mat: torch.Tensor,
        step: int,
        branch_key: str,
    ) -> None:
        self.ditango_plan[branch_key] = torch.zeros(
            (self.total_layers, self.group_num), dtype=torch.bool
        )

        # ase_log = f"\n[ASE log] t{step}-{branch_key} "
        if self.is_anchor_step():
            # ase_log += "Anchor step - full compute, full reuse next step.\n"
            if self._is_local_partition(0):
                self.ditango_plan[branch_key][:, 0] = True
        else:
            for layer in range(self.total_layers):
                # ase_log += f"L{layer} |"
                for group_id in range(self.group_num):
                    if self._is_local_partition(group_id):
                        self.ditango_plan[branch_key][layer, group_id] = True
                        # ase_log += "local "
                        continue
                    ase = float(ase_mat[layer, group_id].item())
                    if ase >= self.ase_threshold:
                        self.ditango_plan[branch_key][layer, group_id] = True
                    # ase_log += f"{ase:.2f} "
                # ase_log += "| \n"

        # if get_cp_group().rank_in_group == 0:
        #     logger.info(f"\n[Group Plan] t{step}-{branch_key} " + str(self.ditango_plan))
        #     logger.info(ase_log)

    def _merge_cfg1_branch_errors(
        self,
        step: int,
        branch_key: str,
        ase_mat: torch.Tensor,
        rel_vec: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """For cfg-size=1 flow: cache POS errors and merge once at NEG branch."""
        if get_cfg_group().group_size != 1:
            return ase_mat, rel_vec

        if branch_key == "pos":
            self.cfg1_pos_error_cache[step] = (ase_mat.clone(), rel_vec.clone())
            return None, None

        pos_cached = self.cfg1_pos_error_cache.pop(step, None)
        if pos_cached is None:
            return ase_mat, rel_vec

        pos_ase, pos_rel = pos_cached
        return (ase_mat + pos_ase) / 2.0, (rel_vec + pos_rel) / 2.0

    def _target_branches_for_step(self, branch_key: str) -> Tuple[str, ...]:
        """Return branches that should receive the synchronized plan at this step."""
        if get_cfg_group().group_size == 1 and branch_key == "neg":
            return ("pos", "neg")
        return (branch_key,)
        

    def run_ditango_planner(self, layer_id: int) -> None:
        if self.is_warmup_or_cooldown_step():
            return

        self.estimate_group_error(layer_id)

        if not self._is_last_layer(layer_id):
            return

        step = get_timestep()
        branch_key = self._branch_key()
        ase_mat, rel_vec = self._build_error_matrices(branch_key)

        # cfg=1: cache POS branch error first, then merge at NEG to make one unified decision.
        ase_mat, rel_vec = self._merge_cfg1_branch_errors(step, branch_key, ase_mat, rel_vec)
        if ase_mat is None or rel_vec is None:
            return

        ase_mat, rel_vec = self._sync_error_matrices(ase_mat, rel_vec)

        target_branches = self._target_branches_for_step(branch_key)

        self._apply_anchor_decision_from_rel(rel_vec, step, target_branches)

        if self.is_anchor_next_step():
            self.ase_threshold = self._decide_ase_threshold_from_matrix(ase_mat, step, branch_key)

        for target_branch in target_branches:
            self._build_plan_from_errors(ase_mat, step, target_branch)

        self._log_step_summary(step, branch_key, ase_mat, rel_vec, target_branches)

    def update_anchor_stats(
        self,
        layer_id: int,
        group_states: Dict[int, AttentionState],
        local_state: Optional[AttentionState],
    ) -> None:
        """
        On anchor step, compute group weights and refresh local reference compression.

        Local partition participates in weight normalization but is not a decision group.
        """
        if local_state is None or local_state.is_empty():
            return

        step = get_timestep()

        local_lse_score = float(torch.norm(local_state.lse.float()).item())
        ordered_group_ids = [g for g in range(self.group_num) if not self._is_local_partition(g)]
        lse_scores = []
        for group_id in ordered_group_ids:
            state = group_states.get(group_id, None)
            if state is not None and not state.is_empty():
                lse_scores.append(float(torch.norm(state.lse.float()).item()))
            else:
                lse_scores.append(0.0)

        if not ordered_group_ids:
            logger.info(
                f"[Anchor T{step}l{layer_id}-{self._branch_key()}] local-only step, skip decision-group stats."
            )
            return

        # Local state affects weight normalization but is never a decision group.
        score_tensor = torch.tensor(lse_scores + [local_lse_score], dtype=torch.float32)
        # Use std as temperature to avoid near one-hot weights when raw norms are large.
        temperature = torch.clamp(torch.std(score_tensor, unbiased=False), min=self.SOFTMAX_TEMPERATURE_MIN)
        normalized_scores = (score_tensor - score_tensor.max()) / temperature
        weight_tensor = torch.softmax(normalized_scores, dim=0)

        log_items = []

        for idx, group_id in enumerate(ordered_group_ids):
            key = self._branch_layer_group_key(layer_id, group_id)
            weight = float(weight_tensor[idx].item())
            # Store group meta: alpha and weight 
            self.group_meta[key] = {
                "weight": weight,
                "ase": 0.0,
            }
            log_items.append(f"g{group_id} w={weight:.2f}")

        self.anchor_local_state_compress[self._branch_local_key(layer_id)] = local_state.out.float().mean(dim=-1).clone()

        if log_items and self._should_log_summary():
            local_weight = float(weight_tensor[-1].item())
            logger.debug(
                f"[Anchor weight t{step}l{layer_id}-{self._branch_key()}] "
                f"local_w={local_weight:.2f}| {' | '.join(log_items)}"
            )

    def get_reuse_key(self, layer_id: int, group_id: int = 0, **kwargs) -> str:
        return self._branch_layer_group_key(layer_id, group_id)

    def reuse(self, layer_id: int, group_id: int = 0, **kwargs) -> AttentionState:
        key = self.get_reuse_key(layer_id, group_id=group_id)
        if key in DiffusionBackend.flexcache.cache:
            state = DiffusionBackend.flexcache.cache[key]
            # logger.info(f"Reuse hit for {key} at t{get_timestep()}l{layer_id} | {self._state_digest(state)}")
            return state

        return AttentionState()

    def get_store_key(self, layer_id: int, group_id: int = 0, **kwargs) -> str:
        return self._branch_layer_group_key(layer_id, group_id)

    def store(self, layer_id: int, group_id: int, group_state: Optional[AttentionState], **kwargs) -> None:
        step = get_timestep()
        if self.is_warmup_or_cooldown_step():
            return

        if group_state is None or group_state.is_empty():
            return

        key = self.get_store_key(layer_id, group_id=group_id)
        DiffusionBackend.flexcache.cache[key] = group_state
        self.group_meta.setdefault(key, {})["last_compute_step"] = step

   
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

    def _print_layer0(self, message: str) -> None:
        if self.global_rank == 0 and self.layer_id == 0:
            logger.info(message)
        

    @staticmethod
    def _is_local_partition_step(outer_step: int, inner_step: int) -> bool:
        """In the rotated view, local partition is always the first block."""
        return (outer_step == 0) and (inner_step == 0)

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

    def _generate_selective_comm_pattern(self, group_plan: torch.Tensor): 

        comm_list = []
        plan_list = group_plan.bool().tolist()
        outer_loop_size = self.cp_size // self.intra_group_size

        inner_offset = self.rank_in_cp // self.intra_group_size * self.intra_group_size
        inner_loop_size = self.intra_group_size // self.ulysses_size
        inner_loop_next_rank = (self.rank_in_cp - self.ulysses_size) % self.intra_group_size + inner_offset
        inner_loop_prev_rank = (self.rank_in_cp + self.ulysses_size) % self.intra_group_size + inner_offset

        skip_group_range = 0

        for group_id, should_compute in enumerate(plan_list):
            if should_compute: 
                if group_id != 0:
                    # 非本地group，需要先获取远端头kv
                    outer_loop_next_rank = (self.rank_in_cp - self.intra_group_size * (skip_group_range + 1)) % self.cp_size
                    outer_loop_prev_rank = (self.rank_in_cp + self.intra_group_size * (skip_group_range + 1)) % self.cp_size
                    comm_list.append((outer_loop_prev_rank, outer_loop_next_rank))
                # 组内ring通信
                for _ in range(inner_loop_size - 1):
                    comm_list.append((inner_loop_prev_rank, inner_loop_next_rank))
            else:
                if group_id != 0:
                    skip_group_range += 1

        # ogger.info(f"Generated selective comm pattern for group plan {plan_list}: {comm_list}")
        return outer_loop_size, inner_loop_size, comm_list


       


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

        warmup_cooldown = strategy.is_warmup_or_cooldown_step()
        is_anchor = strategy.is_anchor_step() 

        if get_cp_group().rank_in_group == 0 and self.layer_id == 0:
            Attn_name = "warmup/cooldown" if warmup_cooldown else ("anchor" if is_anchor else "selective")
            print("" + "=" * 20 + f" T{get_timestep()}-{Attn_name} " + "=" * 20, flush=True)

        if warmup_cooldown or is_anchor:
            final_state, group_states, local_state = self._anchor_step_attn(q, k, v, is_varlen, is_anchor, **attn_kwargs)
        else:
            final_state, group_states, local_state = self._selective_group_attn(
                q, k, v, is_varlen, **attn_kwargs
            )
        if self.ulysses_size > 1:
            out = ulysses_group.all_to_all(final_state.out, seq_dim, head_dim)
        else:
            out = final_state.out

        if is_anchor: # 所有层full compute
            # Per-rank logical local group is always 0 in this rotated loop.
            strategy.update_anchor_stats(self.layer_id, group_states, local_state)

        strategy.run_ditango_planner(self.layer_id)

        return out, None, None

    @Timer.get_timer("anchor_attn")
    def _anchor_step_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_varlen: bool,
        is_anchor: bool,
        **kwargs,
    ) -> Tuple[AttentionState, Dict[int, AttentionState], AttentionState]:
        """Anchor step: compute all group attention states and cache them."""
        strategy: DiTangoV3Strategy = DiffusionBackend.flexcache.strategy
        outer_loop_size = self.cp_size // self.intra_group_size
        inner_loop_size = self.intra_group_size // self.ulysses_size
        outer_loop_prev_rank = (self.rank_in_cp + self.intra_group_size) % self.cp_size
        outer_loop_next_rank = (self.rank_in_cp - self.intra_group_size) % self.cp_size
        inner_offset = self.rank_in_cp // self.intra_group_size * self.intra_group_size
        inner_loop_prev_rank = (self.rank_in_cp + self.ulysses_size) % self.intra_group_size + inner_offset
        inner_loop_next_rank = (self.rank_in_cp - self.ulysses_size) % self.intra_group_size + inner_offset
                                
        final_state = AttentionState()
        computed_group_states: Dict[int, AttentionState] = {}
        local_partition_state = AttentionState()

        current_group_id = 0
        for outer_step in range(outer_loop_size):
            group_state = AttentionState()
            decision_code = (
                strategy.DECISION_CODE_ANCHOR
                if is_anchor
                else strategy.DECISION_CODE_WARMUP_COOLDOWN
            )

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

                is_local_partition = self._is_local_partition_step(outer_step, inner_step)
                if is_local_partition:
                    local_partition_state.update(block_out, block_lse)
                    final_state = AttentionState.merge(final_state, local_partition_state)
                    if is_anchor:
                        strategy.set_curr_local_state_compress(self.layer_id, local_partition_state)
                else:
                    group_state.update(block_out, block_lse)

                if (inner_step + 1 != inner_loop_size) or (outer_step + 1 != outer_loop_size):
                    if is_varlen:
                        k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                    else:
                        k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)

            if not group_state.is_empty():
                if is_anchor:
                    computed_group_states[current_group_id] = group_state
                    strategy.store(self.layer_id, group_id=current_group_id, group_state=group_state)
                    strategy._update_group_local_ref_compress(self.layer_id, current_group_id)
                final_state = AttentionState.merge(final_state, group_state)

            strategy.record_group_decision(self.layer_id, current_group_id, decision_code)

            current_group_id = (current_group_id + 1) % outer_loop_size

        return final_state, computed_group_states, local_partition_state

    def _selective_group_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_varlen: bool,
        **kwargs,
    ) -> Tuple[AttentionState, Dict[int, AttentionState], AttentionState]:
        strategy: DiTangoV3Strategy = DiffusionBackend.flexcache.strategy
        final_state = AttentionState()
        computed_group_states: Dict[int, AttentionState] = {}
        local_partition_state = AttentionState()
        num_groups = self.cp_size // self.intra_group_size
        fallback_count = 0
        current_group_id = 0
        branch_key = strategy._branch_key()
        branch_plan = strategy.ditango_plan.get(branch_key, None)
        if branch_plan is None:
            group_plan = torch.ones((num_groups,), dtype=torch.bool, device=self.group.device)
        else:
            group_plan = branch_plan[self.layer_id]
        if strategy._should_log_summary() and self.layer_id == 0:
            logger.info(f"[Selective Attn t{get_timestep()}l{self.layer_id}b{branch_key}] {group_plan=}")
        # outer step: group-level rotation - selective compute or reuse ; inner step: intra-group rotation

        compute_group = group_plan.bool().tolist().count(True)
        with Timer.get_timer(f"sele_attn_{compute_group}"):
            outer_loop_size, group_size, comm_list = self._generate_selective_comm_pattern(group_plan)

            for outer_step in range(outer_loop_size):
                should_compute = bool(group_plan[current_group_id].item())
                group_state = AttentionState()
                inner_loop_size = group_size if should_compute else 1

                for inner_step in range(inner_loop_size):
                    if is_varlen:
                        cu_seqlens_k = kwargs.get("cu_seqlens_k", None)
                        data_pack = (k, v, cu_seqlens_k)
                    else:
                        data_pack = (k, v)

                    if comm_list:
                        next_rank, prev_rank = comm_list.pop(0)
                        should_comm = True
                    else:
                        next_rank, prev_rank = None, None
                        should_comm = False
                    if should_comm:
                        nxt_data_pack = async_ring_p2p_commit(
                            self.group,
                            data_pack,
                            src_rank=prev_rank,
                            dst_rank=next_rank,
                        )

                    is_local_partition = self._is_local_partition_step(outer_step, inner_step)
                    # Local partition must always be computed and merged into final_state only.
                    if is_local_partition:
                        block_out, block_lse, _ = self.attn_backend(q, k, v, **kwargs)
                        local_partition_state.update(block_out, block_lse)
                        strategy.set_curr_local_state_compress(self.layer_id, local_partition_state)
                        final_state = AttentionState.merge(final_state, local_partition_state)
                    elif should_compute:
                        block_out, block_lse, _ = self.attn_backend(q, k, v, **kwargs)
                        group_state.update(block_out, block_lse)

                    if not should_compute:
                        reused_state = strategy.reuse(self.layer_id, group_id=current_group_id)
                        assert not reused_state.is_empty(), f"Group {current_group_id} reuse miss at t{get_timestep()}l{self.layer_id}"
                        # logger.info(f"Reuse group {current_group_id} at t{get_timestep()}l{self.layer_id} | {strategy._state_digest(reused_state)}")

                    if should_comm:
                        if is_varlen:
                            k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                        else:
                            k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)

                if should_compute:
                    if not group_state.is_empty():
                        strategy.store(self.layer_id, group_id=current_group_id, group_state=group_state)
                        strategy._update_group_local_ref_compress(self.layer_id, current_group_id)
                        computed_group_states[current_group_id] = group_state
                        final_state = AttentionState.merge(final_state, group_state)
                    strategy.record_group_decision(
                        self.layer_id,
                        current_group_id,
                        strategy.DECISION_CODE_COMPUTE,
                    )
                else:
                    final_state = AttentionState.merge(final_state, reused_state)
                    strategy.record_group_decision(
                        self.layer_id,
                        current_group_id,
                        strategy.DECISION_CODE_REUSE,
                    )

                current_group_id = (current_group_id + 1) % num_groups

        strategy.record_step_fallback(self.layer_id, fallback_count)
        return final_state, computed_group_states, local_partition_state