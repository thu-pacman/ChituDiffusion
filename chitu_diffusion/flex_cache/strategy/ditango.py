import torch
import os
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Optional, Tuple

from chitu_core.distributed.parallel_state import get_cp_group, get_up_group
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

    def __init__(
        self,
        task: Optional[DiffusionTask] = None,
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

        self.type = "ditango"
        self.tradeoff_score = float(ase_threshold)
        self.task = task

        self.ase_threshold = float(ase_threshold)
        self.anchor_interval = max(1, int(anchor_interval))
        self.tau_max = max(1, int(tau_max))
        self.intra_group_size_limit = max(1, int(intra_group_size_limit))
        self.enable_dynamic_compose = bool(enable_dynamic_compose)
        self.warmup_steps = max(1, int(warmup_steps))
        self.cooldown_steps = max(1, int(cooldown_steps))
        self.debug_cache_io = self.DEFAULT_DEBUG_CACHE_IO
        self.total_layers = None

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

    def dump_compute_reuse_visualization(self):
        if not self._is_main_rank():
            return

        output_dir = self._get_output_dir()
        self._save_compute_ratio_heatmap(output_dir, "pos")
        logger.info(f"[DiTangoV3] Saved compute/reuse visualization PPM to {output_dir}/ditangov3_compute_ratio_pos.ppm")

    def _should_log_cache_io(self, layer_id: int) -> bool:
        if not self.debug_cache_io:
            return False
        if layer_id != 0:
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

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

    def _is_anchor_step(self, step: int) -> bool:
        if step in self.promoted_anchor_steps:
            return True
        if step == 0:
            return True

        branch = self._branch_key()
        last_anchor_step = self.last_anchor_step.get(branch, 0)
        return (step - last_anchor_step) >= self.anchor_interval

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

    def _mark_anchor_step(self, step: int):
        self.last_anchor_step[self._branch_key()] = step

    def _promote_anchor_if_needed(self, step: int, plan: Dict[int, bool], ase_map: Dict[int, float]) -> bool:
        """
        Promote current step to anchor when all groups are selected to compute
        due to ASE exceeding threshold.
        """
        if self._is_warmup_or_cooldown_step(step):
            return False
        if not plan:
            return False
        if not all(plan.values()):
            return False

        # Ignore local/forced-unknown entries marked as NaN.
        effective_ase = [ase for ase in ase_map.values() if ase == ase]
        if not effective_ase:
            return False

        if all(ase > self.ase_threshold for ase in effective_ase):
            self.promoted_anchor_steps.add(step)
            return True
        return False

    def _local_as_key(self, layer_id: int) -> str:
        return f"{self._branch_key()}_{layer_id}_local"

    def _compress_local_as(self, state: AttentionState) -> torch.Tensor:
        """Compress local AS to a tiny CPU tensor for drift estimation."""
        out_fp32 = state.out.float()
        lse_fp32 = state.lse.float()
        comp = torch.tensor(
            [
                out_fp32.mean().item(),
                out_fp32.std().item(),
                lse_fp32.mean().item(),
                lse_fp32.std().item(),
            ],
            dtype=torch.float32,
            device="cpu",
        )
        return comp

    def update_local_as(self, layer_id: int, local_state: AttentionState, as_anchor: bool):
        if local_state is None or local_state.is_empty():
            return
        key = self._local_as_key(layer_id)
        comp = self._compress_local_as(local_state)
        self.current_local_as[key] = comp
        if as_anchor:
            self.anchor_local_as[key] = comp

    def _get_local_drift(self, layer_id: int) -> float:
        key = self._local_as_key(layer_id)
        curr = self.current_local_as.get(key, None)
        anchor = self.anchor_local_as.get(key, None)
        if curr is None or anchor is None:
            return float("inf")
        return float(torch.norm(curr - anchor).item())

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
        alpha = float(meta.get("alpha", 1.0))
        drift = self._get_local_drift(layer_id)
        ase = w * alpha * drift
        self.group_meta[key]["ase"] = ase
        return ase

    def plan_group_update(
        self,
        layer_id: int,
        num_groups: int,
        local_group_id: int,
    ) -> Tuple[Dict[int, bool], Dict[int, float]]:
        step = get_timestep()
        is_anchor = self._is_anchor_step(step)
        plan: Dict[int, bool] = {}
        ase_map: Dict[int, float] = {}

        for group_id in range(num_groups):
            key = self._group_key(layer_id, group_id)
            if is_anchor or group_id == local_group_id:
                plan[group_id] = True
                ase_map[group_id] = float("nan")
                continue

            if key not in DiffusionBackend.flexcache.cache:
                plan[group_id] = True
                ase_map[group_id] = float("inf")
                continue

            ase = self._estimate_group_ase(layer_id, group_id)
            ase_map[group_id] = ase
            plan[group_id] = ase >= self.ase_threshold

        return plan, ase_map

    def update_anchor_stats(
        self,
        layer_id: int,
        group_states: Dict[int, AttentionState],
        local_group_id: int,
    ):
        if not group_states:
            return

        ordered_group_ids = sorted(group_states.keys())
        lse_scores = []
        for group_id in ordered_group_ids:
            state = group_states[group_id]
            lse_scores.append(float(torch.mean(state.lse.float()).item()))

        score_tensor = torch.tensor(lse_scores, dtype=torch.float32)
        weight_tensor = torch.softmax(score_tensor - score_tensor.max(), dim=0)

        local_state = group_states.get(local_group_id)
        local_norm = 1.0
        if local_state is not None and not local_state.is_empty():
            local_norm = max(1e-6, torch.norm(local_state.out.float()).item())
            # Anchor step stores compressed local AS reference for later drift estimation.
            self.update_local_as(layer_id, local_state, as_anchor=True)

        step = get_timestep()
        for idx, group_id in enumerate(ordered_group_ids):
            state = group_states[group_id]
            key = self._group_key(layer_id, group_id)
            alpha = torch.norm(state.out.float()).item() / local_norm
            self.group_meta[key] = {
                "weight": float(weight_tensor[idx].item()),
                "alpha": float(alpha),
                "last_compute_step": step,
                "ase": 0.0,
            }

    def get_reuse_key(self, layer_id: int, group_id: int = 0, **kwargs):
        return self._group_key(layer_id, group_id)

    def reuse(self, layer_id: int, group_id: int = 0, **kwargs) -> AttentionState:
        key = self.get_reuse_key(layer_id, group_id=group_id)
        step = get_timestep()
        if key in DiffusionBackend.flexcache.cache:
            state = DiffusionBackend.flexcache.cache[key]
            src_step = self.group_meta.get(key, {}).get("last_compute_step", None)
            age_text = "na" if src_step is None else str(step - int(src_step))
            if self._should_log_cache_io(layer_id):
                logger.info(
                    f"[DiTangoV3][Cache REUSE HIT] step={step} key={key} "
                    f"src_step={src_step} age={age_text} {self._state_digest(state)}"
                )
            return state

        if self._should_log_cache_io(layer_id):
            logger.info(f"[DiTangoV3][Cache REUSE MISS] step={step} key={key}")
        return AttentionState()

    def get_store_key(self, layer_id: int, group_id: int = 0, **kwargs):
        return self._group_key(layer_id, group_id)

    def store(self, layer_id: int, group_id: int, group_state: Optional[AttentionState], **kwargs):
        step = get_timestep()
        if self._is_warmup_or_cooldown_step(step):
            if self._should_log_cache_io(layer_id):
                key = self.get_store_key(layer_id, group_id=group_id)
                logger.info(f"[DiTangoV3][Cache STORE SKIP] step={step} key={key} reason=warmup/cooldown")
            return

        if group_state is None or group_state.is_empty():
            if self._should_log_cache_io(layer_id):
                key = self.get_store_key(layer_id, group_id=group_id)
                logger.info(f"[DiTangoV3][Cache STORE SKIP] step={get_timestep()} key={key} state=empty")
            return

        key = self.get_store_key(layer_id, group_id=group_id)
        DiffusionBackend.flexcache.cache[key] = group_state
        self.group_meta.setdefault(key, {})["last_compute_step"] = step
        if self._should_log_cache_io(layer_id):
            logger.info(
                f"[DiTangoV3][Cache STORE] step={step} key={key} "
                f"{self._state_digest(group_state)}"
            )

    def reset_state(self):
        self.anchor_local_as: Dict[str, torch.Tensor] = {}
        self.current_local_as: Dict[str, torch.Tensor] = {}
        self.group_meta: Dict[str, Dict[str, float]] = {}
        self.promoted_anchor_steps = set()
        self.last_anchor_step: Dict[str, int] = {}
        self.compute_ratio_records: Dict[str, Dict[Tuple[int, int], float]] = {"pos": {}, "neg": {}}
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

    def _format_ase_for_log(self, ase_map: Dict[int, float], group_order: Dict[int, bool]) -> str:
        items = []
        for group_id in group_order.keys():
            ase = ase_map.get(group_id, float("nan"))
            if ase != ase:  # NaN
                ase_str = "na"
            elif ase == float("inf"):
                ase_str = "inf"
            else:
                ase_str = f"{ase:.4f}"
            items.append(f"g{group_id}:{ase_str}")
        return "{" + ", ".join(items) + "}"

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

    @Timer.get_timer("ditango_attn_v3")
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
        is_anchor = strategy._is_anchor_step(curr_step)

        if force_full_compute or is_anchor:
            final_state, group_states, plan, ase_map = self._anchor_step_attn(q, k, v, is_varlen, **attn_kwargs)
            fallback_count = 0
            promoted_anchor = False
        else:
            final_state, group_states, plan, ase_map, fallback_count = self._selective_group_attn(
                q, k, v, is_varlen, **attn_kwargs
            )
            promoted_anchor = strategy._promote_anchor_if_needed(curr_step, plan, ase_map)

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
            is_anchor=(not force_full_compute) and (is_anchor or promoted_anchor),
        )
        if (not force_full_compute) and (is_anchor or promoted_anchor):
            strategy._mark_anchor_step(curr_step)
            strategy.update_anchor_stats(self.layer_id, group_states, local_group_id)

        # if self.layer_id % 5 == 0:
        #     compute_groups = sum(1 for _, should_compute in plan.items() if should_compute)
        #     metric = torch.tensor(float(compute_groups), device=out.device)
        #     MagLogger.log_magnitude(metric, get_timestep(), self.layer_id, "v3_compute_groups")

        ase_text = self._format_ase_for_log(ase_map, plan)
        if strategy._should_log_cache_io(self.layer_id):
            logger.info(
                f"[DiTangoV3][Cache STATS] step={curr_step} cache_size={len(DiffusionBackend.flexcache.cache)} "
                f"group_meta={len(strategy.group_meta)}"
            )
        self._print_layer0(
            f"T{curr_step}L{self.layer_id} | plan={plan} ase={ase_text} "
            f"fallback={fallback_count} eps={strategy.ase_threshold:.4f} "
            f"anchor={is_anchor or promoted_anchor} promoted_anchor={promoted_anchor} "
            f"force_full_compute={force_full_compute}"
        )
        return out, None, None

    @Timer.get_timer("ditango_attn_v3_group_select")
    def _anchor_step_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_varlen: bool,
        **kwargs,
    ) -> Tuple[AttentionState, Dict[int, AttentionState], Dict[int, bool], Dict[int, float]]:
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
                group_state.update(block_out, block_lse)

                if (inner_step + 1 != inner_loop_size) or (outer_step + 1 != outer_loop_size):
                    if is_varlen:
                        k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                    else:
                        k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)

            strategy.store(self.layer_id, group_id=current_group_id, group_state=group_state)
            computed_group_states[current_group_id] = group_state
            final_state = AttentionState.merge(final_state, group_state)

            if current_group_id == local_group_id:
                strategy.update_local_as(self.layer_id, group_state, as_anchor=True)

            current_group_id = (current_group_id + 1) % num_groups

        return final_state, computed_group_states, plan, ase_map

    @Timer.get_timer("ditango_attn_v3_group_select")
    def _selective_group_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_varlen: bool,
        **kwargs,
    ) -> Tuple[AttentionState, Dict[int, AttentionState], Dict[int, bool], Dict[int, float], int]:
        strategy: DiTangoV3Strategy = DiffusionBackend.flexcache.strategy

        num_groups = self.cp_size // self.intra_group_size
        local_group_id = self.rank_in_cp // self.intra_group_size
        group_plan, ase_map = strategy.plan_group_update(self.layer_id, num_groups, local_group_id)

        outer_loop_size = num_groups
        outer_loop_next_rank = (self.rank_in_cp - self.intra_group_size) % self.cp_size
        outer_loop_prev_rank = (self.rank_in_cp + self.intra_group_size) % self.cp_size

        inner_offset = self.rank_in_cp // self.intra_group_size * self.intra_group_size
        inner_loop_size = self.intra_group_size // self.ulysses_size
        inner_loop_next_rank = (self.rank_in_cp - self.ulysses_size) % self.intra_group_size + inner_offset
        inner_loop_prev_rank = (self.rank_in_cp + self.ulysses_size) % self.intra_group_size + inner_offset

        final_state = AttentionState()
        computed_group_states: Dict[int, AttentionState] = {}
        fallback_count = 0

        current_group_id = local_group_id
        for outer_step in range(outer_loop_size):
            should_compute = group_plan[current_group_id]
            reused_state = AttentionState()
            if not should_compute:
                reused_state = strategy.reuse(self.layer_id, group_id=current_group_id)
                if reused_state.is_empty():
                    # Missing cache must fallback to compute to preserve correctness.
                    should_compute = True
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

                if should_compute:
                    block_out, block_lse, _ = self.attn_backend(q, k, v, **kwargs)
                    group_state.update(block_out, block_lse)

                if (inner_step + 1 != inner_loop_size) or (outer_step + 1 != outer_loop_size):
                    if is_varlen:
                        k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                    else:
                        k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)

            if should_compute:
                strategy.store(self.layer_id, group_id=current_group_id, group_state=group_state)
                computed_group_states[current_group_id] = group_state
                final_state = AttentionState.merge(final_state, group_state)
                if current_group_id == local_group_id:
                    strategy.update_local_as(self.layer_id, group_state, as_anchor=False)
            else:
                final_state = AttentionState.merge(final_state, reused_state)

            current_group_id = (current_group_id + 1) % num_groups

        return final_state, computed_group_states, group_plan, ase_map, fallback_count