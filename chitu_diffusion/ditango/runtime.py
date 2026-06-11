from logging import getLogger
from typing import Dict, Optional, Tuple

import torch

from chitu_diffusion.core.distributed.parallel_state import get_cp_group, get_up_group
from chitu_diffusion.ditango.planner import DiTangoPlanner, get_timestep
from chitu_diffusion.ditango.state import AttentionState
from chitu_diffusion.observability.timer import Timer
from chitu_diffusion.runtime.backend import DiffusionBackend
from chitu_diffusion.runtime.parallel_utils import async_ring_p2p_commit, async_ring_p2p_wait_and_update

logger = getLogger(__name__)


def get_ditango_planner() -> DiTangoPlanner:
    planner = DiffusionBackend.ditango
    if planner is None:
        raise RuntimeError("DiTango attention is active but DiffusionBackend.ditango is not configured.")
    return planner


class DitangoAttention:
    COMPRESSED_NONLOCAL_TAG = "nonlocal"

    def __init__(self, layer_id: int):
        self.attn_backend = DiffusionBackend.attn
        self.group = get_cp_group()
        self.cp_size = self.group.group_size
        self.global_rank = torch.distributed.get_rank()
        self.rank_in_cp = self.group.rank_in_group
        self.layer_id = layer_id

        strategy = get_ditango_planner()
        self.intra_group_size = min(self.cp_size, strategy.intra_group_size_limit)
        self.ulysses_size = min(DiffusionBackend.args.infer.diffusion.up_limit, self.cp_size, self.intra_group_size)
        assert self.cp_size <= self.intra_group_size or self.cp_size % self.intra_group_size == 0
        if self.global_rank == 0:
            logger.info("L%d | Using Ditango Attn (curvature interval).", layer_id)

    @staticmethod
    def _is_local_partition_step(outer_step: int, inner_step: int) -> bool:
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
                    outer_loop_next_rank = (self.rank_in_cp - self.intra_group_size * (skip_group_range + 1)) % self.cp_size
                    outer_loop_prev_rank = (self.rank_in_cp + self.intra_group_size * (skip_group_range + 1)) % self.cp_size
                    comm_list.append((outer_loop_prev_rank, outer_loop_next_rank))
                    skip_group_range = 0
                for _ in range(inner_loop_size - 1):
                    comm_list.append((inner_loop_prev_rank, inner_loop_next_rank))
            elif group_id != 0:
                skip_group_range += 1
        return outer_loop_size, inner_loop_size, comm_list

    def _is_local_only_state_reuse_plan(
        self,
        strategy: DiTangoPlanner,
        group_plan: torch.Tensor,
        num_groups: int,
        is_varlen: bool,
    ) -> bool:
        if num_groups <= 1 or strategy.groupwise_reuse_stale_kv:
            return False
        if strategy.groupwise_local_expand != 0:
            return False
        plan = group_plan.bool().tolist()
        return bool(plan[0]) and not any(bool(item) for item in plan[1:])

    def _stores_anchor_as_compressed_nonlocal(self, strategy: DiTangoPlanner) -> bool:
        if self.cp_size <= self.intra_group_size or strategy.groupwise_reuse_stale_kv:
            return False
        if strategy.groupwise_local_expand != 0:
            return False
        if strategy.groupwise_stagger_period > 0 and strategy.groupwise_stagger_fresh_count > 0:
            return False
        if strategy.groupwise_topk_mode not in {"", "none", "off"} and strategy.groupwise_extra_topk > 0:
            return False
        return True

    def _compress_nonlocal_cached_states(
        self,
        strategy: DiTangoPlanner,
        num_groups: int,
    ) -> AttentionState:
        compressed = strategy.reuse_compressed(self.layer_id, self.COMPRESSED_NONLOCAL_TAG)
        if not compressed.is_empty():
            return compressed

        compressed = AttentionState()
        for group_id in range(1, num_groups):
            state = strategy.reuse(self.layer_id, group_id=group_id)
            if state.is_empty():
                strategy.discard_compressed_state(self.layer_id, self.COMPRESSED_NONLOCAL_TAG)
                return AttentionState()
            compressed = AttentionState.merge(compressed, state)

        strategy.store_compressed(self.layer_id, self.COMPRESSED_NONLOCAL_TAG, compressed)
        for group_id in range(1, num_groups):
            strategy.discard_group_state(self.layer_id, group_id)
        return compressed

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
        strategy = get_ditango_planner()
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
        else:
            ulysses_group = None

        warmup_cooldown = strategy.is_warmup_or_cooldown_step()
        is_anchor = strategy.is_anchor_step()
        full_compute = strategy.is_full_compute_mode()
        if get_cp_group().rank_in_group == 0 and self.layer_id == 0:
            attn_name = "anchor" if is_anchor else ("full-compute" if full_compute else ("warmup/cooldown" if warmup_cooldown else "selective"))
            print("" + "=" * 20 + f" T{get_timestep()}-{attn_name} " + "=" * 20, flush=True)

        if full_compute or warmup_cooldown or is_anchor:
            final_state, group_states, local_state = self._anchor_step_attn(q, k, v, is_varlen, is_anchor, **attn_kwargs)
            if is_anchor:
                strategy.update_anchor_stats(self.layer_id, group_states, local_state, final_state.out)
        else:
            final_state, _, _ = self._selective_group_attn(q, k, v, is_varlen, **attn_kwargs)

        if ulysses_group is not None:
            out = ulysses_group.all_to_all(final_state.out, seq_dim, head_dim)
        else:
            out = final_state.out

        strategy.run_ditango_planner(self.layer_id)
        return out, None, None

    @Timer.get_timer("ditango_anchor_attn")
    def _anchor_step_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_varlen: bool,
        is_anchor: bool,
        **kwargs,
    ) -> Tuple[AttentionState, Dict[int, AttentionState], AttentionState]:
        strategy = get_ditango_planner()
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
        compressed_nonlocal_state = AttentionState()
        current_group_id = 0
        compress_anchor_nonlocal = is_anchor and self._stores_anchor_as_compressed_nonlocal(strategy)

        if is_anchor:
            strategy.discard_compressed_state(self.layer_id, self.COMPRESSED_NONLOCAL_TAG)

        for outer_step in range(outer_loop_size):
            group_state = AttentionState()
            group_k_to_store = None
            group_v_to_store = None
            group_cu_to_store = None
            decision_code = strategy.DECISION_CODE_ANCHOR if is_anchor else strategy.DECISION_CODE_WARMUP_COOLDOWN
            for inner_step in range(inner_loop_size):
                if is_varlen:
                    cu_seqlens_k = kwargs.get("cu_seqlens_k", None)
                    data_pack = (k, v, cu_seqlens_k)
                else:
                    data_pack = (k, v)

                if inner_step + 1 != inner_loop_size:
                    nxt_data_pack = async_ring_p2p_commit(
                        self.group, data_pack, src_rank=inner_loop_prev_rank, dst_rank=inner_loop_next_rank
                    )
                elif outer_step + 1 != outer_loop_size:
                    nxt_data_pack = async_ring_p2p_commit(
                        self.group, data_pack, src_rank=outer_loop_prev_rank, dst_rank=outer_loop_next_rank
                    )

                block_out, block_lse, _ = self.attn_backend(q, k, v, **kwargs)
                final_state.update(block_out, block_lse)
                if inner_step == 0:
                    group_k_to_store = k
                    group_v_to_store = v
                    group_cu_to_store = kwargs.get("cu_seqlens_k", None)
                is_local_partition = self._is_local_partition_step(outer_step, inner_step)
                if is_local_partition:
                    local_partition_state.update(block_out, block_lse)
                else:
                    group_state.update(block_out, block_lse)

                if (inner_step + 1 != inner_loop_size) or (outer_step + 1 != outer_loop_size):
                    if is_varlen:
                        k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                    else:
                        k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)

            state_to_store = local_partition_state if current_group_id == 0 else group_state
            if current_group_id == 0 and not group_state.is_empty():
                state_to_store = AttentionState.merge(state_to_store, group_state)
            if not state_to_store.is_empty():
                computed_group_states[current_group_id] = state_to_store
                if is_anchor:
                    if compress_anchor_nonlocal and current_group_id > 0:
                        compressed_nonlocal_state = AttentionState.merge(compressed_nonlocal_state, state_to_store)
                        strategy.discard_group_state(self.layer_id, current_group_id)
                    else:
                        strategy.store(self.layer_id, group_id=current_group_id, group_state=state_to_store)
                    if strategy.groupwise_reuse_stale_kv:
                        strategy.store_kv(
                            self.layer_id,
                            group_id=current_group_id,
                            k=group_k_to_store if group_k_to_store is not None else k,
                            v=group_v_to_store if group_v_to_store is not None else v,
                            cu_seqlens_k=group_cu_to_store,
                        )
            strategy.record_group_decision(self.layer_id, current_group_id, decision_code)
            current_group_id = (current_group_id + 1) % outer_loop_size

        if compress_anchor_nonlocal and not compressed_nonlocal_state.is_empty():
            strategy.store_compressed(self.layer_id, self.COMPRESSED_NONLOCAL_TAG, compressed_nonlocal_state)

        return final_state, computed_group_states, local_partition_state

    def _selective_group_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_varlen: bool,
        **kwargs,
    ) -> Tuple[AttentionState, Dict[int, AttentionState], AttentionState]:
        strategy = get_ditango_planner()
        num_groups = self.cp_size // self.intra_group_size
        branch_key = strategy._branch_key()
        branch_plan = strategy.ditango_plan.get(branch_key, None)
        if branch_plan is None:
            group_plan = torch.ones((num_groups,), dtype=torch.bool, device=self.group.device)
        else:
            group_plan = branch_plan[self.layer_id].to(self.group.device)

        group_plan = group_plan.clone().bool()
        old_compute_states: Dict[int, AttentionState] = {}
        use_compressed_nonlocal = self._is_local_only_state_reuse_plan(strategy, group_plan, num_groups, is_varlen)
        compressed_nonlocal_state = AttentionState()
        if use_compressed_nonlocal:
            compressed_nonlocal_state = self._compress_nonlocal_cached_states(strategy, num_groups)
            if compressed_nonlocal_state.is_empty():
                use_compressed_nonlocal = False
        for group_id in range(num_groups):
            if bool(group_plan[group_id].item()):
                old_state = strategy.reuse(self.layer_id, group_id=group_id)
                if not old_state.is_empty():
                    old_compute_states[group_id] = old_state
                strategy.discard_group_state(self.layer_id, group_id)
                if not use_compressed_nonlocal:
                    strategy.discard_compressed_state(self.layer_id, self.COMPRESSED_NONLOCAL_TAG)

        if not bool(group_plan.bool().any().item()):
            final_state = self._merge_cached_group_states(strategy, num_groups, q=q, is_varlen=is_varlen, **kwargs)
            if not final_state.is_empty():
                strategy.record_step_fallback(self.layer_id, 0)
                for group_id in range(num_groups):
                    strategy.record_group_decision(self.layer_id, group_id, strategy.DECISION_CODE_REUSE)
                return final_state, {}, AttentionState()
            group_plan = torch.ones((num_groups,), dtype=torch.bool, device=self.group.device)

        fallback_count = 0
        for group_id in range(num_groups):
            if bool(group_plan[group_id].item()):
                continue
            if use_compressed_nonlocal and group_id > 0:
                continue
            has_cached_state = not strategy.reuse(self.layer_id, group_id=group_id).is_empty()
            has_cached_kv = strategy.reuse_kv(self.layer_id, group_id=group_id) is not None
            if strategy.groupwise_reuse_stale_kv and not self._stale_kv_reuse_supported(is_varlen):
                has_cached_kv = False
            if (strategy.groupwise_reuse_stale_kv and not has_cached_kv) or (
                not strategy.groupwise_reuse_stale_kv and not has_cached_state
            ):
                group_plan[group_id] = True
                fallback_count += 1

        final_state = AttentionState()
        computed_group_states: Dict[int, AttentionState] = {}
        reused_group_states: Dict[int, AttentionState] = {}
        local_partition_state = AttentionState()
        align_source_pairs = []
        align_stats = None
        current_group_id = 0
        compute_group = group_plan.bool().tolist().count(True)
        use_state_align = (
            strategy.groupwise_state_align
            and not strategy.groupwise_reuse_stale_kv
            and compute_group > 0
            and compute_group < num_groups
        )
        if use_state_align:
            align_source_pairs = list(old_compute_states.items())

        with Timer.get_timer(f"ditango_sele_attn_{compute_group}"):
            outer_loop_size, group_size, comm_list = self._generate_selective_comm_pattern(group_plan)
            for outer_step in range(outer_loop_size):
                should_compute = bool(group_plan[current_group_id].item())
                group_state = AttentionState()
                group_k_to_store = None
                group_v_to_store = None
                group_cu_to_store = None
                inner_loop_size = group_size if should_compute else 1

                for inner_step in range(inner_loop_size):
                    if is_varlen:
                        cu_seqlens_k = kwargs.get("cu_seqlens_k", None)
                        data_pack = (k, v, cu_seqlens_k)
                    else:
                        data_pack = (k, v)

                    if should_compute and comm_list:
                        next_rank, prev_rank = comm_list.pop(0)
                        nxt_data_pack = async_ring_p2p_commit(
                            self.group, data_pack, src_rank=prev_rank, dst_rank=next_rank
                        )
                    else:
                        nxt_data_pack = None

                    is_local_partition = self._is_local_partition_step(outer_step, inner_step)
                    if should_compute:
                        block_out, block_lse, _ = self.attn_backend(q, k, v, **kwargs)
                        if inner_step == 0:
                            group_k_to_store = k
                            group_v_to_store = v
                            group_cu_to_store = kwargs.get("cu_seqlens_k", None)
                        if is_local_partition:
                            local_partition_state.update(block_out, block_lse)
                        else:
                            group_state.update(block_out, block_lse)

                    if nxt_data_pack is not None:
                        if is_varlen:
                            k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                        else:
                            k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)

                if should_compute:
                    state_to_store = local_partition_state if current_group_id == 0 else group_state
                    if current_group_id == 0 and not group_state.is_empty():
                        state_to_store = AttentionState.merge(state_to_store, group_state)
                    if not state_to_store.is_empty():
                        strategy.store(self.layer_id, group_id=current_group_id, group_state=state_to_store)
                        if strategy.groupwise_reuse_stale_kv:
                            strategy.store_kv(
                                self.layer_id,
                                group_id=current_group_id,
                                k=group_k_to_store if group_k_to_store is not None else k,
                                v=group_v_to_store if group_v_to_store is not None else v,
                                cu_seqlens_k=group_cu_to_store,
                        )
                        computed_group_states[current_group_id] = state_to_store
                        if current_group_id == 0 and not use_state_align:
                            final_state = state_to_store
                        elif not use_state_align:
                            final_state = AttentionState.merge(final_state, state_to_store)
                        if use_state_align:
                            align_stats = self._update_state_align_stats(
                                current_group_id,
                                state_to_store,
                                align_source_pairs,
                                align_stats,
                            )
                    strategy.record_group_decision(self.layer_id, current_group_id, strategy.DECISION_CODE_COMPUTE)
                else:
                    if use_compressed_nonlocal and current_group_id > 0:
                        strategy.record_group_decision(self.layer_id, current_group_id, strategy.DECISION_CODE_REUSE)
                        current_group_id = (current_group_id + 1) % num_groups
                        continue
                    reused_state = self._reuse_group_state(strategy, current_group_id, q, is_varlen, **kwargs)
                    reused_group_states[current_group_id] = reused_state
                    if not use_state_align:
                        final_state = AttentionState.merge(final_state, reused_state)
                    strategy.record_group_decision(self.layer_id, current_group_id, strategy.DECISION_CODE_REUSE)

                current_group_id = (current_group_id + 1) % num_groups

        if use_compressed_nonlocal:
            final_state = self._merge_selective_states_with_alignment(
                num_groups,
                computed_group_states,
                reused_group_states,
                align_stats,
                compressed_nonlocal_state,
            )
        elif use_state_align:
            final_state = self._merge_selective_states_with_alignment(
                num_groups,
                computed_group_states,
                reused_group_states,
                align_stats,
            )
        strategy.record_step_fallback(self.layer_id, fallback_count)
        return final_state, computed_group_states, local_partition_state

    def _merge_selective_states_with_alignment(
        self,
        num_groups: int,
        computed_group_states: Dict[int, AttentionState],
        reused_group_states: Dict[int, AttentionState],
        align_stats,
        compressed_nonlocal_state: Optional[AttentionState] = None,
    ) -> AttentionState:
        if compressed_nonlocal_state is not None and not compressed_nonlocal_state.is_empty():
            final_state = computed_group_states.get(0, AttentionState())
            stale_state = compressed_nonlocal_state
            if align_stats is not None:
                stale_state = self._apply_state_align_correction(
                    get_ditango_planner(),
                    stale_state,
                    align_stats,
                    group_id=1,
                    num_groups=num_groups,
                )
            return AttentionState.merge(final_state, stale_state)

        final_state = AttentionState()
        strategy = get_ditango_planner()
        for group_id in range(num_groups):
            state = computed_group_states.get(group_id)
            if state is None:
                state = reused_group_states.get(group_id, AttentionState())
                if align_stats is not None and not state.is_empty():
                    state = self._apply_state_align_correction(strategy, state, align_stats, group_id, num_groups)
            final_state = AttentionState.merge(final_state, state)
        return final_state

    def _update_state_align_stats(
        self,
        group_id: int,
        fresh_state: AttentionState,
        source_pairs,
        current_stats,
    ):
        if fresh_state.is_empty():
            return current_stats
        old_state = None
        for source_group_id, source_state in source_pairs:
            if source_group_id == group_id:
                old_state = source_state
                break
        if old_state is None or old_state.is_empty():
            return current_stats
        out_delta = fresh_state.out.detach().float() - old_state.out.detach().to(fresh_state.out.device).float()
        lse_delta = fresh_state.lse.detach().float() - old_state.lse.detach().to(fresh_state.lse.device).float()
        out_scale, out_shift = self._fit_scalar_affine(old_state.out, fresh_state.out)
        out_head_scale, out_head_shift = self._fit_head_affine(old_state.out, fresh_state.out)
        if current_stats is None:
            return {
                "source_groups": [group_id],
                "out_delta": out_delta,
                "lse_delta": lse_delta,
                "out_scale": out_scale,
                "out_shift": out_shift,
                "out_head_scale": out_head_scale,
                "out_head_shift": out_head_shift,
                "count": 1,
            }
        count = current_stats["count"]
        count += 1
        alpha = 1.0 / float(count)
        current_stats["source_groups"].append(group_id)
        current_stats["out_delta"] = current_stats["out_delta"] + (out_delta - current_stats["out_delta"]) * alpha
        current_stats["lse_delta"] = current_stats["lse_delta"] + (lse_delta - current_stats["lse_delta"]) * alpha
        current_stats["out_scale"] = current_stats["out_scale"] + (out_scale - current_stats["out_scale"]) * alpha
        current_stats["out_shift"] = current_stats["out_shift"] + (out_shift - current_stats["out_shift"]) * alpha
        current_stats["out_head_scale"] = current_stats["out_head_scale"] + (out_head_scale - current_stats["out_head_scale"]) * alpha
        current_stats["out_head_shift"] = current_stats["out_head_shift"] + (out_head_shift - current_stats["out_head_shift"]) * alpha
        current_stats["count"] = count
        return current_stats

    def _fit_scalar_affine(self, old: torch.Tensor, fresh: torch.Tensor) -> tuple[float, float]:
        x = old.detach().float().reshape(-1)
        y = fresh.detach().to(old.device).float().reshape(-1)
        x_mean = x.mean()
        y_mean = y.mean()
        var = ((x - x_mean) ** 2).mean().clamp_min(1e-12)
        scale = (((x - x_mean) * (y - y_mean)).mean() / var).item()
        shift = (y_mean - scale * x_mean).item()
        return float(scale), float(shift)

    def _fit_head_affine(self, old: torch.Tensor, fresh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        old = old.detach()
        fresh = fresh.detach().to(old.device)
        if old.ndim != 4:
            scale, shift = self._fit_scalar_affine(old, fresh)
            return torch.tensor([scale], device=old.device), torch.tensor([shift], device=old.device)
        x = old.float().permute(2, 0, 1, 3).reshape(old.shape[2], -1)
        y = fresh.float().permute(2, 0, 1, 3).reshape(fresh.shape[2], -1)
        x_mean = x.mean(dim=1)
        y_mean = y.mean(dim=1)
        var = ((x - x_mean[:, None]) ** 2).mean(dim=1).clamp_min(1e-12)
        scale = ((x - x_mean[:, None]) * (y - y_mean[:, None])).mean(dim=1) / var
        shift = y_mean - scale * x_mean
        return scale, shift

    def _apply_state_align_correction(
        self,
        strategy: DiTangoPlanner,
        state: AttentionState,
        align_stats,
        group_id: int,
        num_groups: int,
    ) -> AttentionState:
        mode = strategy.groupwise_state_align_mode
        if mode in {"delta", "lse_delta"}:
            out_scale = 0.0 if mode == "lse_delta" else float(strategy.groupwise_state_align_out_scale)
            return state.corrected(
                align_stats["out_delta"] * out_scale,
                align_stats["lse_delta"] * float(strategy.groupwise_state_align_lse_scale),
            )
        if mode in {"out_affine", "dist_out_affine"}:
            transformed = AttentionState(
                out=state.out * align_stats["out_scale"] + align_stats["out_shift"],
                lse=state.lse,
            )
            if mode == "dist_out_affine":
                weight = self._state_align_distance_weight(
                    align_stats["source_groups"],
                    group_id,
                    num_groups,
                    strategy.groupwise_state_align_distance_tau,
                )
                return AttentionState(
                    out=state.out + (transformed.out - state.out) * weight,
                    lse=state.lse,
                )
            return transformed
        if mode == "out_affine_per_head":
            scale = align_stats["out_head_scale"].to(device=state.out.device, dtype=state.out.dtype)
            shift = align_stats["out_head_shift"].to(device=state.out.device, dtype=state.out.dtype)
            if state.out.ndim == 4:
                scale = scale.view(1, 1, -1, 1)
                shift = shift.view(1, 1, -1, 1)
            return AttentionState(
                out=state.out * scale + shift,
                lse=state.lse,
            )
        if mode == "out_affine_lse_delta":
            return AttentionState(
                out=state.out * align_stats["out_scale"] + align_stats["out_shift"],
                lse=state.lse
                + align_stats["lse_delta"].to(device=state.lse.device, dtype=state.lse.dtype)
                * float(strategy.groupwise_state_align_lse_scale),
            )
        return state

    def _state_align_distance_weight(
        self,
        source_groups: list[int],
        target_group: int,
        num_groups: int,
        tau: float,
    ) -> float:
        def distance(a: int, b: int) -> int:
            raw = abs(a - b)
            return min(raw, num_groups - raw)

        nearest = min(distance(source, target_group) for source in source_groups)
        return float(torch.exp(torch.tensor(-float(nearest) / max(float(tau), 1e-6))).item())

    def _stale_kv_reuse_supported(self, is_varlen: bool) -> bool:
        return (not is_varlen) and self.intra_group_size == 1 and self.ulysses_size == 1

    def _reuse_group_state(
        self,
        strategy: DiTangoPlanner,
        group_id: int,
        q: torch.Tensor,
        is_varlen: bool,
        **kwargs,
    ) -> AttentionState:
        if not strategy.groupwise_reuse_stale_kv or not self._stale_kv_reuse_supported(is_varlen):
            return strategy.reuse(self.layer_id, group_id=group_id)
        cached = strategy.reuse_kv(self.layer_id, group_id=group_id)
        if cached is None:
            return AttentionState()
        cached_k, cached_v, _ = cached
        block_out, block_lse, _ = self.attn_backend(q, cached_k, cached_v, **kwargs)
        state = AttentionState()
        state.update(block_out, block_lse)
        return state

    def _merge_cached_group_states(
        self,
        strategy: DiTangoPlanner,
        num_groups: int,
        q: Optional[torch.Tensor] = None,
        is_varlen: bool = False,
        **kwargs,
    ) -> AttentionState:
        final_state = AttentionState()
        for group_id in range(num_groups):
            if q is None:
                state = strategy.reuse(self.layer_id, group_id=group_id)
            else:
                state = self._reuse_group_state(strategy, group_id, q, is_varlen, **kwargs)
            if state.is_empty():
                return AttentionState()
            final_state = AttentionState.merge(final_state, state)
        return final_state
