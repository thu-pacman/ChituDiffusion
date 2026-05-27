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
                for _ in range(inner_loop_size - 1):
                    comm_list.append((inner_loop_prev_rank, inner_loop_next_rank))
            elif group_id != 0:
                skip_group_range += 1
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
        current_group_id = 0

        for outer_step in range(outer_loop_size):
            group_state = AttentionState()
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
                is_local_partition = self._is_local_partition_step(outer_step, inner_step)
                if is_local_partition:
                    local_partition_state.update(block_out, block_lse)
                    final_state = AttentionState.merge(final_state, local_partition_state)
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
                    strategy.store(self.layer_id, group_id=current_group_id, group_state=state_to_store)
                if current_group_id == 0:
                    final_state = state_to_store
                else:
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
        strategy = get_ditango_planner()
        num_groups = self.cp_size // self.intra_group_size
        branch_key = strategy._branch_key()
        branch_plan = strategy.ditango_plan.get(branch_key, None)
        if branch_plan is None:
            group_plan = torch.ones((num_groups,), dtype=torch.bool, device=self.group.device)
        else:
            group_plan = branch_plan[self.layer_id].to(self.group.device)

        group_plan = group_plan.clone().bool()
        for group_id in range(num_groups):
            if bool(group_plan[group_id].item()):
                strategy.discard_group_state(self.layer_id, group_id)

        if not bool(group_plan.bool().any().item()):
            final_state = self._merge_cached_group_states(strategy, num_groups)
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
            if strategy.reuse(self.layer_id, group_id=group_id).is_empty():
                group_plan[group_id] = True
                fallback_count += 1

        final_state = AttentionState()
        computed_group_states: Dict[int, AttentionState] = {}
        local_partition_state = AttentionState()
        current_group_id = 0
        compute_group = group_plan.bool().tolist().count(True)

        with Timer.get_timer(f"ditango_sele_attn_{compute_group}"):
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
                        nxt_data_pack = async_ring_p2p_commit(
                            self.group, data_pack, src_rank=prev_rank, dst_rank=next_rank
                        )
                    else:
                        nxt_data_pack = None

                    is_local_partition = self._is_local_partition_step(outer_step, inner_step)
                    if should_compute:
                        block_out, block_lse, _ = self.attn_backend(q, k, v, **kwargs)
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
                        computed_group_states[current_group_id] = state_to_store
                        if current_group_id == 0:
                            final_state = state_to_store
                        else:
                            final_state = AttentionState.merge(final_state, state_to_store)
                    strategy.record_group_decision(self.layer_id, current_group_id, strategy.DECISION_CODE_COMPUTE)
                else:
                    reused_state = strategy.reuse(self.layer_id, group_id=current_group_id)
                    final_state = AttentionState.merge(final_state, reused_state)
                    strategy.record_group_decision(self.layer_id, current_group_id, strategy.DECISION_CODE_REUSE)

                current_group_id = (current_group_id + 1) % num_groups

        strategy.record_step_fallback(self.layer_id, fallback_count)
        return final_state, computed_group_states, local_partition_state

    def _merge_cached_group_states(self, strategy: DiTangoPlanner, num_groups: int) -> AttentionState:
        final_state = AttentionState()
        for group_id in range(num_groups):
            state = strategy.reuse(self.layer_id, group_id=group_id)
            if state.is_empty():
                return AttentionState()
            final_state = AttentionState.merge(final_state, state)
        return final_state
