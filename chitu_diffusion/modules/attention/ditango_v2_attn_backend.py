import torch
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from logging import getLogger
import random

from chitu_core.distributed.parallel_state import get_cp_group, get_up_group
from chitu_diffusion.backend import DiffusionBackend
from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttnBackend
from chitu_diffusion.utils.shared_utils import (update_out_and_lse, 
                                                transpose_and_unsqueeze,
                                                squeeze_and_transpose, 
                                                async_ring_p2p_commit, 
                                                async_ring_p2p_wait_and_update)

logger = getLogger(__name__)


# ==================== Helper Functions ====================

def get_timestep() -> int:
    """Get current diffusion timestep."""
    return DiffusionBackend.generator.current_task.buffer.current_step

@dataclass
class AttentionState:
    """Attention output and log-sum-exp state."""
    out: Optional[torch.Tensor] = None # [b,s,n,d]
    lse: Optional[torch.Tensor] = None # [b,s,n,1] 
    
    def is_empty(self) -> bool:
        return self.lse is None
    
    def update(self, block_out: torch.Tensor, block_lse: torch.Tensor):
        """Update state with new block output."""
        self.out, self.lse = update_out_and_lse(self.out, self.lse, block_out, block_lse)

    def __repr__(self):
        if self.is_empty():
            return f"Empty"
        return f"out: {self.out.shape}, lse: {self.lse.shape}"

    @staticmethod
    def merge(state1: "AttentionState", state2: "AttentionState") -> "AttentionState":
        if state2.is_empty():
            return state1
        else:
            input_lse = squeeze_and_transpose(state2.lse)
            input_out = state2.out
            state1.update(input_out, input_lse)
            return state1


class Ditangov2Attention:
    def __init__(self, layer_id: int):
        self.attn_backend = DiffusionAttnBackend()
        self.group = get_cp_group()
        self.group_rank = self.group.rank_list
        self.cp_size = self.group.group_size
        self.global_rank = torch.distributed.get_rank()
        self.rank_in_cp = self.group.rank_in_group
        self.layer_id = layer_id
        self.intra_group_size = 2 # gpus per node
        assert self.cp_size <= self.intra_group_size or self.group.group_size % self.intra_group_size == 0
        self.ulysses_size = min(DiffusionBackend.args.infer.diffusion.up_limit, self.cp_size, self.intra_group_size)

        self.curr_importance_level = 3

        self.state_buffer: Dict[str, Optional[AttentionState]] = {
            "intra": None,
            "inter": None,
        }

        if self.global_rank == 0:
            logger.info(f"L{layer_id} | Using Ditango Attn v2.")
                  
    def _print_layer0(self, message: str):
        enable = self.global_rank in [0]
        # enable = True
        if enable and self.layer_id == 0:
            logger.info(message)
    
    def _is_varlen_mode(
        self,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
    ) -> bool:
        """Check if using variable-length attention mode."""
        return (cu_seqlens_q is not None and cu_seqlens_k is not None and
                max_seqlen_q is not None and max_seqlen_k is not None)


    # ----------- Get target rank KVs and compute ---------------

    def _local_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs
    ) -> AttentionState:    
        """Compute attention for local KV chunk."""
        block_out, block_lse, _ = self.attn_backend(
            q, k, v,
            **kwargs,
        )
        block_lse = transpose_and_unsqueeze(block_lse)
        return AttentionState(out=block_out, lse=block_lse)

    def _intra_group_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_varlen: bool,
        **kwargs,
    ) -> Tuple[AttentionState, AttentionState]:
        """
        """
        ring_offset = self.rank_in_cp // self.intra_group_size * self.intra_group_size
        ring_size = self.intra_group_size // self.ulysses_size
        ring_next_rank = (self.rank_in_cp - self.ulysses_size) % self.intra_group_size + ring_offset
        ring_prev_rank = (self.rank_in_cp + self.ulysses_size) % self.intra_group_size + ring_offset
        # logger.info(f"Inner group R{self.rank_in_cp} | {ring_size=} {self.ulysses_size=} {ring_next_rank=} {ring_prev_rank=}")

        intra_group_attn_state = AttentionState(None, None)
        
        for ring_step in range(ring_size):
            if is_varlen:
                cu_seqlens_k = kwargs.get("cu_seqlens_k", None)
                data_pack = (k, v, cu_seqlens_k)
            else:
                data_pack = (k, v)

            if ring_step + 1 != ring_size:
                nxt_data_pack = async_ring_p2p_commit(
                    self.group,
                    data_pack,
                    src_rank=ring_prev_rank,
                    dst_rank=ring_next_rank,
                )

            self._print_layer0(f"{ring_step=} | {q.shape=} {k.shape=} {v.shape=}")
            if ring_step == 0:
                local_attn_state = self._local_attn(
                    q, k, v, **kwargs
                )
            else:
                block_out, block_lse, _  = self.attn_backend(
                        q, k, v,
                        **kwargs,
                    ) # out:(b,s,n,d) lse:(b,n,s)
                
                intra_group_attn_state.update(block_out, block_lse)
            
            if ring_step + 1 != ring_size:
                if is_varlen:
                    k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                else:
                    k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)

        return local_attn_state, intra_group_attn_state



    def _global_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_varlen: bool,
        **kwargs,
    ) -> Tuple[AttentionState, AttentionState, AttentionState]:
        
        cp_size = self.group.group_size
        # outer loop
        outer_loop_size = cp_size // self.intra_group_size
        outer_loop_next_rank = (self.rank_in_cp - self.intra_group_size) % cp_size
        outer_loop_prev_rank =  (self.rank_in_cp + self.intra_group_size) % cp_size
        # inner loop
        inner_offset = self.rank_in_cp // self.intra_group_size * self.intra_group_size
        inner_loop_size = self.intra_group_size // self.ulysses_size
        inner_loop_next_rank = (self.rank_in_cp - self.ulysses_size) % self.intra_group_size + inner_offset
        inner_loop_prev_rank = (self.rank_in_cp + self.ulysses_size) % self.intra_group_size + inner_offset
        # (f"Inner group R{self.rank_in_cp} | {inner_loop_size=} {self.ulysses_size=} {inner_loop_next_rank=} {inner_loop_prev_rank=}")

        intra_group_attn_state = AttentionState(None, None)
        outer_group_attn_state = AttentionState(None, None)

        # logger.info(f"Outer loop R{self.rank_in_cp} | {outer_loop_size=} {outer_loop_next_rank=} {outer_loop_prev_rank=}")

        # Outer loop: Switch group
        for outer_loop_step in range(outer_loop_size):
            for inner_loop_step in range(inner_loop_size):

                # communication send
                if is_varlen:
                    cu_seqlens_k = kwargs.get("cu_seqlens_k", None)
                    data_pack = (k, v, cu_seqlens_k)
                else:
                    data_pack = (k, v)

                if inner_loop_step + 1 != inner_loop_size:
                    nxt_data_pack = async_ring_p2p_commit(
                        self.group,
                        data_pack,
                        src_rank=inner_loop_prev_rank,
                        dst_rank=inner_loop_next_rank,
                    )
                elif (inner_loop_step + 1 == inner_loop_size) and (outer_loop_step + 1 != outer_loop_size):
                    # logger.info(f"Rank{self.global_rank} | send to {outer_loop_prev_rank}, recv from {outer_loop_next_rank}")
                    nxt_data_pack = async_ring_p2p_commit(
                        self.group,
                        data_pack,
                        src_rank=outer_loop_prev_rank,
                        dst_rank=outer_loop_next_rank,
                    )

                # Compute
                if inner_loop_step == 0 and outer_loop_step == 0:
                    local_attn_state = self._local_attn(
                        q, k, v, **kwargs
                    )
                else:
                    block_out, block_lse, _  = self.attn_backend(
                        q, k, v,
                        **kwargs,
                    ) # out:(b,s,n,d) lse:(b,n,s)
                    if outer_loop_step == 0:
                        intra_group_attn_state.update(block_out, block_lse)
                    else:
                        outer_group_attn_state.update(block_out, block_lse)
            
        
                # Communicate Recv
                if (inner_loop_step + 1 != inner_loop_size) or (outer_loop_step + 1 != outer_loop_size):
                    if is_varlen:
                        k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                    else:
                        k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)

        return local_attn_state, intra_group_attn_state, outer_group_attn_state
        


    # -------------------- Main Forward Pass --------------------
    
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
        """
        Forward pass with two-phase computation:
        1. Reuse Phase: Reuse cached attention states
        2. Compute Phase: Compute fresh attention with ring-reduce
        """
        is_varlen = self._is_varlen_mode(cu_seqlens_k, cu_seqlens_q, max_seqlen_k, max_seqlen_q)
        seq_dim, head_dim = (0, 1) if is_varlen else (1, 2)

        self.curr_importance_level = self.get_importance()
        # logger.info(f"T{get_timestep()}L{self.layer_id} | Importance: {self.curr_importance_level}")

        # 创建kwargs字典
        attn_kwargs = {
            'cu_seqlens_q': cu_seqlens_q,
            'cu_seqlens_k': cu_seqlens_k,
            'max_seqlen_q': max_seqlen_q,
            'max_seqlen_k': max_seqlen_k,
            'dropout_p': dropout_p,
            'softmax_scale': softmax_scale,
            'causal': causal,
            'window_size': window_size,
            'deterministic': deterministic,
            'return_attn_probs': True
        }


        if self.ulysses_size > 1:
            ulysses_group = get_up_group(self.ulysses_size)
            q = ulysses_group.all_to_all(q, head_dim, seq_dim)  
            k = ulysses_group.all_to_all(k, head_dim, seq_dim)
            v = ulysses_group.all_to_all(v, head_dim, seq_dim)

        self._print_layer0(f"After ulysses_a2a | {q.shape=} {k.shape=} {v.shape=}")


        if self.curr_importance_level == 3: # global attn
            local_state, intra_group_state, inter_group_state = self._global_attn(
                q, k, v, 
                is_varlen,
                **attn_kwargs,
            )
            self.state_buffer["intra"] = intra_group_state
            self.state_buffer["inter"] = inter_group_state

        elif self.curr_importance_level == 2: # intra-group attn
            local_state, intra_group_state = self._intra_group_attn(
                q, k, v,
                is_varlen,
                **attn_kwargs
            )
            self.state_buffer["intra"] = intra_group_state
            inter_group_state = self.state_buffer["inter"]

        elif self.curr_importance_level == 1: # local attn
            local_state = self._local_attn(
                q, k, v,
                **attn_kwargs
            )
            intra_group_state = self.state_buffer["intra"]
            inter_group_state = self.state_buffer["inter"]

        self._print_layer0(f"T{get_timestep()} | {local_state=} {intra_group_state=} {inter_group_state=}")
        
        
        # ulysses all2all
        final_attn_state = AttentionState.merge(local_state, intra_group_state)
        final_attn_state = AttentionState.merge(final_attn_state, inter_group_state)
        if self.ulysses_size > 1:
            out = ulysses_group.all_to_all(final_attn_state.out, seq_dim, head_dim)      
        else:
            out = final_attn_state.out

        self._print_layer0(f"{out.shape=}")

        return out, None, None

    # -------------------- Strategy --------------------
    def get_importance(self):
        """
        1: Local
        2: Group (usually single node)
        3: Global
        """
        return 3
        # if get_timestep() == 0:
        #     return 3
        # else:
        #     if self.global_rank == 0:
        #         curr_importance_level = random.choice([1,2,3])
        #         importance_tensor = torch.tensor(curr_importance_level, device='cuda')
        #     else:
        #         importance_tensor = torch.tensor(0, device='cuda')  # 创建一个占位tensor

        #     # 从rank 0广播到所有进程
        #     torch.distributed.broadcast(importance_tensor, src=0)

        #     # 将tensor转回int
        #     return importance_tensor.item()
    

