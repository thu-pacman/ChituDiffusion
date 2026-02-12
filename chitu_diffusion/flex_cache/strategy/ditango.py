import torch
import functools
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from logging import getLogger
import random
from chitu_diffusion.backend import DiffusionBackend, CFGType
from chitu_diffusion.flex_cache.flexcache_manager import FlexCacheStrategy

from chitu_core.distributed.parallel_state import get_cp_group, get_up_group
from chitu_diffusion.backend import DiffusionBackend
from chitu_diffusion.utils.shared_utils import (update_out_and_lse, 
                                                transpose_and_unsqueeze,
                                                squeeze_and_transpose, 
                                                async_ring_p2p_commit, 
                                                async_ring_p2p_wait_and_update)
from chitu_diffusion.bench import Timer, MagLogger



logger = getLogger(__name__)

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


class DiTangoStrategy(FlexCacheStrategy):
    def __init__(self):
        pass
        
    def get_reuse_key(self, layer_id: int, level: str, importance: int, **kwargs):
        is_pos = DiffusionBackend.cfg_type == CFGType.POS
        branch_key = 'pos' if is_pos else 'neg'
        if importance < 3:
            return f"{branch_key}_{layer_id}_{level}"
        else:
            return None
        
    def reuse(self, layer_id: int, importance: int, **kwargs):
        intra_state, inter_state = AttentionState(), AttentionState()
        msg = "Reuse: "
        if importance == 1:
            intra_key = self.get_reuse_key(layer_id, "intra", importance)
            inter_key = self.get_reuse_key(layer_id, "inter", importance)
            if intra_key is not None and intra_key in DiffusionBackend.flexcache.cache:
                intra_state = DiffusionBackend.flexcache.cache[intra_key]
                msg += f" {intra_key}: {intra_state}"
            if inter_key is not None and inter_key in DiffusionBackend.flexcache.cache:
                inter_state = DiffusionBackend.flexcache.cache[inter_key]
                msg += f" {inter_key}: {inter_state}"

        elif importance == 2:
            inter_key = self.get_reuse_key(layer_id, "inter", importance)
            if inter_key is not None and inter_key in DiffusionBackend.flexcache.cache:
                inter_state = DiffusionBackend.flexcache.cache[inter_key]
                msg += f" {inter_key}: {inter_state}"

        if layer_id == 0:
            logger.info(msg)
        return intra_state, inter_state

    
    def get_store_key(self, layer_id: int, level: str, importance: int,  **kwargs):
        is_pos = DiffusionBackend.cfg_type == CFGType.POS
        branch_key = 'pos' if is_pos else 'neg'
        if importance > 1:
            return f"{branch_key}_{layer_id}_{level}"
        else:
            return None
        
    def store(self, layer_id: int, 
              importance: int,
              intra_state: Optional[AttentionState], 
              inter_state: Optional[AttentionState], 
              **kwargs):
        msg = "Store:"
        if not (intra_state is None or intra_state.is_empty()):
            intra_key = self.get_store_key(layer_id, "intra", importance)
            if intra_key is not None:
                DiffusionBackend.flexcache.cache[intra_key] = intra_state
                msg += f" {intra_key}: {intra_state}"
        if not (inter_state is None or inter_state.is_empty()):
            inter_key = self.get_store_key(layer_id, "inter", importance)
            if inter_key is not None:
                DiffusionBackend.flexcache.cache[inter_key] = inter_state
                msg += f" {inter_key}: {inter_state}"
        if layer_id == 0:
            logger.info(msg)


    
    def reset_state(self):
        self.last_residual_norm = None
        self.accumulated_ratio = 1
        self.accumulated_skip_error = 0.0
        self.step_importance = 3
        self.error_log = []
        DiffusionBackend.flexcache.cache.clear() 
        logger.debug("DiTango state reset")

    
    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        # Replace self-attention module with DitangoAttn

        if not hasattr(module, '_original_attn'):
            module._original_attn = module.blocks[0].self_attn.attn_func
        
        for layer_id, block in enumerate(module.blocks):
            block.self_attn.attn_func = Ditangov2Attention(
                layer_id=layer_id
            )

        logger.info(f"Module {module.__class__.__name__} wrapped with ditango strategy")
    
    
    def unwrap_module(self, module: torch.nn.Module):
        # if hasattr(module, '_original_forward'):
        #     module.model_compute = module._original_forward
        #     delattr(module, '_original_forward')
            
        if hasattr(module, '_original_attn'):
            for _, block in enumerate(module.blocks):
                block.self_attn.attn_func = module._original_attn
            delattr(module, '_original_attn')
                
        logger.info(f"Module {module.__class__.__name__} unwrapped")


class Ditangov2Attention:
    def __init__(self, layer_id: int):
        self.attn_backend = DiffusionBackend.attn
        self.group = get_cp_group()
        self.group_rank = self.group.rank_list
        self.cp_size = self.group.group_size
        self.global_rank = torch.distributed.get_rank()
        self.rank_in_cp = self.group.rank_in_group
        self.layer_id = layer_id
        self.intra_group_size = min(self.cp_size, 4) # gpus per node

        # ============= strategy ===============
        self.enable_cfg_parallel = DiffusionBackend.args.infer.diffusion.enable_cfg_parallel
        self.prev_local_attn_output_mean = None
        self.importance = 3
        self.diff_mean = None
        self.diff_log = []
        
        assert self.cp_size <= self.intra_group_size or self.group.group_size % self.intra_group_size == 0
        self.ulysses_size = min(DiffusionBackend.args.infer.diffusion.up_limit, self.cp_size, self.intra_group_size)

        if self.global_rank == 0:
            logger.info(f"L{layer_id} | Using Ditango Attn v2.")

    # ===================== Helper function ================
                  
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
    
    # ==================== strategy =====================

    def _update_importance(self, local_out: torch.Tensor):
        # 目前的strategy是写死的，直接通过get_importance得到，这个函数只做记录用

        block_out_mean = torch.mean(local_out, dim=-1)
        curr_step = get_timestep()
        if curr_step == 0:
            self.prev_local_attn_output_mean = block_out_mean
            self.importance = 3
        else:
            diff = torch.norm(block_out_mean - self.prev_local_attn_output_mean)
            self.group.all_reduce(diff)

            MagLogger.log_magnitude(
                diff,
                curr_step,
                self.layer_id,
                "diff"
            )

            diff = diff.item()
            self.prev_local_attn_output_mean = block_out_mean
            self.diff_log.append(diff)

    def _get_importance(self):
        """
        1: Local
        2: Group (usually single node)
        3: Global
        """
        # importance = DiffusionBackend.flexcache.strategy.current_step_importance
        curr_step = get_timestep()
        if curr_step < 4 or curr_step > 45 or curr_step % 5 == 0:
            self.importance = 3
        else:
            self.importance = 1

        if self.layer_id % 5 == 0:
            importance = torch.tensor(self.importance)
            MagLogger.log_magnitude(
                importance,
                curr_step,
                self.layer_id,
                "importance"
            )
        self._print_layer0(f"T{get_timestep()}L{self.layer_id} | =================> Ditango Importance: {self.importance} <================")
        return self.importance
        

    # ----------- Get target rank KVs and compute ---------------
    @Timer.get_timer("attn1")
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
        if self.enable_cfg_parallel or DiffusionBackend.cfg_type == CFGType.POS:
            self._update_importance(block_out)
        # branch_key = 'pos' if DiffusionBackend.cfg_type == CFGType.POS else 'neg'
        # if self.global_rank == 0 and self.layer_id % 5 == 0 and branch_key == 'pos':
        #     block_out_mean = block_out.mean(dim=-1)
        #     if get_timestep() == 0:
        #         self.prev_local_attn_output = torch.zeros_like(block_out)
        #         self.prev_local_attn_output_mean = torch.zeros_like(block_out_mean)

        #     diff = block_out - self.prev_local_attn_output
        #     mean_diff = block_out_mean - self.prev_local_attn_output_mean

        #     self.prev_local_attn_output = block_out
        #     self.prev_local_attn_output_mean = block_out_mean


        #     MagLogger.log_magnitude(tensor=diff, 
        #                             step=get_timestep(),
        #                             layer=self.layer_id,
        #                             name=f"ground_truth_diff_{branch_key}")
            
        #     MagLogger.log_magnitude(tensor=mean_diff, 
        #                             step=get_timestep(),
        #                             layer=self.layer_id,
        #                             name=f"mean_diff_{branch_key}")

        return AttentionState(out=block_out, lse=block_lse)

    @Timer.get_timer("attn2")
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

        if self.layer_id == 10:
            logger.info(f"Intra group R{self.rank_in_cp} | {ring_size=} {self.ulysses_size=} {ring_next_rank=} {ring_prev_rank=}")

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


    @Timer.get_timer("attn3")
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
        # self._print_layer0(f"Inner group R{self.rank_in_cp} | {inner_loop_size=} {self.ulysses_size=} {inner_loop_next_rank=} {inner_loop_prev_rank=}")

        intra_group_attn_state = AttentionState(None, None)
        outer_group_attn_state = AttentionState(None, None)

        # self._print_layer0(f"Outer loop R{self.rank_in_cp} | {outer_loop_size=} {outer_loop_next_rank=} {outer_loop_prev_rank=}")

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
    
    @Timer.get_timer("ditango_attn")
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
        step_importance = self._get_importance()
        ditango: DiTangoStrategy = DiffusionBackend.flexcache.strategy

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

        # self._print_layer0(f"After ulysses_a2a | {q.shape=} {k.shape=} {v.shape=}")


        if step_importance == 3: # global attn
            local_state, intra_group_state, inter_group_state = self._global_attn(
                q, k, v, 
                is_varlen,
                **attn_kwargs,
            )
            # self._print_layer0(f"After global attn | {local_state=} {intra_group_state=} {inter_group_state=}")
            ditango.store(
                self.layer_id,
                step_importance,
                intra_group_state,
                inter_group_state
            )

        elif step_importance == 2: # intra-group attn
            local_state, intra_group_state = self._intra_group_attn(
                q, k, v,
                is_varlen,
                **attn_kwargs
            )
            # self._print_layer0(f"After intra attn | {local_state=} {intra_group_state=}")
            _, inter_group_state = ditango.reuse(self.layer_id, step_importance)
            ditango.store(self.layer_id, 
                          step_importance,
                          intra_group_state, 
                          None)

        elif step_importance == 1: # local attn
            local_state = self._local_attn(
                q, k, v,
                **attn_kwargs
            )
            intra_group_state, inter_group_state = ditango.reuse(self.layer_id, step_importance)

        # self._print_layer0(f"T{get_timestep()} | {local_state=} {intra_group_state=} {inter_group_state=}")
        
        
        # ulysses all2all
        final_attn_state = AttentionState.merge(local_state, intra_group_state)
        final_attn_state = AttentionState.merge(final_attn_state, inter_group_state)

        if self.ulysses_size > 1:
            out = ulysses_group.all_to_all(final_attn_state.out, seq_dim, head_dim)      
        else:
            out = final_attn_state.out

        self._print_layer0(f"{out.shape=}")

        return out, None, None

    



    
