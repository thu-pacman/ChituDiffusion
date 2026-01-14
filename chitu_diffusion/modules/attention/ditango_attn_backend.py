import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass
from logging import getLogger

from chitu_core.distributed.parallel_state import get_cp_group, get_up_group
from chitu_diffusion.backend import DiffusionBackend
from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttnBackend
from chitu_diffusion.utils.shared_utils import update_out_and_lse, squeeze_and_transpose

logger = getLogger(__name__)


# ==================== Helper Functions ====================

def get_timestep() -> int:
    """Get current diffusion timestep."""
    return DiffusionBackend.generator.current_task.buffer.current_step


# ==================== Data Structures ====================

@dataclass
class BlockInfo:
    """Information about cache blocks."""
    total_blocks: int
    local_block_id: int
    target_block_id: int


@dataclass
class RingConfig:
    """Configuration for ring communication."""
    ring_steps: int
    ulysses_size: int
    ring_steps_to_update: int
    ring_prev_rank: int
    ring_next_rank: int


@dataclass
class AttentionState:
    """Attention output and log-sum-exp state."""
    out: Optional[torch.Tensor] = None
    lse: Optional[torch.Tensor] = None
    
    def is_empty(self) -> bool:
        return self.lse is None
    
    def update(self, block_out: torch.Tensor, block_lse: torch.Tensor):
        """Update state with new block output."""
        self.out, self.lse = update_out_and_lse(self.out, self.lse, block_out, block_lse)


# ==================== Cache Management ====================

class AttentionStateCache:
    """Cache for attention states across context parallel blocks."""
    
    def __init__(self, cp_size: int, layer_id: int):
        self.cp_size = cp_size
        self.layer_id = layer_id
        
        # Cache state
        self.curr_cp_stride = cp_size
        self.curr_block_num = cp_size
        self.out_block_cache: List[Optional[torch.Tensor]] = [None] * cp_size
        self.lse_block_cache: List[Optional[torch.Tensor]] = [None] * cp_size
        self.block_age: List[int] = [0] * cp_size
        
        # Metrics
        self.block_size_mb: Optional[float] = None
        self.memory_constraint: float = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    
    # -------------------- Cache Operations --------------------
    
    def get_block(self, block_id: int) -> AttentionState:
        """Retrieve cached attention state for a block."""
        if block_id >= len(self.out_block_cache):
            logger.error(f"T{get_timestep()} | Cache not ready for block {block_id}")
            return AttentionState()
        
        return AttentionState(
            out=self.out_block_cache[block_id],
            lse=self.lse_block_cache[block_id]
        )
    
    def store_block(self, block_id: int, out: torch.Tensor, lse: torch.Tensor):
        """Store attention state for a block."""
        self.out_block_cache[block_id] = out
        self.lse_block_cache[block_id] = lse
        self.block_age[block_id] = 1
    
    def evict_block(self, block_id: int):
        """Remove cached state for a block."""
        self.out_block_cache[block_id] = None
        self.lse_block_cache[block_id] = None
        self.block_age[block_id] = 0
    
    # -------------------- Cache Reshaping --------------------
    
    def adjust_cache_shape(self, new_cp_stride: int, num_chunks_per_block: int):
        """Adjust cache structure for new CP stride."""
        assert self.cp_size % new_cp_stride == 0, f"Unsupported cp stride: {new_cp_stride}"
        
        # No change needed for these cases
        if new_cp_stride in (self.cp_size, self.curr_cp_stride):
            return
        
        # Full to partial: reinitialize cache
        if self.curr_cp_stride == self.cp_size:
            new_block_num = self.cp_size // num_chunks_per_block
            self._reinitialize_cache(new_block_num)
    
    def merge_and_evict_blocks(self, new_cp_stride: int, target_block_id: int):
        """Merge blocks and evict as needed for new stride."""
        # Merge if going from full->partial or partial->larger
        should_merge = (self.curr_cp_stride == self.cp_size or 
                       new_cp_stride > self.curr_cp_stride)
        
        if should_merge:
            new_block_num = self.cp_size // new_cp_stride
            self._merge_blocks(new_block_num)
            self.curr_block_num = new_block_num
            self.curr_cp_stride = new_cp_stride
        
        # Evict based on new stride
        if new_cp_stride == self.cp_size:
            self._clear_all_blocks()
        else:
            self.evict_block(target_block_id)
    
    def _merge_blocks(self, new_block_num: int):
        """Merge existing blocks into fewer, larger blocks."""
        assert self.curr_block_num % new_block_num == 0, \
            f"Cannot merge {self.curr_block_num} blocks into {new_block_num}"
        
        chunks_to_merge = self.curr_block_num // new_block_num
        new_out_cache = [None] * new_block_num
        new_lse_cache = [None] * new_block_num
        new_age = [0] * new_block_num
        
        for i in range(new_block_num):
            merged_state = AttentionState()
            total_age = 0
            
            # Merge consecutive chunks
            for j in range(chunks_to_merge):
                block_id = i * chunks_to_merge + j
                cached_state = self.get_block(block_id)
                total_age += self.block_age[block_id]
                
                if not cached_state.is_empty():
                    merged_state.update(cached_state.out, cached_state.lse)
            
            # Store merged result
            if not merged_state.is_empty():
                merged_state.lse = squeeze_and_transpose(merged_state.lse)
                new_out_cache[i] = merged_state.out
                new_lse_cache[i] = merged_state.lse
                new_age[i] = total_age / chunks_to_merge
        
        self.out_block_cache = new_out_cache
        self.lse_block_cache = new_lse_cache
        self.block_age = new_age
    
    def _reinitialize_cache(self, new_block_num: int):
        """Reinitialize cache with new block count."""
        self.curr_block_num = new_block_num
        self.out_block_cache = [None] * new_block_num
        self.lse_block_cache = [None] * new_block_num
        self.block_age = [0] * new_block_num
    
    def _clear_all_blocks(self):
        """Clear all cached blocks."""
        for i in range(self.curr_block_num):
            self.evict_block(i)
    
    # -------------------- Aging and Metrics --------------------
    
    def update_block_age(self):
        """Increment age for all non-empty blocks."""
        for block_id, lse in enumerate(self.lse_block_cache):
            if lse is not None:
                self.block_age[block_id] += 1
    
    def report_cache_status(self):
        """Log detailed cache status information."""
        timestep = get_timestep()
        cached_blocks = sum(1 for out in self.out_block_cache if out is not None)
        
        # Calculate memory usage
        block_size_mb = self._calculate_block_size()
        total_cache_mb = block_size_mb * cached_blocks
        current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        
        # Generate status visualization
        block_status = ['O' if out is not None else 'X' for out in self.out_block_cache]
        
        logger.info(f"===== Cache Status: Layer {self.layer_id}, Timestep {timestep} =====")
        logger.info(f"CP Size: {self.cp_size} | CP Stride: {self.curr_cp_stride}")
        logger.info(f"Block Count: {self.curr_block_num} | Filled: {cached_blocks} "
                   f"({cached_blocks/self.curr_block_num*100:.1f}%)")
        logger.info(f"Block Status: [{' '.join(block_status)}]")
        logger.info(f"Memory - Block: {block_size_mb:.2f}MB | Cache: {total_cache_mb:.2f}MB | "
                   f"GPU: {current_memory_mb:.2f}/{self.memory_constraint:.2f}MB")
        
        # Log tensor shapes if available
        self._log_tensor_shapes()
        logger.info("=" * 80)
    
    def _calculate_block_size(self) -> float:
        """Calculate size of a single cache block in MB."""
        for block_id in range(len(self.out_block_cache)):
            out = self.out_block_cache[block_id]
            lse = self.lse_block_cache[block_id]
            
            if out is not None:
                size_bytes = out.element_size() * out.nelement()
                if lse is not None:
                    size_bytes += lse.element_size() * lse.nelement()
                
                block_size_mb = size_bytes / (1024 ** 2)
                self.block_size_mb = block_size_mb
                return block_size_mb
        
        return 0.0
    
    def _log_tensor_shapes(self):
        """Log shapes of cached tensors."""
        for block_id in range(len(self.out_block_cache)):
            out = self.out_block_cache[block_id]
            lse = self.lse_block_cache[block_id]
            
            if out is not None:
                logger.info(f"Block {block_id} - OUT: {tuple(out.shape)}, "
                           f"LSE: {tuple(lse.shape) if lse is not None else 'None'}")
                return


# ==================== Attention Implementation ====================

class DitangoAttention:
    """
    DiTango Attention: Context Parallel Attention with State Reuse.
    
    Combines attention state caching and flash attention computation for efficient
    diffusion model inference with context parallelism.
    """
    
    def __init__(self, ulysses_limit: int, layer_id: int):
        self.layer_id = layer_id
        self.attn_backend = DiffusionAttnBackend()
        
        # Distributed setup
        self.cp_group = get_cp_group()
        self.cp_size = self.cp_group.group_size
        self.global_rank = torch.distributed.get_rank()
        self.local_chunk_id = self.cp_group.rank_in_group
        
        # State management
        self.cache = AttentionStateCache(self.cp_size, layer_id)
        self.target_chunk_id = self.local_chunk_id
        self.reuse_phase_done = True
        
        # Configuration
        self.ulysses_limit = ulysses_limit
        
        if self.global_rank == 0:
            logger.info(f"L{layer_id} | Initialized DiTango Attention")
    
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
        # Setup
        use_varlen = self._is_varlen_mode(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
        curr_cp_stride, next_cp_stride = self._get_current_and_next_stride()
        
        # Initialize reuse phase
        self.reuse_phase_done = (curr_cp_stride == self.cp_size)
        if curr_cp_stride == self.cp_size:
            self.target_chunk_id = self.local_chunk_id
        
        block_info = self._get_block_info()
        
        logger.info(f"{q.shape=} {k.shape=} {v.shape}")
        # Phase 1: Reuse cached attention states
        reused_state, k, v, cu_seqlens_k = self._reuse_phase(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            block_info, curr_cp_stride, use_varlen,
            dropout_p, softmax_scale, causal, window_size, deterministic
        )
        if not reused_state.is_empty():
            logger.info(f"{reused_state.out.shape=}")
        logger.info(f"After reuse: {k.shape=} {v.shape=}")

        # Adjust cache for new stride
        ring_config = self._prepare_compute_phase(curr_cp_stride, next_cp_stride)
        block_info = self._get_block_info()
        
        # Phase 2: Compute fresh attention states
        final_state = self._compute_phase(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            reused_state, ring_config, block_info, curr_cp_stride, next_cp_stride,
            use_varlen, dropout_p, softmax_scale, causal, window_size, deterministic
        )
        
        # Cleanup and prepare for next iteration
        self._finalize_iteration(curr_cp_stride, next_cp_stride, block_info)
        
        return final_state.out, None, None
    
    # -------------------- Phase 1: Reuse --------------------
    
    def _reuse_phase(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
        block_info: BlockInfo,
        curr_cp_stride: int,
        use_varlen: bool,
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        deterministic: bool,
    ) -> Tuple[AttentionState, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Reuse Phase: Aggregate cached attention states and optionally compute local attention.
        
        Returns:
            - Aggregated attention state from cache
            - Updated k, v tensors (may be transformed via P2P)
            - Updated cu_seqlens_k (if varlen)
        """
        reused_state = AttentionState()
        
        # Transform KV if needed (async P2P communication)
        if self.target_chunk_id != self.local_chunk_id:
            k, v, cu_seqlens_k = self._transform_kv_for_target_chunk(
                k, v, cu_seqlens_k, block_info, curr_cp_stride, use_varlen
            )
            
            # Overlap: Reuse cached blocks while waiting for P2P
            reused_state = self._aggregate_cached_blocks(block_info.total_blocks, block_info.target_block_id)
            
            # Overlap: Compute local attention while waiting for P2P
            local_state = self._compute_local_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p, softmax_scale, causal, window_size, deterministic
            )
            reused_state.update(local_state.out, local_state.lse)
            
            # Wait for P2P to complete
            k, v, cu_seqlens_k = self._wait_for_kv_transform(k, v, cu_seqlens_k, use_varlen)
        else:
            # No KV transform needed, just aggregate cache
            reused_state = self._aggregate_cached_blocks(block_info.total_blocks, block_info.target_block_id)
        
        return reused_state, k, v, cu_seqlens_k
    
    def _transform_kv_for_target_chunk(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_k: Optional[torch.Tensor],
        block_info: BlockInfo,
        curr_cp_stride: int,
        use_varlen: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Send/receive KV tensors to get target chunk's data."""
        block_id_stride = (block_info.target_block_id - block_info.local_block_id) % block_info.total_blocks
        send_block_id = (block_info.local_block_id - block_id_stride) % block_info.total_blocks
        inner_block_rank = self.local_chunk_id % curr_cp_stride
        
        recv_rank = block_info.target_block_id * curr_cp_stride + inner_block_rank
        send_rank = send_block_id * curr_cp_stride + inner_block_rank
        
        data_pack = (k, v, cu_seqlens_k) if use_varlen else (k, v)
        self._pending_kv_data = self._async_ring_p2p_commit(data_pack, recv_rank, send_rank)
        
        self._log_debug(f"Receiving target chunk {self.target_chunk_id} from rank {recv_rank}")
        
        return k, v, cu_seqlens_k
    
    def _wait_for_kv_transform(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_k: Optional[torch.Tensor],
        use_varlen: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Wait for async KV transform to complete."""
        received_data = self._async_ring_p2p_wait(self._pending_kv_data)
        
        if use_varlen:
            k, v, cu_seqlens_k = received_data
        else:
            k, v = received_data
        
        return k, v, cu_seqlens_k
    
    def _aggregate_cached_blocks(self, total_blocks: int, target_block_id: int) -> AttentionState:
        """Aggregate attention states from all cached blocks except target."""
        if self.reuse_phase_done:
            return AttentionState()
        
        aggregated_state = AttentionState()
        
        for i in range(total_blocks - 1):
            cached_block_id = (target_block_id + 1 + i) % total_blocks
            cached_state = self.cache.get_block(cached_block_id)
            
            if not cached_state.is_empty():
                self._log_debug(f"Reusing cached block {cached_block_id}")
                aggregated_state.update(cached_state.out, cached_state.lse)
            else:
                # Should only happen for local chunk in full CP mode
                assert total_blocks == self.cp_size and cached_block_id == self.local_chunk_id, \
                    f"Failed to get cached block {cached_block_id}"
        
        self.reuse_phase_done = True
        return aggregated_state
    
    def _compute_local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        deterministic: bool,
    ) -> AttentionState:
        """Compute attention for local KV chunk."""
        block_out, block_lse, _ = self.attn_backend(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        
        return AttentionState(out=block_out, lse=block_lse)
    
    # -------------------- Phase 2: Compute --------------------
    
    def _compute_phase(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
        reused_state: AttentionState,
        ring_config: RingConfig,
        block_info: BlockInfo,
        curr_cp_stride: int,
        next_cp_stride: int,
        use_varlen: bool,
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        deterministic: bool,
    ) -> AttentionState:
        """
        Compute Phase: Ring-reduce attention computation with Ulysses parallelism.
        
        Combines ring communication with Ulysses all-to-all for efficient
        distributed attention computation.
        """
        final_state = reused_state
        fresh_state = AttentionState()
        
        # Get dimensions for Ulysses all-to-all
        seq_dim, head_dim = (0, 1) if use_varlen else (1, 2)
        
        # Ulysses all-to-all: distribute heads
        q, k, v = self._apply_ulysses_transform(
            q, k, v, ring_config.ulysses_size, head_dim, seq_dim
        )
        
        logger.info(f"After ulysses transforme: {q.shape=} {k.shape=} {v.shape=}")

        if use_varlen:
            max_seqlen_q *= ring_config.ulysses_size
            max_seqlen_k *= ring_config.ulysses_size
        
        target_block_id = block_info.target_block_id
        
        # Ring-reduce loop
        for ring_step in range(ring_config.ring_steps):
            # Setup async communication for next step
            if ring_step + 1 < ring_config.ring_steps:
                pending_kv = self._start_ring_communication(
                    k, v, cu_seqlens_k, ring_config, use_varlen
                )
            
            # Compute attention for current KV chunk
            block_state = self._compute_local_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p, softmax_scale, causal, window_size, deterministic
            )
            fresh_state.update(block_state.out, block_state.lse)
            
            # Wait for next KV chunk
            if ring_step + 1 < ring_config.ring_steps:
                k, v, cu_seqlens_k = self._complete_ring_communication(
                    pending_kv, use_varlen
                )
            
            # Update cache periodically
            if (ring_step + 1) % ring_config.ring_steps_to_update == 0:
                final_state, fresh_state, target_block_id = self._update_cache_and_state(
                    final_state, fresh_state, target_block_id,
                    block_info.total_blocks, curr_cp_stride, next_cp_stride,
                    ring_config.ulysses_size, seq_dim, head_dim
                )
        
        return final_state
    
    def _apply_ulysses_transform(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ulysses_size: int,
        head_dim: int,
        seq_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Ulysses all-to-all transformation to distribute heads."""
        if ulysses_size <= 1:
            return q, k, v
        
        ulysses_group = get_up_group(ulysses_size)
        q = ulysses_group.all_to_all(q, head_dim, seq_dim)
        k = ulysses_group.all_to_all(k, head_dim, seq_dim)
        v = ulysses_group.all_to_all(v, head_dim, seq_dim)
        
        return q, k, v
    
    def _start_ring_communication(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_k: Optional[torch.Tensor],
        ring_config: RingConfig,
        use_varlen: bool,
    ):
        """Start asynchronous ring communication for next KV chunk."""
        data_pack = (k, v, cu_seqlens_k) if use_varlen else (k, v)
        return self._async_ring_p2p_commit(
            data_pack,
            src_rank=ring_config.ring_prev_rank,
            dst_rank=ring_config.ring_next_rank,
        )
    
    def _complete_ring_communication(
        self,
        pending_kv,
        use_varlen: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Complete asynchronous ring communication."""
        received_data = self._async_ring_p2p_wait(pending_kv)
        
        if use_varlen:
            k, v, cu_seqlens_k = received_data
        else:
            k, v = received_data
            cu_seqlens_k = None
        
        return k, v, cu_seqlens_k
    
    def _update_cache_and_state(
        self,
        final_state: AttentionState,
        fresh_state: AttentionState,
        target_block_id: int,
        total_blocks: int,
        curr_cp_stride: int,
        next_cp_stride: int,
        ulysses_size: int,
        seq_dim: int,
        head_dim: int,
    ) -> Tuple[AttentionState, AttentionState, int]:
        """Update cache with fresh state and merge into final state."""
        # Reverse Ulysses transformation
        fresh_state.lse = squeeze_and_transpose(fresh_state.lse)

        if ulysses_size > 1:
            fresh_state = self._reverse_ulysses_transform(
                fresh_state, ulysses_size, seq_dim, head_dim
            )
        
        # Merge fresh state into final state
        final_state.update(fresh_state.out, fresh_state.lse)
        
        # Cache fresh state if needed
        should_cache = self._should_cache_block(
            target_block_id, total_blocks, curr_cp_stride, next_cp_stride
        )
        
        if should_cache:
            self._log_debug(f"Storing block {target_block_id}")
            self.cache.store_block(target_block_id, fresh_state.out, fresh_state.lse)
        
        # Move to next target block
        target_block_id = (target_block_id + 1) % total_blocks
        
        # Reset fresh state
        fresh_state = AttentionState()
        
        return final_state, fresh_state, target_block_id
    
    def _reverse_ulysses_transform(
        self,
        state: AttentionState,
        ulysses_size: int,
        seq_dim: int,
        head_dim: int,
    ) -> AttentionState:
        """Reverse Ulysses all-to-all transformation."""
        ulysses_group = get_up_group(ulysses_size)
        
        state.lse = squeeze_and_transpose(state.lse)
        state.out = ulysses_group.all_to_all(state.out, seq_dim, head_dim)
        state.lse = ulysses_group.all_to_all(state.lse, head_dim, seq_dim)
        
        return state
    
    def _should_cache_block(
        self,
        target_block_id: int,
        total_blocks: int,
        curr_cp_stride: int,
        next_cp_stride: int,
    ) -> bool:
        """Determine whether to cache the current block."""
        # Don't cache if going to full CP or computing local chunk in full mode
        if next_cp_stride == self.cp_size:
            return False
        
        if total_blocks == self.cp_size and target_block_id == self.local_chunk_id:
            return False
        
        return True
    
    # -------------------- Finalization --------------------
    
    def _finalize_iteration(
        self,
        curr_cp_stride: int,
        next_cp_stride: int,
        block_info: BlockInfo,
    ):
        """Update target chunk and cache for next iteration."""
        # Update target chunk ID
        if next_cp_stride > curr_cp_stride:
            # Merge case: stay within current block
            stride_offset = self.local_chunk_id % next_cp_stride
            self.target_chunk_id = (self.target_chunk_id // next_cp_stride * 
                                   next_cp_stride + stride_offset)
        else:
            # Normal case: advance to next chunk
            self.target_chunk_id = (self.target_chunk_id + curr_cp_stride) % self.cp_size
        
        # Update cache
        new_target_block_id = self.target_chunk_id // next_cp_stride
        self.cache.merge_and_evict_blocks(next_cp_stride, new_target_block_id)
        self.cache.update_block_age()
    
    # -------------------- Helper Methods --------------------
    
    def _prepare_compute_phase(
        self,
        curr_cp_stride: int,
        next_cp_stride: int,
    ) -> RingConfig:
        """Prepare configuration for compute phase."""
        ring_steps, ulysses_size, ring_steps_to_update = self._get_ring_and_ulysses_steps(
            curr_cp_stride, next_cp_stride
        )
        
        self.cache.adjust_cache_shape(
            new_cp_stride=next_cp_stride,
            num_chunks_per_block=ulysses_size
        )
        
        block_info = self._get_block_info()
        ring_offset = self.local_chunk_id // curr_cp_stride * curr_cp_stride
        ring_next_rank = (self.local_chunk_id - ulysses_size) % curr_cp_stride + ring_offset
        ring_prev_rank = (self.local_chunk_id + ulysses_size) % curr_cp_stride + ring_offset
        
        return RingConfig(
            ring_steps=ring_steps,
            ulysses_size=ulysses_size,
            ring_steps_to_update=ring_steps_to_update,
            ring_prev_rank=ring_prev_rank,
            ring_next_rank=ring_next_rank,
        )
    
    def _get_block_info(self) -> BlockInfo:
        """Get current block information."""
        curr_block_num = self.cache.curr_block_num
        chunks_per_block = self.cp_size // curr_block_num
        local_block_id = self.local_chunk_id // chunks_per_block
        target_block_id = self.target_chunk_id // chunks_per_block
        
        return BlockInfo(
            total_blocks=curr_block_num,
            local_block_id=local_block_id,
            target_block_id=target_block_id,
        )
    
    def _get_ring_and_ulysses_steps(
        self,
        curr_cp_stride: int,
        next_cp_stride: int,
    ) -> Tuple[int, int, int]:
        """Calculate ring steps and Ulysses size."""
        if curr_cp_stride == self.cp_size:
            # Multi-block caching mode
            ulysses_size = min(next_cp_stride, self.ulysses_limit)
            ring_steps = curr_cp_stride // ulysses_size
            ring_steps_to_update = 1
        else:
            # Single-block caching mode
            if curr_cp_stride <= self.ulysses_limit:
                ulysses_size = curr_cp_stride
            else:
                # Find largest power-of-2 factor within limit
                ulysses_size = curr_cp_stride & -curr_cp_stride
                ulysses_size = min(self.ulysses_limit, ulysses_size)
            
            ring_steps = curr_cp_stride // ulysses_size
            ring_steps_to_update = ring_steps
        
        return ring_steps, ulysses_size, ring_steps_to_update
    
    def _get_current_and_next_stride(self) -> Tuple[int, int]:
        """
        Get current and next CP stride.
        
        TODO: Implement adaptive stride selection strategy.
        Current: Fixed ratio except first step.
        """
        curr_cp_stride = self.cp_size // 2 if get_timestep() > 0 else self.cp_size
        next_cp_stride = self.cp_size // 2
        return curr_cp_stride, next_cp_stride
    
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
    
    # -------------------- Communication --------------------
    
    def _async_ring_p2p_commit(
        self,
        tensors: Tuple[torch.Tensor, ...],
        src_rank: int,
        dst_rank: int,
    ) -> Tuple[torch.Tensor, ...]:
        """Start asynchronous ring P2P communication."""
        recv_tensors = []
        
        for tensor in tensors:
            if tensor is None:
                recv_tensors.append(None)
                continue
            
            self.cp_group.p2p_isend(tensor, dst=dst_rank)
            recv_tensor = self.cp_group.p2p_irecv(
                size=tensor.shape,
                dtype=tensor.dtype,
                src=src_rank
            )
            recv_tensors.append(recv_tensor)
        
        self.cp_group.p2p_commit()
        return tuple(recv_tensors)
    
    def _async_ring_p2p_wait(
        self,
        recv_tensors: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Wait for asynchronous ring P2P communication to complete."""
        self.cp_group.p2p_wait()
        return recv_tensors
    
    # -------------------- Logging --------------------
    
    def _log_debug(self, message: str):
        """Log debug message (layer 0, rank 0 only by default)."""
        if self.global_rank == 0 and self.layer_id == 0:
            logger.info(f"R{self.global_rank}T{get_timestep()}L{self.layer_id} | {message}")