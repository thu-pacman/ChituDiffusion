import torch
from typing import Optional, Tuple
from chitu_core.distributed.parallel_state import get_cp_group, get_up_group

from logging import getLogger
from chitu_diffusion.utils.shared_utils import update_out_and_lse, squeeze_and_transpose

# try:
#     import flash_attn_interface       
#     FLASH_ATTN_3_AVAILABLE = True
# except ModuleNotFoundError:
FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn                   
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


logger = getLogger(__name__)
class DiffusionAttnBackend:
    """
    统一 flash-attn v2/v3 的调用入口。
    优先用 v3，v3 不存在再降级到 v2。
    """

    def __init__(self) -> None:
        if FLASH_ATTN_3_AVAILABLE:
            self.impl = "v3"
        elif FLASH_ATTN_2_AVAILABLE:
            self.impl = "v2"
        else:
            raise RuntimeError("Neither flash-attn v2 nor v3 found.")
        if torch.distributed.get_rank() == 0:
            logger.info(f"Using Flash Attention {self.impl} as Diffusion attention backend.")

    # ------------- 统一入口 -------------
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        return_attn_probs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns
        -------
        out : torch.Tensor
        softmax_lse : torch.Tensor, optional
        attn_probs : torch.Tensor, optional
        """
        use_varlen = cu_seqlens_q is not None

        if self.impl == "v3":
            return self._fwd_v3(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p, softmax_scale,
                causal, window_size,
                deterministic,return_attn_probs,
                use_varlen,
            )
        else:  # v2
            return self._fwd_v2(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p, softmax_scale,
                causal, window_size,
                deterministic, return_attn_probs,
                use_varlen,
            )

    # ------------- v3 分支 -------------
    def _fwd_v3(
        self,
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale,
        causal, window_size, softcap,
        alibi_slopes, deterministic,
        return_attn_probs, block_table,
        use_varlen: bool,
    ):
        # FIXME: support FA3
        pass

    # ------------- v2 分支 -------------
    def _fwd_v2(
        self,
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale,
        causal, window_size,
        deterministic,
        return_attn_probs,
        use_varlen: bool,
    ):
        if use_varlen:
            fn = flash_attn.flash_attn_varlen_func
            kwargs = dict(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
            )
            
        else:
            fn = flash_attn.flash_attn_func
            kwargs = dict(
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
            )
        if return_attn_probs:
            out, lse, s_mask = fn(q, k, v, **kwargs)
        else:
            out = fn(q, k, v, **kwargs)
            lse, s_mask = None, None
       
            
        return out, lse, s_mask

class DiffusionAttention_with_CP:
    def __init__(self, attn, ulysses_limit):
        self.attn = attn
        self.ulysses_limit = ulysses_limit
        self.group = get_cp_group()
        self.cp_size = self.group.group_size
        self.global_rank = torch.distributed.get_rank()
        self.local_chunk_id = self.group.rank_in_group

    
    def async_ring_p2p_commit(self, tensors: Tuple[torch.Tensor, ...], src_rank: int, dst_rank: int):
        """Set up ring communication for sending and receiving tensors asynchronously.
    
        Args:
            tensors: Tuple of tensors to be sent
            dst_rank: Destination rank to send tensors to
            src_rank: Source rank to receive tensors from
            
        Returns:
            Tuple[torch.Tensor, ...]: Tuple of tensors to be received after wait
        """
        recv_tensors = []
        
        for tensor in tensors:
            send_tensor = tensor
            recv_size = send_tensor.shape
            recv_dtype = send_tensor.dtype
            self.group.p2p_isend(send_tensor, dst=dst_rank)
            next_tensor = self.group.p2p_irecv(size=recv_size, dtype=recv_dtype, src=src_rank)
            recv_tensors.append(next_tensor)
            
        self.group.p2p_commit()
        return tuple(recv_tensors)
    
    def async_ring_p2p_wait_and_update(self, recv_tensors: Tuple[torch.Tensor, ...]):
        """Wait for asynchronous communication to complete and return received tensors.
    
        Args:
            recv_tensors: Tuple of tensors returned from async_ring_p2p_commit
            
        Returns:
            Tuple[torch.Tensor, ...]: Tuple of received tensors after communication completes
        """
        self.group.p2p_wait()
        return recv_tensors

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
    ):
        '''
        Flash attention with context parallelism support.
        Input Q,K,V shape: 
        * flash attn func: (B, Seq_len, nheads, headdim)
        * varlen flash attn func: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        Will choose between flash_attn_func and flash_attn_varlen_func based on inputs
        '''
        cp_size = self.cp_size
        local_rank = self.local_chunk_id

        ulysses_size = cp_size if cp_size <= self.ulysses_limit else self.ulysses_limit
        if ulysses_size > 1:
            ulysses_group = get_up_group(ulysses_size)
        ring_steps = cp_size // ulysses_size

        use_varlen = cu_seqlens_q is not None and cu_seqlens_k is not None and \
            max_seqlen_q is not None and max_seqlen_k is not None

        if use_varlen:
            seq_dim = 0
            head_dim = 1
            max_seqlen_q *= ulysses_size
            max_seqlen_k *= ulysses_size
        else:
            seq_dim = 1
            head_dim=2

        ring_next_rank = (local_rank - ulysses_size) % cp_size
        ring_prev_rank = (local_rank + ulysses_size) % cp_size
        out, lse = None, None
        fresh_out, fresh_lse = None, None
        
        if ulysses_size > 1:
            q = ulysses_group.all_to_all(q, head_dim, seq_dim)  
            k = ulysses_group.all_to_all(k, head_dim, seq_dim)
            v = ulysses_group.all_to_all(v, head_dim, seq_dim)
        # for tensor in [q,k,v]:
        #     assert not torch.isnan(tensor).any(), f"NaN detected in qkv: {tensor}"

        
        for ring_step in range(ring_steps):
            if ring_step + 1 != ring_steps:
                if use_varlen:
                    data_pack = (k, v, cu_seqlens_k)
                else:
                    data_pack = (k, v)

                nxt_data_pack = self.async_ring_p2p_commit(
                    data_pack,
                    src_rank=ring_prev_rank,
                    dst_rank=ring_next_rank,
                )
            
            block_out, block_lse, _ = self.attn(
                q, k, v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                return_attn_probs=True,
            )
            # assert not torch.isnan(block_out).any(), f"NaN detected in blockout: {block_out}"


            fresh_out, fresh_lse = update_out_and_lse(fresh_out, fresh_lse, block_out, block_lse)
            # for tensor in [fresh_out, fresh_lse]:
            #     assert not torch.isnan(tensor).any(), f"NaN detected in fresh out lse: {tensor}"

            if ring_step + 1 != ring_steps:
                if use_varlen:
                    k, v, cu_seqlens_k = self.async_ring_p2p_wait_and_update(nxt_data_pack)
                else:
                    k, v = self.async_ring_p2p_wait_and_update(nxt_data_pack)
            
            fresh_lse = squeeze_and_transpose(fresh_lse)

            if ulysses_size > 1:
                fresh_out = ulysses_group.all_to_all(fresh_out, seq_dim, head_dim)
                fresh_lse = ulysses_group.all_to_all(fresh_lse, head_dim, seq_dim)

            out, lse = update_out_and_lse(out, lse, fresh_out, fresh_lse)
            fresh_out, fresh_lse = None, None

        # assert not torch.isnan(out).any(), f"NaN detected in only output"

        return out, None, None
            
