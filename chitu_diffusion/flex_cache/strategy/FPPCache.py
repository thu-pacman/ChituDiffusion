import torch
from typing import Optional, Any
import torch.distributed as dist
from logging import getLogger
from chitu_diffusion.backend import CFGType
from chitu_diffusion.flex_cache.flexcache_manager import FlexCacheStrategy
from chitu_core.distributed.parallel_state import get_fpp_group
from chitu_diffusion.backend import DiffusionBackend, CFGType

logger = getLogger(__name__)


class FPPCache():
    """

    """
    
    def __init__(
        self,
    ):
        """
        Args:
            warmup_steps: 开始PAB的step
            cooldown_steps: 结束PAB的step
            skip_self_range: self_attn复用的间隔
            skip_cross_range: cross_attn复用的间隔
        """
        super().__init__()
        self.type = 'FPP'
        
        fpp_group = get_fpp_group()
        fpp_rank = fpp_group.rank_in_group if fpp_group.group_size > 1 else 0
        fpp_size = fpp_group.group_size
        global_rank = dist.get_rank()
        print(f"[FPP Init] Global Rank {global_rank}: fpp_size={fpp_size}, fpp_rank={fpp_rank}, "
              f"rank_list={fpp_group.rank_list}")
        logger.info(f"[FPP Init] Global Rank {global_rank}: fpp_size={fpp_size}, fpp_rank={fpp_rank}")

    def get_key(self, layer_index: int, is_pos: Optional[bool] = None) -> Optional[str]:
        is_pos = DiffusionBackend.cfg_type == CFGType.POS if is_pos is None else is_pos

        branch_key = f"{'pos' if is_pos else 'neg'}_l{layer_index}"
        return branch_key
    
    def get_tokens_key(self, is_pos: Optional[bool] = None) -> Optional[str]:
        is_pos = DiffusionBackend.cfg_type == CFGType.POS if is_pos is None else is_pos

        branch_key = f"{'pos' if is_pos else 'neg'}_tokens"
        return branch_key

    def init_layer_stale_kv(self, k: torch.Tensor, v: torch.Tensor, layer_index: int, is_pos: Optional[bool] = None): 
        DiffusionBackend.flexcache.cache[self.get_key(layer_index, is_pos)] = (k, v)

    def init_stale_tokens(self, latents: torch.Tensor, is_pos: Optional[bool] = None):
        DiffusionBackend.flexcache.cache[self.get_tokens_key(is_pos)] = latents
    
    def update_layer_stale_kv_patch(self, k: torch.Tensor, v: torch.Tensor, layer_index: int, patch_range: tuple[int, int], is_pos: Optional[bool] = None):

        k_ref, v_ref = DiffusionBackend.flexcache.cache[self.get_key(layer_index, is_pos)]
        # b, s, n, c
        assert k_ref.shape[0] == 1 and v_ref.shape[0] == 1, "Expected batch size of 1 for stale KV"
        k_ref[:, patch_range[0]:patch_range[1], :, :] = k
        v_ref[:, patch_range[0]:patch_range[1], :, :] = v

        return k_ref, v_ref

    # check how xdit implement this scheduler patch 
    def update_stale_tokens_patch(self, tokens: torch.Tensor, patch_range: tuple[int, int], is_pos: Optional[bool] = None):

        tokens_ref = DiffusionBackend.flexcache.cache[self.get_tokens_key(is_pos)]
        # b, s, c
        assert tokens_ref.shape[0] == 1, "Expected batch size of 1 for stale tokens"
        original_seq_len = patch_range[1] - patch_range[0]
        tokens_ref[:, patch_range[0]:patch_range[1], :] = tokens[:, :original_seq_len, :]

        return tokens_ref
    
    def reset_state(self):
        """重置所有内部状态"""
        DiffusionBackend.flexcache.cache.clear() 
        logger.debug("FPP state reset")
