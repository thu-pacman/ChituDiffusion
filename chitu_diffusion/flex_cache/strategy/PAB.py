import torch
import numpy as np
from typing import Optional, Any, Dict
import torch.distributed as dist
import functools
from logging import getLogger
from chitu_diffusion.flex_cache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.task import DiffusionTask
from chitu_diffusion.backend import DiffusionBackend, CFGType
from chitu_core.distributed.parallel_state import get_cp_group

logger = getLogger(__name__)
is_main_process = dist.get_rank() == 0


class PABStrategy(FlexCacheStrategy):
    """
    PAB (Pyramid Attention Broadcast) 策略实现
    
    核心思路：
    - 在指定的 step 范围内，按固定间隔复用 attention 的输出
    - 分别为 self-attention 和 cross-attention 设置不同的复用间隔
    - 分别为条件/非条件分支维护独立缓存
    """
    
    def __init__(
        self,
        task: DiffusionTask,
        # PAB 专用参数
        warmup_steps: int = None,
        cooldown_steps: int = None,
        skip_self_range: int = None,
        skip_cross_range: int = None
    ):
        """
        Args:
            warmup_steps: 开始PAB的step
            cooldown_steps: 结束PAB的step
            skip_self_range: self_attn复用的间隔
            skip_cross_range: cross_attn复用的间隔
        """
        super().__init__()
        self.type = 'PAB'
        
        # 获取并打印CP信息（所有rank都打印）
        cp_group = get_cp_group()
        cp_rank = cp_group.rank_in_group if cp_group.group_size > 1 else 0
        cp_size = cp_group.group_size
        global_rank = dist.get_rank()
        
       
        print(f"[PAB Init] Global Rank {global_rank}: cp_size={cp_size}, cp_rank={cp_rank}, "
              f"rank_list={cp_group.rank_list}")
        logger.info(f"[PAB Init] Global Rank {global_rank}: cp_size={cp_size}, cp_rank={cp_rank}")
        
        # 步数管理
        self.num_steps = task.req.params.num_inference_steps
        self.self_broadcast = True 
        self.cross_broadcast = True # 固定使用self和cross都broadcast
          
        # 自动设置参数（会设置 warmup_steps, cooldown_steps, skip_self_range 等）
        self._setup_PAB(warmup_steps, cooldown_steps, skip_self_range, skip_cross_range)
        
        # 在参数设置后计算 tradeoff_score
        self.tradeoff_score = (self.cooldown_steps - self.warmup_steps) / self.skip_self_range
        
    def _setup_PAB(
        self, 
        warmup_steps: int = None,
        cooldown_steps: int = None,
        skip_self_range: int = None,
        skip_cross_range: int = None
    ):
        """
        根据模型类型和任务类型自动配置PAB参数
        
        Args:
            warmup_steps: 自定义开始step，若为None则自动选择
            cooldown_steps: 自定义结束step，若为None则自动设置
            skip_self_range: 自定义self_attn复用的间隔，若为None则自动设置
            skip_cross_range: 自定义cross_attn复用的间隔，若为None则自动设置
        """
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        else:
            self.warmup_steps = 5
        if cooldown_steps is not None:
            self.cooldown_steps = cooldown_steps
        else:
            self.cooldown_steps = self.num_steps - 5
        if skip_self_range is not None:
            self.skip_self_range = skip_self_range
        else:
            self.skip_self_range = 2
        if skip_cross_range is not None:
            self.skip_cross_range = skip_cross_range
        else:
            self.skip_cross_range = 3
        
        model_name = DiffusionBackend.args.models.name
        logger.info(f"[PAB setup] model={model_name}, "
                   f"warmup_steps={self.warmup_steps}, cooldown_steps={self.cooldown_steps}, "
                   f"skip_self_range={self.skip_self_range}, skip_cross_range={self.skip_cross_range}")
        
        
    def get_reuse_key(self, range) -> Optional[str]:
        """
        判断是否可以复用缓存
        
        Args:
            range: 复用间隔
        Returns:
            缓存键 'neg_cpX' 或 'pos_cpX'，若不可复用则返回 None
        """
        # 从后端获取必要的信息
        is_pos = DiffusionBackend.cfg_type == CFGType.POS
        current_step = DiffusionBackend.generator.current_task.buffer.current_step

        # 获取CP rank信息（如果有CP并行）
        cp_group = get_cp_group()
        cp_rank = cp_group.rank_in_group if cp_group.group_size > 1 else 0
        
        branch_key = f"{'pos' if is_pos else 'neg'}_cp{cp_rank}"
        
        # 在指定范围外，不使用缓存
        if current_step < self.warmup_steps or current_step >= self.cooldown_steps:
            return None
        elif (current_step - self.warmup_steps) % range == 0:
            return None  # 该步需要重新计算
        else: 
            return branch_key  # 可以复用
        
    
    def reuse(self, cached_feature: torch.Tensor, 
              **kwargs) -> torch.Tensor:
        """
        复用缓存的 attention output
        
        Args:
            cached_feature: 缓存的attn_output
            **kwargs: 其他参数
            
        Returns:
            缓存的attention output
        """
        return cached_feature
    
    def get_store_key(self,  **kwargs) -> Optional[str]:
        """
        判断是否需要存储特征
        
        Args:
            **kwargs: 其他参数
            
        Returns:
            存储键 'pos_cpX' 或 'neg_cpX'
        """
        is_pos = DiffusionBackend.cfg_type == CFGType.POS
        
        # 获取CP rank信息（如果有CP并行）
        cp_group = get_cp_group()
        cp_rank = cp_group.rank_in_group if cp_group.group_size > 1 else 0
        
        return f"{'pos' if is_pos else 'neg'}_cp{cp_rank}"
    
    def store(self, fresh_feature: torch.Tensor,
              **kwargs) -> torch.Tensor:
        """
        返回 attention output
        
        Args:
            fresh_feature: 新计算的输出特征
            **kwargs: 其他参数
            
        Returns:
            attn output
        """
        return fresh_feature
    
    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        """
        使用PAB策略包装DiT模块的所有 attention blocks
        
        Args:
            module: 要包装的PyTorch模块（DiT model）
        """
        # 遍历所有 blocks，分别包装 self-attention 和 cross-attention
        for block_idx, block in enumerate(module.blocks):
            # 包装 self-attention
            if not hasattr(block.self_attn, '_original_forward'): 
                block.self_attn._original_forward = block.self_attn.forward
            
            # 显式捕获原始函数，避免闭包引用最后一个 block
            original_self_attn = block.self_attn._original_forward
            
            @functools.wraps(original_self_attn)
            def self_attn_forward_with_pab(*args, block_idx=block_idx, original_fn=original_self_attn, **kwargs):
                """
                带PAB缓存的 self-attention forward
                """
                reuse_key = self.get_reuse_key(range=self.skip_self_range)
                
                # 构建完整缓存键
                if reuse_key is not None:
                    cache_key = f"{reuse_key}_block{block_idx}_self"
                    
                    if cache_key in DiffusionBackend.flexcache.cache:
                        cached_output = DiffusionBackend.flexcache.cache[cache_key]
                        # 所有rank都打印读取信息（只在block 0）
                        # if block_idx == 0:
                        #     print(f"[Rank {dist.get_rank()}] REUSE self-attn from key: {cache_key}, shape: {cached_output.shape}")
                        return self.reuse(cached_feature=cached_output)
                
                # 完整计算 - 使用捕获的原始函数
                output = original_fn(*args, **kwargs)
                
                # 存储缓存
                store_key = self.get_store_key()
                if store_key is not None:
                    cache_key = f"{store_key}_block{block_idx}_self"
                    DiffusionBackend.flexcache.cache[cache_key] = output
                    # 所有rank都打印存储信息（只在block 0）
                    # if block_idx == 0:
                    #     print(f"[Rank {dist.get_rank()}] STORE self-attn to key: {cache_key}, shape: {output.shape}")
                
                return output
            
            block.self_attn.forward = self_attn_forward_with_pab
            
            # 包装 cross-attention
            if not hasattr(block.cross_attn, '_original_forward'):
                block.cross_attn._original_forward = block.cross_attn.forward
            
            # 显式捕获原始函数，避免闭包引用最后一个 block
            original_cross_attn = block.cross_attn._original_forward
            
            @functools.wraps(original_cross_attn)
            def cross_attn_forward_with_pab(*args, block_idx=block_idx, original_fn=original_cross_attn, **kwargs):
                """
                带PAB缓存的 cross-attention forward
                """
                reuse_key = self.get_reuse_key(range=self.skip_cross_range)
                
                # 构建完整缓存键
                if reuse_key is not None:
                    cache_key = f"{reuse_key}_block{block_idx}_cross"
                    
                    if cache_key in DiffusionBackend.flexcache.cache:
                        cached_output = DiffusionBackend.flexcache.cache[cache_key]
                        # 所有rank都打印读取信息（只在block 0）
                        # if block_idx == 0:
                        #     print(f"[Rank {dist.get_rank()}] REUSE cross-attn from key: {cache_key}, shape: {cached_output.shape}")
                        return self.reuse(cached_feature=cached_output)
                
                # 完整计算 - 使用捕获的原始函数
                output = original_fn(*args, **kwargs)
                
                # 存储缓存
                store_key = self.get_store_key()
                if store_key is not None:
                    cache_key = f"{store_key}_block{block_idx}_cross"
                    DiffusionBackend.flexcache.cache[cache_key] = output
                    # 所有rank都打印存储信息（只在block 0）
                    # if block_idx == 0:
                    #     print(f"[Rank {dist.get_rank()}] STORE cross-attn to key: {cache_key}, shape: {output.shape}")
                
                return output
            
            block.cross_attn.forward = cross_attn_forward_with_pab
        
        logger.info(f"Module {module.__class__.__name__} wrapped with PAB strategy")
    
    def unwrap_module(self, module: torch.nn.Module) -> None:
        """
        恢复模块的原始forward方法
        
        Args:
            module: 要恢复的PyTorch模块
        """
        # 遍历所有 blocks，恢复 attention 的原始 forward
        for block in module.blocks:
            # 恢复 self-attention
            if hasattr(block.self_attn, '_original_forward'):
                block.self_attn.forward = block.self_attn._original_forward
                delattr(block.self_attn, '_original_forward')
            
            # 恢复 cross-attention
            if hasattr(block.cross_attn, '_original_forward'):
                block.cross_attn.forward = block.cross_attn._original_forward
                delattr(block.cross_attn, '_original_forward')
        
        logger.info(f"Module {module.__class__.__name__} unwrapped from PAB strategy")
    
    def reset_state(self):
        """重置所有内部状态"""
        DiffusionBackend.flexcache.cache.clear() 
        logger.debug("PAB state reset")

