import torch
import numpy as np
from typing import Optional, Any, Dict
import torch.distributed as dist
import functools
from logging import getLogger
from chitu_diffusion.task import DiffusionTask
from chitu_diffusion.backend import DiffusionBackend, CFGType
from chitu_diffusion.flex_cache.flexcache_manager import FlexCacheStrategy
from chitu_core.logging_utils import should_log_info_on_rank

logger = getLogger(__name__)


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


class TeaCacheStrategy(FlexCacheStrategy):
    """
    TeaCache策略实现 - 基于时间步嵌入变化的自适应特征缓存
    
    核心思路：
    - 追踪时间步嵌入(e0/e)的相对L1距离变化
    - 当累积变化小于阈值时复用缓存的残差
    - 分别为条件/非条件分支维护独立缓存
    """
    
    def __init__(
        self,
        task: DiffusionTask,
        # teacache 专用参数
        teacache_thresh: float = 0.2,
        coefficients: list = None,
        warmup_steps: int = None,
        cooldown_steps: int = None
    ):
        """
        Args:
            num_steps: 总采样步数 * 2 (因为CFG需要条件+非条件两次forward)
            teacache_thresh: 缓存复用阈值，越高速度越快但质量可能下降
            coefficients: 距离缩放多项式系数 (可选，自动根据模型类型设置)
            warmup_steps: 前期保留steps数量 (可选，自动设置)
            cooldown_steps: 截止步数 (可选，自动设置)
        """
        super().__init__()
        self.type = 'teacache'
        self.tradeoff_score = teacache_thresh  # 使用thresh作为tradeoff指标
        
        # 步数管理
        self.num_steps = task.req.params.num_inference_steps
        
        # 阈值和缩放
        self.teacache_thresh = teacache_thresh
        self.use_ref_steps = True # 固定使用retention steps策略
        
        # 条件分支(pos)状态
        self.accumulated_rel_l1_distance_pos = 0.0
        self.previous_e0_pos = None
        
        # 非条件分支(neg)状态  
        self.accumulated_rel_l1_distance_neg = 0.0
        self.previous_e0_neg = None
        
        # 自动设置参数
        self._setup_teacache(coefficients, warmup_steps, cooldown_steps)
        
    def _setup_teacache(
        self, 
        thresh: Optional[float] = None,
        coefficients: Optional[list] = None,
        warmup_steps: Optional[int] = None,
        cooldown_steps: Optional[int] = None
    ):
        """
        根据模型类型和任务类型自动配置TeaCache参数
        
        Args:
            coefficients: 自定义系数，若为None则自动选择
            warmup_steps: 自定义retention steps，若为None则自动设置
            cooldown_steps: 自定义cutoff steps，若为None则自动设置
        """
        if thresh is not None:
            self.teacache_thresh = thresh
        else:
            # TODO: 测试，并找到最优参数
            thresh = 0.3
        # 如果提供了自定义参数，直接使用
        if coefficients is not None:
            self.coefficients = coefficients
            self.warmup_steps = warmup_steps if warmup_steps is not None else 5
            self.cooldown_steps = cooldown_steps if cooldown_steps is not None else self.num_steps
            return
        
        model_name = DiffusionBackend.args.models.name
        
        # 自动选择coefficients
        if self.use_ref_steps:
            if '1.3B' in model_name:
                self.teacache_thresh=0.2
                self.coefficients = [
                    -5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 
                    1.36987616e+01, -4.99875664e-02
                ]
            elif '14B' in model_name:
                if 't2v' in model_name or 't2i' in model_name:
                    self.teacache_thresh=0.08
                    self.coefficients = [
                        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 
                        5.87365115e+01, -3.15583525e-01
                    ]
                elif 'i2v-480P' in model_name:
                    self.coefficients = [
                        2.57151496e+05, -3.54229917e+04, 1.40286849e+03, 
                        -1.35890334e+01, 1.32517977e-01
                    ]
                elif 'i2v-720P' in model_name:
                    self.coefficients = [
                        8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 
                        1.66203073e+01, -4.17769401e-02
                    ]
                else:
                    # 默认使用t2v系数
                    self.coefficients = [
                        -3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 
                        5.87365115e+01, -3.15583525e-01
                    ]
            else:
                # 默认系数
                self.coefficients = [1.0]
            
            self.warmup_steps = 5
            self.cooldown_steps = self.num_steps - 3
            
        else:  # not use_ref_steps: 简便起见，暂时不会走这个分支
            if '1.3B' in model_name:
                self.coefficients = [
                    2.39676752e+03, -1.31110545e+03, 2.01331979e+02, 
                    -8.29855975e+00, 1.37887774e-01
                ]
            elif '14B' in model_name:
                if 't2v' in model_name or 't2i' in model_name:
                    self.coefficients = [
                        -5784.54975374, 5449.50911966, -1811.16591783, 
                        256.27178429, -13.02252404
                    ]
                elif 'i2v-480P' in model_name:
                    self.coefficients = [
                        -3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 
                        5.87348440e+00, -2.01973289e-01
                    ]
                elif 'i2v-720P' in model_name:
                    self.coefficients = [
                        -114.36346466, 65.26524496, -18.82220707, 
                        4.91518089, -0.23412683
                    ]
                else:
                    self.coefficients = [
                        -5784.54975374, 5449.50911966, -1811.16591783, 
                        256.27178429, -13.02252404
                    ]
            else:
                self.coefficients = [1.0]
            
            self.warmup_steps = 1
            self.cooldown_steps = self.num_steps - 2
        
        if should_log_info_on_rank():
            logger.info(f"[TeaCache setup] model={model_name}, "
                        f"use_ref_steps={self.use_ref_steps}, thresh={self.teacache_thresh}, "
                        f"warmup_steps={self.warmup_steps}, cooldown_steps={self.cooldown_steps}")
        
    def get_reuse_key(self, e0: torch.Tensor = None, 
                      **kwargs) -> Optional[str]:
        """
        判断是否可以复用缓存
        
        Args:
            e0: 时间步嵌入(已投影，用于use_ref_steps=True)
            **kwargs: 其他参数
            
        Returns:
            缓存键 'neg' 或 'pos'，若不可复用则返回 None
        """
        # 确定使用哪个嵌入来计算距离
        modulated_inp = e0
        if modulated_inp is None:
            return None
            
        # 从后端获取必要的信息
        is_pos = DiffusionBackend.cfg_type == CFGType.POS
        current_step = DiffusionBackend.generator.current_task.buffer.current_step

        branch_key = 'pos' if is_pos else 'neg'
        
        # 在retention steps或cutoff之后的步骤，不使用缓存
        if current_step < self.warmup_steps or current_step >= self.cooldown_steps:
            return None
            
        # 获取对应分支的状态
        if is_pos:
            previous_e0 = self.previous_e0_pos
            accumulated_distance = self.accumulated_rel_l1_distance_pos
        else:
            previous_e0 = self.previous_e0_neg
            accumulated_distance = self.accumulated_rel_l1_distance_neg
            
        # 首次调用该分支，没有历史数据
        if previous_e0 is None:
            return None
            
        # 计算相对L1距离
        with torch.no_grad():
            abs_diff = (modulated_inp - previous_e0).abs().mean()
            abs_mean = previous_e0.abs().mean()
            
            # 避免除零
            if abs_mean < 1e-8:
                return None
                
            rel_l1_distance = (abs_diff / abs_mean).cpu().item()
            
        # 使用多项式缩放距离
        rescale_func = np.poly1d(self.coefficients)
        scaled_distance = rescale_func(rel_l1_distance)
        
        # 更新累积距离
        new_accumulated = accumulated_distance + scaled_distance
        
        if _is_main_process():
            logger.debug(f"[TeaCache] step={current_step} acc/thresh={new_accumulated:.3f}/{self.teacache_thresh}")

        # 判断是否可以复用
        if new_accumulated < self.teacache_thresh: # 若是，返回key并更新误差
            if is_pos:
                self.accumulated_rel_l1_distance_pos = new_accumulated
            else:
                self.accumulated_rel_l1_distance_neg = new_accumulated
            return branch_key
        else: # 若否，完整计算并重置累积距离
            if is_pos:
                self.accumulated_rel_l1_distance_pos = 0.0
            else:
                self.accumulated_rel_l1_distance_neg = 0.0
            return None
    
    def reuse(self, cached_feature: torch.Tensor, x: torch.Tensor, 
              **kwargs) -> torch.Tensor:
        """
        复用缓存的残差特征
        
        Args:
            cached_feature: 缓存的残差
            x: 当前输入特征
            **kwargs: 其他参数
            
        Returns:
            加上残差后的特征
        """
        return x + cached_feature
    
    def get_store_key(self, e0: torch.Tensor = None, **kwargs) -> Optional[str]:
        """
        判断是否需要存储特征
        
        Args:
            x: 输入特征
            output: 输出特征
            e0: 时间步嵌入(已投影)
            e: 原始时间步嵌入
            **kwargs: 其他参数
            
        Returns:
            存储键 'pos' 或 'neg'
        """
        # 更新时间步嵌入历史
        is_pos = DiffusionBackend.cfg_type == CFGType.POS

        modulated_inp = e0
        if modulated_inp is not None:
            if is_pos:
                self.previous_e0_pos = modulated_inp.clone().detach()
            else:
                self.previous_e0_neg = modulated_inp.clone().detach()
            
        return 'pos' if is_pos else 'neg'
    
    def store(self, fresh_feature: torch.Tensor, x: torch.Tensor,
              **kwargs) -> torch.Tensor:
        """
        计算并返回残差特征
        
        Args:
            fresh_feature: 新计算的输出特征
            x: 输入特征
            **kwargs: 其他参数
            
        Returns:
            残差特征 (output - input)
        """
        return fresh_feature - x
    
    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        """
        使用TeaCache策略包装DiT模块
        
        Args:
            module: 要包装的PyTorch模块（DiT model）
        """
        if not hasattr(module, '_original_forward'):
            module._original_forward = module.model_compute
            
        @functools.wraps(module._original_forward)
        def model_compute_with_teacache(x: torch.Tensor, **kwargs):
            """
            适配了TeaCache的forward方法
            """

            e0 = kwargs.get('e', None)
            # 检查是否可以复用缓存
            reuse_key = self.get_reuse_key(e0=e0)
            
            if reuse_key is not None and reuse_key in DiffusionBackend.flexcache.cache:
                cached_residual = DiffusionBackend.flexcache.cache[reuse_key]
                if _is_main_process():
                    logger.debug(f"[TeaCache] reuse key={reuse_key} residual_shape={cached_residual.shape}")
                x_with_residual = self.reuse(
                    cached_feature=cached_residual,
                    x=x,
                )
                return x_with_residual
            
            # 原始model compute
            ori_x = x.clone()
            
            # 调用原始forward的核心逻辑
            output = module._original_forward(x, **kwargs)
            
            # 存储缓存
            store_key = self.get_store_key(x=ori_x, output=output, e0=e0)
            if store_key is not None:
                # 需要获取blocks后的x来计算残差
                # 这里简化处理，实际需要在blocks循环中插入逻辑
                residual = self.store(fresh_feature=output, x=ori_x)  # 简化
                DiffusionBackend.flexcache.cache[store_key] = residual
                
            return output
        
        module.model_compute = model_compute_with_teacache
        if should_log_info_on_rank():
            logger.info(f"Module {module.__class__.__name__} wrapped with TeaCache strategy")
    
    def unwrap_module(self, module: torch.nn.Module) -> None:
        """
        恢复模块的原始forward方法
        
        Args:
            module: 要恢复的PyTorch模块
        """
        if hasattr(module, '_original_forward'):
            module.model_compute = module._original_forward
            delattr(module, '_original_forward')
            if should_log_info_on_rank():
                logger.info(f"Module {module.__class__.__name__} unwrapped")
    
    def reset_state(self):
        """重置所有内部状态"""
        self.accumulated_rel_l1_distance_pos = 0.0
        self.accumulated_rel_l1_distance_neg = 0.0
        self.previous_e0_pos = None
        self.previous_e0_neg = None
        self.previous_residual_pos = None
        self.previous_residual_neg = None
        DiffusionBackend.flexcache.cache.clear() 
        logger.debug("TeaCache state reset")
