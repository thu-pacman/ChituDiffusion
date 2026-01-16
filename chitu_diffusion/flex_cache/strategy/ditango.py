import torch
import functools
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from logging import getLogger
import random
from chitu_diffusion.backend import DiffusionBackend
from chitu_diffusion.flex_cache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.modules.attention.ditango_v2_attn_backend import Ditangov2Attention



logger = getLogger(__name__)

def get_timestep() -> int:
    """Get current diffusion timestep."""
    return DiffusionBackend.generator.current_task.buffer.current_step


class DiTangoStrategy(FlexCacheStrategy):
    def __init__(self):
        self.error_threshold = 0.1
        self.last_residual_norm = None
        self.accumulated_ratio = 1
        self.accumulated_skip_error = 0.0
        self.step_importance = 3
        self.error_log = []

    @property
    def current_step_importance(self):
        return self.step_importance
    

    def _compute_magnitude_ratio(self, current_residual):
        # 计算幅度比
            
        current_norm = torch.norm(current_residual, dim=-1)
        if self.last_residual_norm is None:
            self.last_residual_norm = current_norm
            return 1
        else:
            magnitude_ratio = current_norm / self.last_residual_norm
            return magnitude_ratio.mean().item()  # 返回平均幅度比

    def _estimate_skip_error(self, current_residual):
        # 估计跳过误差
        magnitude_ratio = self._compute_magnitude_ratio(current_residual)
        logger.info(f"{get_timestep()} | mag ratio = {magnitude_ratio}")
        self.error_log.append({"timestep": get_timestep(), "mag ratio": magnitude_ratio})
        self.accumulated_ratio *= magnitude_ratio
        skip_error = 1.0 - self.accumulated_ratio
        return skip_error
    
    def update_importance(self, current_residual):
        self.accumulated_skip_error += self._estimate_skip_error(current_residual)
        # logger.info(f"{get_timestep()} | estimated error = {self.accumulated_skip_error}")
        self.error_log.append({"timestep": get_timestep(), "estimated_error": self.accumulated_skip_error})

    def save_error_log(self):
        # 将误差日志保存到 CSV 文件
        df = pd.DataFrame(self.error_log)
        log_file = "./error_log.csv"
        df.to_csv(log_file, index=False)
        logger.info(f"Error log saved to {log_file}")

    
    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        # Replace self-attention module with DitangoAttn

        for layer_id, block in enumerate(module.blocks):
            block.self_attn.attn_func = Ditangov2Attention(
                layer_id=layer_id
            )

        # 然后给model compute添加上下文
        if not hasattr(module, '_original_forward'):
            module._original_forward = module.model_compute
            
        @functools.wraps(module._original_forward)
        def forward_with_ditango_scheduler(x: torch.Tensor, **kwargs):
            """
            ditango的online调度器。
            """
            ori_x = x.clone()
            # 调用原始forward的核心逻辑
            output = module._original_forward(x, **kwargs)
            res = output - ori_x
            self.update_importance(res)
            return output

        module.model_compute = forward_with_ditango_scheduler
        logger.info(f"Module {module.__class__.__name__} wrapped with ditango strategy")

    def __del__(self):
        self.save_error_log()

    def get_reuse_key(self, **kwargs):
        return super().get_reuse_key(**kwargs)
    def get_store_key(self, **kwargs):
        return super().get_store_key(**kwargs)
    def reset_state(self):
        return
    def reuse(self, cached_feature, **kwargs):
        return super().reuse(cached_feature, **kwargs)
    def store(self, fresh_feature, **kwargs):
        return super().store(fresh_feature, **kwargs)
    def unwrap_module(self, module):
        return
