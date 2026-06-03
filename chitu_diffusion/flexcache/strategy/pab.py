import torch
import numpy as np
import os
from typing import Optional, Any, Dict
import torch.distributed as dist
import functools
from logging import getLogger
from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.flexcache.model_adapters import get_flexcache_adapter
from chitu_diffusion.runtime.task import DiffusionTask
from chitu_diffusion.runtime.backend import DiffusionBackend, CFGType
from chitu_diffusion.runtime.output_layout import debug_output_dir
from chitu_diffusion.core.distributed.parallel_state import get_cp_group
from chitu_diffusion.core.logging_utils import should_log_info_on_rank

logger = getLogger(__name__)


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


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
        
       
        if should_log_info_on_rank():
            logger.info(
                f"[PAB Init] global_rank={global_rank} cp_size={cp_size} "
                f"cp_rank={cp_rank} rank_list={cp_group.rank_list}"
            )
        
        # 步数管理
        self.num_steps = task.req.params.num_inference_steps
        self.self_broadcast = True 
        self.cross_broadcast = True # 固定使用self和cross都broadcast
          
        # 自动设置参数（会设置 warmup_steps, cooldown_steps, skip_self_range 等）
        self._setup_PAB(warmup_steps, cooldown_steps, skip_self_range, skip_cross_range)

        self._vis_records: Dict[int, int] = {}
        self._vis_max_step = -1
        
        # 在参数设置后计算 tradeoff_score
        active_steps = max(1, self.num_steps - self.warmup_steps - self.cooldown_steps)
        self.tradeoff_score = active_steps / self.skip_self_range

    def _get_output_dir(self) -> str:
        env_output = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if env_output:
            return debug_output_dir(env_output)

        task = DiffusionBackend.generator.current_task
        if task is not None and task.req is not None and task.req.params is not None:
            if task.req.params.save_dir:
                return os.path.join(task.req.params.save_dir, "..", "logs")

        args = getattr(DiffusionBackend, "args", None)
        if args is not None:
            output_cfg = getattr(args, "output", None)
            if output_cfg is not None:
                root_dir = getattr(output_cfg, "root_dir", None)
                if root_dir:
                    return str(root_dir)

        return "./outputs"

    def _record_step_policy(self, step: int, code: int):
        # 仅记录 pos 分支，避免 CFG 双分支重复写入
        if DiffusionBackend.cfg_type != CFGType.POS:
            return
        prev = self._vis_records.get(step)
        if prev is None:
            self._vis_records[step] = code
        else:
            # warmup/cooldown(0) < compute(1) < reuse(2)
            self._vis_records[step] = max(prev, code)
        self._vis_max_step = max(self._vis_max_step, step)

    def _save_policy_ppm(self):
        if not _is_main_process() or self._vis_max_step < 0:
            return

        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        cell = 12
        width = (self._vis_max_step + 1) * cell
        height = cell
        rgb = bytearray(width * height * 3)

        for step in range(self._vis_max_step + 1):
            code = self._vis_records.get(step, 0)
            if code == 2:
                color = (255, 180, 40)   # reuse
            elif code == 1:
                color = (40, 140, 255)   # compute
            else:
                color = (160, 160, 160)  # warmup/cooldown

            for yy in range(height):
                for xx in range(step * cell, (step + 1) * cell):
                    idx = (yy * width + xx) * 3
                    rgb[idx] = color[0]
                    rgb[idx + 1] = color[1]
                    rgb[idx + 2] = color[2]

        ppm_path = os.path.join(output_dir, "pab_policy_timestep_pos.ppm")
        with open(ppm_path, "wb") as f:
            header = f"P6\n{width} {height}\n255\n".encode("ascii")
            f.write(header)
            f.write(bytes(rgb))

        logger.info(f"[PAB] Saved policy visualization PPM to {ppm_path}")
        
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
            self.cooldown_steps = 5
        if skip_self_range is not None:
            self.skip_self_range = skip_self_range
        else:
            self.skip_self_range = 2
        if skip_cross_range is not None:
            self.skip_cross_range = skip_cross_range
        else:
            self.skip_cross_range = 3
        
        model_name = DiffusionBackend.args.models.name
        if should_log_info_on_rank():
            logger.info(f"[PAB setup] model={model_name}, "
                        f"warmup_steps={self.warmup_steps}, cooldown_steps={self.cooldown_steps}, "
                        f"skip_self_range={self.skip_self_range}, skip_cross_range={self.skip_cross_range}")
        
        
    def get_reuse_key(self, range, attn_kind: str = "self") -> Optional[str]:
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
        
        # warmup: 前 warmup_steps 步完整计算
        # cooldown: 后 cooldown_steps 步完整计算
        cooldown_start = max(0, self.num_steps - self.cooldown_steps)
        if current_step < self.warmup_steps or current_step >= cooldown_start:
            self._record_step_policy(current_step, 0)
            return None
        elif (current_step - self.warmup_steps) % range == 0:
            self._record_step_policy(current_step, 1)
            return None  # 该步需要重新计算
        else: 
            self._record_step_policy(current_step, 2)
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
        adapter = get_flexcache_adapter(module)
        modules = adapter.attention_modules()
        if not modules:
            raise ValueError(f"PAB strategy found no attention modules on {module.__class__.__name__}.")

        module._pab_wrapped_modules = []

        for block_idx, attn_kind, attn_module in modules:
            if attn_kind not in {"self", "cross"}:
                continue
            if not hasattr(attn_module, '_pab_original_forward'):
                attn_module._pab_original_forward = attn_module.forward
            original_forward = attn_module._pab_original_forward
            module._pab_wrapped_modules.append(attn_module)

            @functools.wraps(original_forward)
            def attn_forward_with_pab(
                *args,
                block_idx=block_idx,
                attn_kind=attn_kind,
                original_fn=original_forward,
                **kwargs,
            ):
                reuse_range = self.skip_self_range if attn_kind == "self" else self.skip_cross_range
                reuse_key = self.get_reuse_key(range=reuse_range, attn_kind=attn_kind)
                scope = f"{attn_kind}_attn"

                if reuse_key is not None:
                    cache_key = f"{reuse_key}_block{block_idx}_{attn_kind}"
                    if cache_key in DiffusionBackend.flexcache.cache:
                        cached_output = DiffusionBackend.flexcache.cache[cache_key]
                        DiffusionBackend.flexcache.record_compute(
                            baseline_units=1.0,
                            actual_units=0.0,
                            task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
                            scope=scope,
                            unit="attention_module",
                            extra={"decision": "reuse", "block_idx": block_idx},
                        )
                        return self.reuse(cached_feature=cached_output)

                output = original_fn(*args, **kwargs)
                DiffusionBackend.flexcache.record_compute(
                    baseline_units=1.0,
                    actual_units=1.0,
                    task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
                    scope=scope,
                    unit="attention_module",
                    extra={"decision": "compute", "block_idx": block_idx},
                )

                store_key = self.get_store_key()
                if store_key is not None:
                    cache_key = f"{store_key}_block{block_idx}_{attn_kind}"
                    DiffusionBackend.flexcache.cache[cache_key] = output
                    DiffusionBackend.flexcache.record_cache_memory(
                        "flexcache_store",
                        task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
                        extra={"cache_key": str(cache_key)},
                    )

                return output

            attn_module.forward = attn_forward_with_pab
        
        logger.info(f"Module {module.__class__.__name__} wrapped with PAB strategy")
    
    def unwrap_module(self, module: torch.nn.Module) -> None:
        """
        恢复模块的原始forward方法
        
        Args:
            module: 要恢复的PyTorch模块
        """
        for attn_module in getattr(module, "_pab_wrapped_modules", []):
            if hasattr(attn_module, '_pab_original_forward'):
                attn_module.forward = attn_module._pab_original_forward
                delattr(attn_module, '_pab_original_forward')
        if hasattr(module, "_pab_wrapped_modules"):
            delattr(module, "_pab_wrapped_modules")

        self._save_policy_ppm()
        DiffusionBackend.flexcache.flush_cache_memory_events()
        
        logger.info(f"Module {module.__class__.__name__} unwrapped from PAB strategy")
    
    def reset_state(self):
        """重置所有内部状态"""
        DiffusionBackend.flexcache.clear_cache()
        logger.debug("PAB state reset")
