import os
import torch
from typing import Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod
from chitu_diffusion.core.logging_utils import should_log_on_rank
from chitu_diffusion.runtime.output_layout import append_json_list_item, memory_metrics_dir


class FlexCacheStrategy(ABC):
    """
    Abstract base class for cache strategies.
    每个task维护一个strategy实例，用于控制特征复用的行为。
    
    Attributes:
        type: str, 缓存策略类型(如'teacache'/'pab'等)，由用户针对每个task设置
        tradeoff_score: float, latency和quality之间的权衡分数，由用户针对每个task设置
    """
    def __init__(self):
        self.type = None  
        self.tradeoff_score = None
        
    @abstractmethod
    def get_reuse_key(self, **kwargs) -> Optional[Any]:
        """
        判断当前是否可以复用缓存特征。
        
        Args:
            **kwargs: 策略所需的参数，如输入特征、计算状态等

        Returns:
            Optional[Any]: 若可复用则返回缓存键值，否则返回None
                         键值可以是任意类型，由具体策略定义
        """
        pass
    
    @abstractmethod 
    def reuse(self, **kwargs) ->Any:
        """
        复用缓存特征的具体策略。
        
        Args:
            cached_feature: 从缓存中获取的特征
            **kwargs: 其他需要的参数，如当前特征、模型状态等
            
        Returns:
            torch.Tensor: 处理后的特征。不同策略有不同处理方式：
                         - PAB策略直接返回cached_feature
                         - TeaCache策略会加入残差项
        """
        pass
    
    @abstractmethod
    def get_store_key(self, **kwargs) -> Optional[Any]:
        """
        判断当前特征是否需要存入缓存。
        
        Args:
            **kwargs: 策略所需的参数，如特征、计算状态等
            
        Returns:
            Optional[Any]: 若需要存储则返回存储用的键值，否则返回None
                         键值可以是任意类型，由具体策略定义
        """
        pass
    
    @abstractmethod
    def store(self, **kwargs) -> None:
        """
        将特征存入缓存的预处理逻辑。
        
        Args:
            fresh_feature: 需要存储的特征
            **kwargs: 其他所需参数，如存储条件、处理选项等
        """
        pass

    @abstractmethod
    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        """
        使用缓存策略包装模块
        
        Args:
            module: 要包装的PyTorch模块
        """
        pass

    @abstractmethod
    def unwrap_module(self, module: torch.nn.Module) -> None:
        """
        恢复模块的原始forward方法
        
        Args:
            module: 要恢复的PyTorch模块
        """
        pass

    @abstractmethod
    def reset_state():
        """
        开始新任务，重置所有状态量。
        """
        pass
        
        
class FlexCacheManager():
    """
    目前没有多任务多负载，因此只考虑单task场景，比较naive。
    维护统一的cache buffer，避免内存使用超出了限制
    """
    def __init__(self, max_cache_memory: float):
        self.max_cache_memory = max_cache_memory # 目前仍未支持动态内存调整技术
        self.current_timestep: int = 0 # 每次denoise step更新会更新次timestep
        self.strategy: FlexCacheStrategy = None
        self.cache = {} # 目前采用字典，但应该有更优的数据结构
        self.peak_cache_bytes = 0
        self.peak_cache_entries = 0
        self.peak_cache_tensors = 0
    
    def set_strategy(self, strategy: FlexCacheStrategy):
        self.strategy = strategy
        self.strategy.reset_state()

    def reset_cache_stats(self):
        self.peak_cache_bytes = 0
        self.peak_cache_entries = 0
        self.peak_cache_tensors = 0

    def clear_cache(self):
        self.cache.clear()
        self.reset_cache_stats()

    def cache_tensor_stats(self) -> Dict[str, Any]:
        tensor_count = 0
        total_bytes = 0
        tensors: List[Dict[str, Any]] = []
        tensor_summary: Dict[Tuple[Tuple[int, ...], str, str], Dict[str, Any]] = {}

        def visit(value, path: str):
            nonlocal tensor_count, total_bytes
            if isinstance(value, torch.Tensor):
                nbytes = int(value.numel() * value.element_size())
                shape = tuple(int(dim) for dim in value.shape)
                dtype = str(value.dtype)
                device = str(value.device)
                tensor_count += 1
                total_bytes += nbytes
                summary_key = (shape, dtype, device)
                summary = tensor_summary.setdefault(
                    summary_key,
                    {
                        "shape": list(shape),
                        "dtype": dtype,
                        "device": device,
                        "count": 0,
                        "bytes": 0,
                    },
                )
                summary["count"] += 1
                summary["bytes"] += nbytes
                tensors.append(
                    {
                        "path": path,
                        "shape": list(shape),
                        "dtype": dtype,
                        "device": device,
                        "numel": int(value.numel()),
                        "element_size": int(value.element_size()),
                        "bytes": nbytes,
                    }
                )
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, f"{path}.{key}")
                return
            if isinstance(value, (list, tuple, set)):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")

        visit(self.cache, "cache")
        return {
            "entries": self.cache_entry_count(),
            "tensors": tensor_count,
            "bytes": total_bytes,
            "tensor_summary": list(tensor_summary.values()),
            "tensor_details": tensors,
        }

    def cache_memory_bytes(self) -> int:
        return int(self.cache_tensor_stats()["bytes"])

    def update_peak_cache_memory(self) -> Tuple[bool, Dict[str, Any]]:
        stats = self.cache_tensor_stats()
        increased = int(stats["bytes"]) > self.peak_cache_bytes
        if increased:
            self.peak_cache_bytes = int(stats["bytes"])
            self.peak_cache_entries = int(stats["entries"])
            self.peak_cache_tensors = int(stats["tensors"])
        stats.update(
            {
                "peak_bytes": self.peak_cache_bytes,
                "peak_entries": self.peak_cache_entries,
                "peak_tensors": self.peak_cache_tensors,
                "peak_increased": increased,
            }
        )
        return increased, stats

    def cache_entry_count(self) -> int:
        return len(self.cache)

    def record_cache_memory(self, stage: str, task_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        _, cache_stats = self.update_peak_cache_memory()

        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else int(os.getenv("RANK", "0"))
        if not torch.cuda.is_available():
            return
        if not should_log_on_rank(rank):
            return

        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if not run_output_dir:
            return

        payload = {
            "stage": stage,
            "rank": rank,
            "flexcache_strategy": getattr(self.strategy, "type", None),
            "flexcache_cache_entries": cache_stats["entries"],
            "flexcache_cache_tensors": cache_stats["tensors"],
            "flexcache_cache_bytes": cache_stats["bytes"],
            "flexcache_cache_gb": cache_stats["bytes"] / 1024**3,
            "flexcache_peak_cache_entries": cache_stats["peak_entries"],
            "flexcache_peak_cache_tensors": cache_stats["peak_tensors"],
            "flexcache_peak_cache_bytes": cache_stats["peak_bytes"],
            "flexcache_peak_cache_gb": cache_stats["peak_bytes"] / 1024**3,
            "flexcache_peak_increased": cache_stats["peak_increased"],
            "flexcache_cache_tensor_summary": cache_stats["tensor_summary"],
            "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }
        if task_id is not None:
            payload["task_id"] = task_id
        if extra:
            payload.update(extra)

        append_json_list_item(
            os.path.join(memory_metrics_dir(run_output_dir), f"rank{rank}.json"),
            "events",
            payload,
            base_payload={"rank": rank},
        )
    
