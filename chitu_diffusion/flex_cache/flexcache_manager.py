import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable, Tuple, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import functools
from logging import getLogger

logger = getLogger(__name__)

from abc import ABC, abstractmethod
import torch
from typing import Optional, Any

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
    
    def set_strategy(self, strategy: FlexCacheStrategy):
        self.strategy = strategy
        self.strategy.reset_state()
    
