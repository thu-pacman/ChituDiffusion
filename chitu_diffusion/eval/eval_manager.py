import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable, Tuple, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import functools
from logging import getLogger
from .utils.distributed import dist_run
logger = getLogger(__name__)

from abc import ABC, abstractmethod
import torch
from typing import Optional, Any

class EvalStrategy(ABC):
    def __init__(self):
        self.type = None
        
    @abstractmethod
    def get_eval_videos(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass



class EvalManager():
    def __init__(self):
        self.strategy: EvalStrategy = None
        self.eval_result={}

    def set_strategy(self, strategy: EvalStrategy):
        self.strategy = strategy
    
    def run(self, args, **kwargs):
        dist_run(self.strategy, args, **kwargs)


