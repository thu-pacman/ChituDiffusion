# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import time
import torch
import torch.distributed as dist
import tqdm
import pickle
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from logging import getLogger
from typing import Any, Optional, Union, Dict, List, Deque, Tuple
from pathlib import Path
from collections import deque

from chitu_diffusion.backend import DiffusionBackend
from chitu_core.distributed.parallel_state import get_cfg_group

logger = getLogger(__name__)


import time
import torch
from enum import Enum
from logging import getLogger
from typing import Optional, List
from dataclasses import dataclass

logger = getLogger(__name__)


class DiffusionTaskType(Enum):
    TextEncode = 1
    VAEEncode = 2
    Denoise = 3
    VAEDecode = 4
    Terminate = 5


class DiffusionTaskStatus(Enum):
    Pending = 1     # 任务创建，等待执行
    Running = 2     # 任务执行中
    Completed = 3   # 任务完成
    Failed = 4      # 任务失败


@dataclass
class FlexCacheParams:
    """统一的 FlexCache 用户参数。"""
    strategy: Optional[str] = None
    cache_ratio: float = 0.5
    warmup: int = 5
    cooldown: int = 5


@dataclass
class DiffusionUserParams:
    """Diffusion生成参数"""
    role: str = "user"
    size: tuple[int, int] = (512, 512)
    frame_num: int = 81
    prompt: str = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    # 调度器参数
    sample_solver: str = "ddpm"
    num_inference_steps: int = None
    # 其他参数
    save_dir: Optional[str] = "./output"  # 输出保存路径
    # FlexCache 兼容字段: 仅指定策略名
    flexcache: Optional[str] = None
    # FlexCache 统一参数对象
    flexcache_params: Optional[Union[FlexCacheParams, Dict[str, Any]]] = None

    def __post_init__(self):
        if isinstance(self.flexcache_params, dict):
            self.flexcache_params = FlexCacheParams(**self.flexcache_params)

    def resolve_flexcache_params(self) -> Optional[FlexCacheParams]:
        """
        统一 FlexCache 参数入口。
        优先使用 flexcache_params，缺省时回退到 legacy 字段 flexcache。
        """
        params = self.flexcache_params

        if params is None:
            strategy = (self.flexcache or "").strip()
            if not strategy:
                return None
            params = FlexCacheParams(strategy=strategy)

        strategy = (params.strategy or "").strip().lower()
        if strategy in {"", "none", "off", "disable", "disabled"}:
            return None

        if strategy not in {"teacache", "pab", "ditango"}:
            raise ValueError(
                f"Unsupported flexcache strategy '{params.strategy}'. "
                "Supported strategies are: teacache, pab, ditango."
            )

        cache_ratio = float(params.cache_ratio)
        if cache_ratio < 0.0 or cache_ratio > 1.0:
            raise ValueError(f"flexcache cache_ratio must be in [0, 1], got {cache_ratio}.")

        warmup = int(params.warmup)
        cooldown = int(params.cooldown)
        if warmup < 0:
            raise ValueError(f"flexcache warmup must be >= 0, got {warmup}.")
        if cooldown < 0:
            raise ValueError(f"flexcache cooldown must be >= 0, got {cooldown}.")

        return FlexCacheParams(
            strategy=strategy,
            cache_ratio=cache_ratio,
            warmup=warmup,
            cooldown=cooldown,
        )
    

class DiffusionUserRequest:
    """用户请求封装"""
    
    def __init__(
        self,
        request_id,
        params: DiffusionUserParams = None,
        init_image: Optional[torch.Tensor] = None,  # for i2v
        # txt_emb = None,
        # img_emb = None,
        # latents = None,
        # mask: Optional[torch.Tensor] = None,        # for inpainting
        
    ):
        self.request_id = request_id
        self.params = params
        self.init_image = init_image
        
    def init_user_params(self, params: DiffusionUserParams):
        if params.num_inference_steps is None:
            params.num_inference_steps = DiffusionBackend.args.models.sampler.sample_steps
        elif params.num_inference_steps != DiffusionBackend.args.models.sampler.sample_steps:
            logger.warning(f"Denoising step of request {self.request_id} is {params.num_inference_steps}, but default setting is {DiffusionBackend.args.models.sampler.sample_steps}. Generation quality might be degraded.")

    def get_role(self):
        return self.params.role
    
    def get_prompt(self):
        return self.params.prompt

    def get_n_prompt(self):
        if self.params.negative_prompt is not None:
            return self.params.negative_prompt
        return ""

    def __repr__(self):
        return f"DiffusionUserRequest(id={self.request_id}, request={self.params})"

@dataclass
class DiffusionTaskBuffer:
    """存储扩散任务的缓冲区数据"""
    # Text encode buffers
    text_embeddings: Optional[torch.Tensor] = field(default=None)
    negative_embeddings: Optional[torch.Tensor] = field(default=None)
    seq_len: Optional[int] = field(default=None)
    
    # Denoise buffers
    seed_g: Optional[torch.Generator] = field(default=None)
    sampler: Optional[Any] = field(default=None)
    latents: Optional[torch.Tensor] = field(default=None)
    timesteps: Optional[List[int]] = field(default=None)
    current_step: int = field(default=0)
    denoised_latents: Optional[torch.Tensor] = field(default=None)
    
    # VAE Decode buffers
    generated_image: Optional[torch.Tensor] = field(default=None)

class DiffusionTask:
    
    def __init__(
        self,
        task_id: str,
        task_type: DiffusionTaskType = None,
        req: Optional[DiffusionUserRequest] = None,
        buffer: Optional[DiffusionTaskBuffer] = None,
        signal_data: Optional[Dict] = None, # 系统信号携带的数据

    ):
        logger.debug(f"Create DiffusionTask {task_id}")
        
        # 基本信息
        self.task_id = task_id
        self.task_type = DiffusionTaskType.TextEncode if task_type is None else task_type # T2V task is always text encode
        self.status = DiffusionTaskStatus.Pending
        self.progress_bar = None

        self.req = req
        self.buffer = DiffusionTaskBuffer() if buffer is None else buffer
         # 系统信号数据
        self.signal_data = signal_data or {}
        
        # 错误信息
        self.error_message: Optional[str] = None

    @classmethod
    def create_terminate_signal(
        cls, 
        task_id: str = None,
        reason: str = "Normal shutdown"
    ) -> 'DiffusionTask':
        """
        创建终止信号任务
        
        Args:
            task_id: 任务ID，如果为None则自动生成
            reason: 终止原因
            
        Returns:
            DiffusionTask: 终止信号任务
        """
        if task_id is None:
            task_id = f"terminate_signal_{int(time.time() * 1000)}"
        
        return cls(
            task_id=task_id,
            task_type=DiffusionTaskType.Terminate,
            req=None,  # 终止信号没有用户请求
            buffer=None,  # 终止信号不需要buffer
            signal_data={'reason': reason, 'timestamp': time.time()}
        )

    def is_terminate_signal(self) -> bool:
        """检查是否为终止信号"""
        return self.task_type == DiffusionTaskType.Terminate

    def is_completed(self) -> bool:
        """检查任务是否完成"""
        if self.is_terminate_signal():
            return True
        return self.status in [DiffusionTaskStatus.Completed, DiffusionTaskStatus.Failed]

    def is_running(self) -> bool:
        """检查任务是否正在运行"""
        if self.is_terminate_signal():
            return False
        return self.status == DiffusionTaskStatus.Running

    def __repr__(self):
        return (
            f"DiffusionTask(id={self.task_id}, type={self.task_type}, "
            f"status={self.status}"
        )
    
    # ================= 通信相关 ====================
    def serialize(self, device: str = "cpu") -> torch.Tensor:
        """将DiffusionTask序列化为torch.Tensor"""
        try:
            # 1. 准备基本数据
            serializable_data = {
                'task_id': self.task_id,
                'task_type': self.task_type,
                'status': self.status,
                'error_message': self.error_message,
                'is_terminate_signal': self.is_terminate_signal(),
                'signal_data': self.signal_data,
            }
            
            # 2. 自动序列化用户请求数据
            if self.req is not None:
                # 使用asdict自动转换dataclass
                serializable_data['user_params'] = asdict(self.req.params)
                # 如果有非dataclass字段也需要保存
                serializable_data['request_id'] = self.req.request_id
                serializable_data['init_image_exists'] = self.req.init_image is not None
            else:
                serializable_data['user_params'] = None
            
            # 3. 自动处理Buffer数据
            if self.buffer is not None:
                buffer_dict = {}
                tensor_fields = []
                
                # 遍历buffer的所有字段
                for field_info in fields(self.buffer):
                    field_name = field_info.name
                    field_value = getattr(self.buffer, field_name)
                    
                    # 区分tensor和非tensor字段
                    if isinstance(field_value, torch.Tensor):
                        tensor_fields.append(field_name)
                    else:
                        buffer_dict[field_name] = field_value
                
                serializable_data['buffer_metadata'] = buffer_dict
                serializable_data['tensor_field_names'] = tensor_fields
            else:
                serializable_data['buffer_metadata'] = None
                serializable_data['tensor_field_names'] = []
            
            # 4. 序列化tensor数据
            tensor_data = {}
            if self.buffer is not None:
                for field_name in serializable_data['tensor_field_names']:
                    tensor_value = getattr(self.buffer, field_name)
                    if tensor_value is not None:
                        tensor_data[field_name] = tensor_value.detach().clone()
            
            # 5. 打包所有数据
            full_data = {
                'metadata': serializable_data,
                'tensors': tensor_data
            }
            
            # 6. 使用pickle序列化
            serialized_bytes = pickle.dumps(full_data)
            serialized_array = torch.frombuffer(serialized_bytes, dtype=torch.uint8)
            final_tensor = serialized_array.to(device)
            
            logger.debug(f"Serialized {'terminate signal' if self.is_terminate_signal() else 'task'} {self.task_id}, size: {len(serialized_array)} bytes")
            return final_tensor
            
        except Exception as e:
            logger.error(f"Failed to serialize task {self.task_id}: {e}")
            raise

    @staticmethod
    def deserialize(serialized_tensor: torch.Tensor) -> 'DiffusionTask':
        """从torch.Tensor反序列化DiffusionTask - 自适应版本"""
        try:
            # 1. 移到CPU进行反序列化
            tensor_cpu = serialized_tensor.cpu()
            serialized_bytes = tensor_cpu.byte().numpy().tobytes()
            full_data = pickle.loads(serialized_bytes)
            metadata = full_data['metadata']
            tensor_data = full_data['tensors']
            
            # 2. 重建用户请求对象
            user_request = None
            if metadata['user_params'] is not None:
                # 自动从字典创建DiffusionUserParams
                user_params = DiffusionUserParams(**metadata['user_params'])
                user_request = DiffusionUserRequest(
                    request_id=metadata['request_id'],
                    params=user_params,
                )
            
            # 3. 自动重建buffer
            buffer = None
            if metadata['buffer_metadata'] is not None:
                buffer = DiffusionTaskBuffer()
                
                # 恢复非tensor字段
                for key, value in metadata['buffer_metadata'].items():
                    setattr(buffer, key, value)
                
                # 恢复tensor字段
                for field_name in metadata['tensor_field_names']:
                    if field_name in tensor_data:
                        setattr(buffer, field_name, tensor_data[field_name])
            
            # 4. 重建任务对象
            task = DiffusionTask(
                task_id=metadata['task_id'],
                task_type=metadata['task_type'],
                req=user_request,
                buffer=buffer,
                signal_data=metadata.get('signal_data', {})
            )
            
            task.status = metadata['status']
            task.error_message = metadata['error_message']
            
            logger.debug(f"Deserialized {'terminate signal' if task.is_terminate_signal() else 'task'} {task.task_id}")
            return task
            
        except Exception as e:
            logger.error(f"Failed to deserialize task: {e}")
            raise

    @staticmethod
    def create_empty_serialization(size: int, device: str = "cpu") -> torch.Tensor:
        """创建一个空的序列化张量，用于接收广播数据"""
        empty_tensor = torch.zeros(size, dtype=torch.uint8, device=device)
        logger.info(f"Created {empty_tensor.shape=}")
        return empty_tensor


class DiffusionTaskPool:
    pool: dict[str, DiffusionTask] = {}
    id_list: list[str] = []
    pending_queue: deque[DiffusionTask] = Deque()

    def __bool__(self):
        return len(self.pool) > 0

    def __len__(self):
        return len(self.pool)

    @classmethod
    def reset(cls):
        cls.pool = {}
        cls.id_list = []

    @classmethod
    def is_empty(cls):
        return len(cls.pool) == 0

    @classmethod
    def all_finished(cls) -> bool:
        if len(cls.pool) == 0:
            return True
        return all(task.is_completed() for task in cls.pool.values())

    @classmethod
    def add(cls, task: DiffusionTask):
        if task.task_id in cls.pool:
            return False  # Task already exists, failed to add
        cls.pool[task.task_id] = task
        cls.id_list.append(task.task_id)
        return True

    @classmethod
    def enqueue(cls, task: DiffusionTask):
        cls.pending_queue.append(task)

    @classmethod
    def add_all_queued(cls):
        while cls.pending_queue:
            cls.add(cls.pending_queue.popleft())

    @classmethod
    def remove(cls, task_id: str):
        assert task_id in cls.pool, "Task not found in pool"
        task = cls.pool.pop(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found in pool")
        cls.id_list.remove(task_id)
        del task.buffer
        
