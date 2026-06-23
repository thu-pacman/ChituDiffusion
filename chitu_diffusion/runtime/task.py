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
from typing import Any, Optional, Union, Dict, List, Deque
from pathlib import Path
from collections import deque

from chitu_diffusion.runtime.backend import DiffusionBackend
from chitu_diffusion.core.distributed.parallel_state import get_cfg_group
from chitu_diffusion.flexcache.params import FLEXCACHE_PARAM_CLASSES, FlexCacheParams
from chitu_diffusion.parallel.state import ParallelTaskState

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
    Cancel = 6


class DiffusionTaskStatus(Enum):
    Pending = 1     # 任务创建，等待执行
    Running = 2     # 任务执行中
    Completed = 3   # 任务完成
    Failed = 4      # 任务失败


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
    # Acceleration compatibility field: only specify a strategy name.
    flexcache: Optional[str] = None
    # Unified acceleration parameter object.
    flexcache_params: Optional[Union[FlexCacheParams, Dict[str, Any]]] = None

    def __post_init__(self):
        if isinstance(self.flexcache_params, dict):
            strategy = (self.flexcache_params.get("strategy") or "").strip().lower()
            cls = FLEXCACHE_PARAM_CLASSES.get(strategy)
            if cls is None:
                raise ValueError(f"Unsupported acceleration strategy '{self.flexcache_params.get('strategy')}'.")
            self.flexcache_params = cls(**self.flexcache_params)

    def resolve_flexcache_params(self) -> Optional[FlexCacheParams]:
        """
        Unified acceleration parameter entry.
        Prefer flexcache_params, falling back to the legacy flexcache field.
        """
        params = self.flexcache_params

        if params is None:
            strategy = (self.flexcache or "").strip().lower()
            if not strategy:
                return None
            params_cls = FLEXCACHE_PARAM_CLASSES.get(strategy)
            if params_cls is None:
                supported = ", ".join(sorted(FLEXCACHE_PARAM_CLASSES))
                raise ValueError(
                    f"Unsupported acceleration strategy '{strategy}'. "
                    f"Supported strategies are: {supported}."
                )
            params = params_cls()

        strategy = (params.strategy or "").strip().lower()
        if strategy in {"", "none", "off", "disable", "disabled"}:
            return None

        if strategy not in FLEXCACHE_PARAM_CLASSES:
            supported = ", ".join(sorted(FLEXCACHE_PARAM_CLASSES))
            raise ValueError(
                f"Unsupported acceleration strategy '{params.strategy}'. "
                f"Supported strategies are: {supported}."
            )

        warmup = int(params.warmup)
        cooldown = int(params.cooldown)
        if warmup < 0:
            raise ValueError(f"acceleration warmup must be >= 0, got {warmup}.")
        if cooldown < 0:
            raise ValueError(f"acceleration cooldown must be >= 0, got {cooldown}.")

        resolved = params
        resolved.strategy = strategy
        resolved.warmup = warmup
        resolved.cooldown = cooldown

        cache_ratio = getattr(resolved, "cache_ratio", None)
        if cache_ratio is not None:
            cache_ratio = float(cache_ratio)
            if cache_ratio < 0.0 or cache_ratio > 1.0:
                raise ValueError(f"acceleration cache_ratio must be in [0, 1], got {cache_ratio}.")
            resolved.cache_ratio = cache_ratio
        if hasattr(resolved, "tau_max"):
            resolved.tau_max = int(resolved.tau_max)
        if hasattr(resolved, "anchor_interval") and resolved.anchor_interval is not None:
            resolved.anchor_interval = int(resolved.anchor_interval)
            if resolved.anchor_interval < 1:
                raise ValueError(
                    "acceleration anchor_interval must be >= 1, "
                    f"got {resolved.anchor_interval}."
                )
        if hasattr(resolved, "curvature_interval_power"):
            resolved.curvature_interval_power = float(resolved.curvature_interval_power)
        if hasattr(resolved, "locality_group_compute_boost"):
            resolved.locality_group_compute_boost = float(resolved.locality_group_compute_boost)
            if resolved.locality_group_compute_boost < 0.0:
                raise ValueError(
                    "acceleration locality_group_compute_boost must be >= 0, "
                    f"got {resolved.locality_group_compute_boost}."
                )
        for attr in (
            "groupwise_state_align_out_scale",
            "groupwise_state_align_lse_scale",
            "groupwise_state_align_distance_tau",
        ):
            if hasattr(resolved, attr):
                value = float(getattr(resolved, attr))
                if value < 0.0:
                    raise ValueError(f"acceleration {attr} must be >= 0, got {value}.")
                setattr(resolved, attr, value)
        if hasattr(resolved, "groupwise_state_align_mode"):
            resolved.groupwise_state_align_mode = str(resolved.groupwise_state_align_mode)
        if hasattr(resolved, "groupwise_topk_mode"):
            resolved.groupwise_topk_mode = str(resolved.groupwise_topk_mode)
        if hasattr(resolved, "groupwise_fixed_anchor_steps"):
            resolved.groupwise_fixed_anchor_steps = str(resolved.groupwise_fixed_anchor_steps)
        for attr in (
            "groupwise_stagger_period",
            "groupwise_stagger_fresh_count",
            "groupwise_stagger_layer_start",
            "groupwise_force_tail_full_layers",
            "groupwise_local_expand",
            "groupwise_extra_topk",
        ):
            if hasattr(resolved, attr):
                value = int(getattr(resolved, attr))
                if value < 0 and attr != "groupwise_local_expand":
                    raise ValueError(f"acceleration {attr} must be >= 0, got {value}.")
                setattr(resolved, attr, value)
        if hasattr(resolved, "groupwise_stagger_layer_end"):
            resolved.groupwise_stagger_layer_end = int(resolved.groupwise_stagger_layer_end)
        for attr in ("groupwise_keep_local", "groupwise_reuse_stale_kv", "groupwise_state_align"):
            if not hasattr(resolved, attr):
                continue
            raw = getattr(resolved, attr)
            if isinstance(raw, str):
                value = raw.strip().lower() in {"1", "true", "yes", "y", "on"}
            else:
                value = bool(raw)
            setattr(resolved, attr, value)
        if hasattr(resolved, "intra_group_size_limit") and resolved.intra_group_size_limit is not None:
            resolved.intra_group_size_limit = int(resolved.intra_group_size_limit)
            if resolved.intra_group_size_limit < 1:
                raise ValueError(
                    "acceleration intra_group_size_limit must be >= 1, "
                    f"got {resolved.intra_group_size_limit}."
                )
        return resolved
    

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
    text_embeddings_mask: Optional[torch.Tensor] = field(default=None)
    negative_embeddings_mask: Optional[torch.Tensor] = field(default=None)
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

    # FLUX2-specific buffers
    ctx_ids: Optional[torch.Tensor] = field(default=None)
    x_ids: Optional[torch.Tensor] = field(default=None)
    guidance_vec: Optional[torch.Tensor] = field(default=None)
    pooled_prompt_embeds: Optional[torch.Tensor] = field(default=None)
    text_ids: Optional[torch.Tensor] = field(default=None)
    latent_image_ids: Optional[torch.Tensor] = field(default=None)
    image_size: Optional[tuple[int, int]] = field(default=None)
    parallel: ParallelTaskState = field(default_factory=ParallelTaskState)

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

    @classmethod
    def create_cancel_signal(
        cls,
        task_id: str = None,
        reason: str = "Current generation cancelled"
    ) -> 'DiffusionTask':
        if task_id is None:
            task_id = f"cancel_signal_{int(time.time() * 1000)}"

        return cls(
            task_id=task_id,
            task_type=DiffusionTaskType.Cancel,
            req=None,
            buffer=None,
            signal_data={'reason': reason, 'timestamp': time.time()}
        )

    def is_terminate_signal(self) -> bool:
        """检查是否为终止信号"""
        return self.task_type == DiffusionTaskType.Terminate

    def is_cancel_signal(self) -> bool:
        return self.task_type == DiffusionTaskType.Cancel

    def is_control_signal(self) -> bool:
        return self.is_terminate_signal() or self.is_cancel_signal()

    def is_completed(self) -> bool:
        """检查任务是否完成"""
        if self.is_control_signal():
            return True
        return self.status in [DiffusionTaskStatus.Completed, DiffusionTaskStatus.Failed]

    def is_running(self) -> bool:
        """检查任务是否正在运行"""
        if self.is_control_signal():
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
    shutdown_task: DiffusionTask | None = None
    cancel_task: DiffusionTask | None = None

    def __bool__(self):
        return len(self.pool) > 0

    def __len__(self):
        return len(self.pool)

    @classmethod
    def reset(cls):
        cls.pool = {}
        cls.id_list = []
        cls.shutdown_task = None
        cls.cancel_task = None

    @classmethod
    def is_empty(cls):
        return len(cls.pool) == 0 and cls.shutdown_task is None and cls.cancel_task is None

    @classmethod
    def all_finished(cls) -> bool:
        if cls.shutdown_task is not None or cls.cancel_task is not None:
            return False
        if len(cls.pool) == 0:
            return True
        return all(task.is_completed() for task in cls.pool.values())

    @classmethod
    def request_shutdown(cls, reason: str = "Normal shutdown") -> DiffusionTask:
        if cls.shutdown_task is None:
            cls.shutdown_task = DiffusionTask.create_terminate_signal(reason=reason)
        return cls.shutdown_task

    @classmethod
    def request_cancel(cls, reason: str = "Current generation cancelled") -> DiffusionTask:
        if cls.cancel_task is None:
            cls.cancel_task = DiffusionTask.create_cancel_signal(reason=reason)
        return cls.cancel_task

    @classmethod
    def has_shutdown_request(cls) -> bool:
        return cls.shutdown_task is not None and cls.shutdown_task.status == DiffusionTaskStatus.Pending

    @classmethod
    def has_cancel_request(cls) -> bool:
        return cls.cancel_task is not None and cls.cancel_task.status == DiffusionTaskStatus.Pending

    @classmethod
    def get_shutdown_task(cls) -> DiffusionTask | None:
        return cls.shutdown_task

    @classmethod
    def get_cancel_task(cls) -> DiffusionTask | None:
        return cls.cancel_task

    @classmethod
    def get_control_task(cls, task_id: str) -> DiffusionTask | None:
        for task in (cls.shutdown_task, cls.cancel_task):
            if task is not None and task.task_id == task_id:
                return task
        return None

    @classmethod
    def clear_shutdown_request(cls):
        cls.shutdown_task = None

    @classmethod
    def clear_cancel_request(cls):
        cls.cancel_task = None

    @classmethod
    def cancel_active_tasks(cls, reason: str = "Current generation cancelled"):
        for task in cls.pool.values():
            if not task.is_completed():
                task.status = DiffusionTaskStatus.Failed
                task.error_message = reason

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
        
