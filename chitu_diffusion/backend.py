# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import gc
import itertools
from functools import partial
import resource
import os
import time
import re
import json
import copy
import torch.cuda.amp as amp

from argparse import Namespace
from collections import deque
from enum import Enum
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Deque, Optional, Iterable, Any
import safetensors.torch as st
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from safetensors.torch import safe_open
from tqdm import tqdm
from chitu_core.distributed.parallel_state import (
    get_world_group,
    initialize_diffusion_parallel_groups,
    get_cp_group
)
from chitu_core.models.registry import ModelType, get_model_class
from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttnBackend, DiffusionAttention_with_CP
from chitu_diffusion.modules.rope.diffusion_rope_backend import naive_rope_apply, rope_apply_with_cp, rope_apply_with_position
from chitu_diffusion.flex_cache.flexcache_manager import FlexCacheManager

# from chitu_core.distributed.moe_token_dispatcher import init_token_dispatcher
if TYPE_CHECKING:
    from chitu_diffusion.generator import Generator
    from chitu_diffusion.scheduler import DiffusionScheduler

logger = getLogger(__name__)

class BackendState(Enum):
    """
    Enumeration of possible backend lifecycle states.
    
    Attributes:
        Running: Backend is actively processing tasks.
        Terminating: All tasks are done, rank 0 should signal others to terminate.
        Terminated: Backend has been fully terminated.
    """
    Running = 1
    Terminating = 2  # All tasks done, but rank 0 should tell others to terminate
    Terminated = 3

class CFGType(Enum):
    """
    Enumeration of Classifier-Free Guidance (CFG) types.
    
    Attributes:
        POS: Process only positive (conditioned) prompts.
        NEG: Process only negative (unconditioned) prompts.
        BOTH: Process both positive and negative prompts.
    """
    POS = "pos"
    NEG = "neg"
    BOTH = "both"
    
class DiffusionBackend:
    """
    Main diffusion model inference backend.
    
    This class manages the lifecycle of diffusion models, including loading,
    parallelism setup, and inference execution. It provides a centralized
    interface for model components (text encoder, VAE, DiT models) and
    handles distributed training configurations.
    
    Class Attributes:
        model_pool (list): Pool of DiT models (for low memory mode, offloaded to CPU).
        formatter: Text formatting utility.
        args: Global configuration arguments.
        cache_type (str): Type of cache management.
        is_main_rank (bool): Whether this is the main process rank.
        use_gloo (bool): Whether to use Gloo backend for communication.
        group_gloo: Gloo communication group.
        scheduler (DiffusionScheduler): Task scheduler instance.
        state (BackendState): Current backend state.
        indexer_cache_manager: Cache manager for indexing.
        do_cfg (bool): Whether to use Classifier-Free Guidance.
        cfg_type (CFGType): Type of CFG to use.
        generator (Generator): Generation executor instance.
        text_encoder: Text encoder model (e.g., T5).
        active_model: Currently active DiT model on GPU.
        active_model_id (int): Index of the active model.
        vae: VAE decoder model.
        boundary: Noise boundary values for multi-stage models.
        guidance_scale: CFG guidance scale values.
        flexcache (FlexCacheManager): Feature reuse cache manager.
    """
    # init once
    model_pool = [] # In low memory mode, DiT models are offloaded to CPU in model pool
    formatter = None
    args = None
    # --- cache_manager related (not used in the current code)
    curr_req_ids = None
    cache_type = ""
    # ---
    is_main_rank = True
    use_gloo = False
    group_gloo = None

    # components
    attn = None
    scheduler: Optional["DiffusionScheduler"] = None

    # mutable
    # ongoing_reqs: list["OngoingRequests"] = []
    state = BackendState.Running
    indexer_cache_manager = None
    
    # diffusion
    do_cfg = True
    cfg_type: Optional[CFGType] = None # pos, neg, both
    generator: Optional["Generator"] = None # 加双引号是正向引用，Generator定义在此段代码之后
    text_encoder = None
    active_model = None
    active_model_id = 0
    vae = None
    boundary = None
    guidance_scale = None
    flexcache: Optional["FlexCacheManager"] = None


    @staticmethod
    def check_and_convert_config(args):
        """
        Convert configuration lists to tuples recursively.
        
        YAML configs can only represent lists, but some places need tuples.
        This method converts all lists in args to tuples recursively, except
        for variables in KEEPLIST which must remain as lists.
        
        Args:
            args: Configuration object to convert.
            
        Returns:
            Configuration object with lists converted to tuples.
            
        Raises:
            AssertionError: If boundary and guidance_scale lengths are inconsistent.
        """ 
        # check validation
        assert len(args.sampler.boundary) == len(args.sampler.guidance_scale) - 1
        KEEP_LIST = {'boundary', 'guidance_scale'}

        # convert config
        def _convert(obj, parent_key=None):
            if isinstance(obj, list):
                if parent_key in KEEP_LIST:
                    return obj
                return tuple(_convert(item) for item in obj)
            if isinstance(obj, dict):
                return {k: _convert(v, parent_key=k) for k, v in obj.items()}
            if hasattr(obj, '__dict__'):
                new_obj = copy.copy(obj)
                for attr_name in dir(new_obj):
                    if attr_name.startswith('_'):
                        continue
                    try:
                        attr_val = getattr(new_obj, attr_name)
                        if callable(attr_val):
                            continue
                        setattr(new_obj, attr_name,
                                _convert(attr_val, parent_key=attr_name))
                    except Exception:
                        pass
                return new_obj
            return obj
        
        return _convert(args)


    @staticmethod
    def _build_model_architecture(args, attn_backend, rope_impl) -> torch.nn.Module:
        try:
            model_type = ModelType(args.type)
        except ValueError:
            raise ValueError(
                f"Model type '{args.type}' is not supported. "
                f"Available types: {[t.value for t in ModelType]}"
            )

        model_cls = get_model_class(model_type)
        logger.info(f"Building model with args: {args.transformer}")

        # # 从 args.transformer 构建正确的模型参数
        # if args.type in ["diff-wan", "diff-wan-22"]:
        try:
            model_kwargs = args.transformer
        except:
            raise ValueError(f"Unsupported model type: {args.type}")
        
        # 创建模型实例
        model = model_cls(model_type=args.task, attn_backend=attn_backend, rope_impl=rope_impl, **model_kwargs)
        
        return model
    
    @staticmethod
    def _load_checkpoint(model: torch.nn.Module, path: str, args: Any, target_device: torch.device):
        """
        每个节点的local rank从目录或单个文件加载多部分 checkpoint (*.safetensors)。
        支持从 meta device 转换到实际设备。
        """
        is_main_process = dist.get_rank() == 0
        path = os.path.expanduser(path)
        checkpoint_files = []

        # --- 1. Scan and validate checkpoint path ---
        if os.path.isfile(path):
            checkpoint_files = [path]
        elif os.path.isdir(path):
            checkpoint_files = sorted(glob(os.path.join(path, "*.safetensors")))
        else:
            # Process index.json
            base_dir = os.path.dirname(path)
            base_name = os.path.basename(path)
            index_file = os.path.join(base_dir, f"{base_name}.index.json")
            
            if os.path.exists(index_file):
                try:
                    with open(index_file, 'r') as f:
                        index_data = json.load(f)
                    shard_files = set(index_data.get('weight_map', {}).values())
                    checkpoint_files = sorted([
                        os.path.join(base_dir, shard_file) 
                        for shard_file in shard_files 
                        if shard_file.endswith('.safetensors')
                    ])
                except Exception as e:
                    if is_main_process:
                        logger.error(f"Failed to parse index file {index_file}: {e}")
            
            # 如果还是没找到，尝试模糊匹配分片
            if not checkpoint_files:
                base_pattern = base_name.replace('.safetensors', '')
                pattern = os.path.join(base_dir, f"{base_pattern}-*-of-*.safetensors")
                checkpoint_files = sorted(glob(pattern))

        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found for path: {path}")

        is_meta = any(p.is_meta for p in model.parameters())
        if is_meta:
            if is_main_process:
                logger.info(f"Materializing model on {target_device}...")
            model.to_empty(device=target_device)
        else:
            model.to(target_device)

        # 获取 state_dict 的引用 (注意：这是引用，修改里面的 Tensor 会直接反映到模型上)
        model_dict = model.state_dict()
        all_loaded_keys = set()
        
        if target_device.type == 'cpu':
            load_device = "cpu"
        elif target_device.type == 'cuda':
            load_device = f"cuda:{target_device.index}" if target_device.index is not None else "cuda"
        else:
            # 其他设备类型，转为字符串
            load_device = str(target_device)
        
        # 开启无梯度模式，减少开销
        with torch.no_grad():
            if is_main_process:
                pbar = tqdm(
                    total=len(checkpoint_files),
                    desc="Loading checkpoints",
                    unit="files"
                )
                
            for i, ckpt_file in enumerate(checkpoint_files):
                if is_main_process:
                    pbar.set_description(f"Loading {os.path.basename(ckpt_file)}")
                
                # --- Fast loading: 直接加载到目标设备 ---
                try:
                    # safetensors + mmap loading
                    checkpoint = st.load_file(ckpt_file, device=load_device)
                except Exception as e:
                    if is_main_process:
                        logger.error(f"Error loading {ckpt_file}: {e}")
                    continue

                # ----------  In-place Copy -------------
                for k, v in checkpoint.items():
                    if k in model_dict:
                        dest = model_dict[k]
                        if v.shape == dest.shape:
                            # 直接将加载的 tensor 拷贝到模型参数的内存中
                            # copy_ 会自动处理跨设备的传输 (CPU -> GPU) 和 dtype 转换
                            dest.copy_(v)
                            all_loaded_keys.add(k)
                        elif is_main_process:
                            logger.warning(f"Shape mismatch for {k}")
                    
                # 及时释放当前分片的显存/内存
                del checkpoint
                # 只有在确实遇到显存压力时才调用，否则会减慢速度
                # torch.cuda.empty_cache()
                
                if is_main_process:
                    pbar.update(1)

            if is_main_process:
                pbar.close()

        # --- 4. 最终验证 ---
        missing_keys = set(model_dict.keys()) - all_loaded_keys
        if missing_keys and is_main_process:
            # 过滤掉一些常见的不需要加载的参数（如某些 buffer）
            if len(missing_keys) > 0:
                logger.warning(f"Missing {len(missing_keys)} keys: {list(missing_keys)[:5]}...")
        
        if is_main_process:
            logger.info(f"Successfully loaded {len(all_loaded_keys)} parameters.")

    @staticmethod
    def memory_used(msg: str = "Memory Usage"):
        """
        Log current GPU and CPU memory usage.
        
        Args:
            msg (str): Description message for the memory log. Will be truncated
                      or padded to 20 characters for consistent formatting.
        """
        # 固定 msg 的输出长度为 20 个字符，超出部分用 ... 代替
        msg = (msg[:17] + '...') if len(msg) > 20 else msg.ljust(20)
        logger.info(
            f"[{msg}] | GPU-Alloc:{torch.cuda.memory_allocated()/1024**3:.3f} Max:{torch.cuda.max_memory_allocated()/1024**3:.3f} Rsrv:{torch.cuda.memory_reserved()/1024**3:.3f} GB  | CPU:{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
        )

    @staticmethod
    def switch_active_model(flush: bool):
        """
        Switch the active DiT model on GPU.
        
        In low memory mode (level >= 3), this function offloads the current
        active model to CPU and loads the next model to GPU. It also updates
        the corresponding guidance scale and boundary configurations.
        
        Args:
            flush (bool): If True, start from the first model (noise model).
                         If False, advance to the next model in sequence.
                         
        Raises:
            IndexError: If the computed model index is out of range.
        """
        # 先将当前活跃模型offload到CPU
        low_mem_level = getattr(DiffusionBackend.args.infer.diffusion, "low_mem_level", 0)

        # offload active model if low memory
        if low_mem_level >= 3 and DiffusionBackend.active_model is not None:
            current_idx = DiffusionBackend.active_model_id
            if current_idx < len(DiffusionBackend.model_pool):
                # 将当前模型移回CPU并保存到model_pool
                DiffusionBackend.model_pool[current_idx] = DiffusionBackend.active_model.to("cpu")
            
            # 清空当前活跃模型引用
            DiffusionBackend.active_model = None
            torch.cuda.empty_cache()
            DiffusionBackend.memory_used("offloaded previous model")

        if flush:
            model_idx = 0  # start from noise
        else:
            model_idx = DiffusionBackend.active_model_id + 1

        # 加载新的活跃模型到GPU
        if model_idx < len(DiffusionBackend.model_pool):
            DiffusionBackend.active_model = DiffusionBackend.model_pool[model_idx].to(torch.cuda.current_device())
            DiffusionBackend.active_model_id = model_idx
            
            # 更新对应的配置
            DiffusionBackend.guidance_scale = DiffusionBackend.args.models.sampler.guidance_scale[model_idx]
            if len(DiffusionBackend.args.models.sampler.boundary) > model_idx:
                DiffusionBackend.boundary = DiffusionBackend.args.models.sampler.boundary[model_idx]
            
            DiffusionBackend.memory_used("loaded new DiT model")
        else:
            raise IndexError(f"Model index {model_idx} out of range for model pool size {len(DiffusionBackend.model_pool)}")


    @staticmethod
    def _init_distributed(args):
        """
        Initialize distributed training environment with tensor and pipeline parallelism.

        Arguments:
            args: Configuration object with distributed parameters
        """
        is_router_process = os.environ.get("CHITU_ROUTER_PROCESS", "0") == "1"
        if is_router_process:
            # Router process: as independent subprocess, skip CUDA device binding
            logger.info(f"[Router] Router subprocess skip CUDA device binding")
            return

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Bind process to GPU. Please put it before init_process_group
        if args.infer.op_impl != "cpu":
            torch.cuda.set_device(local_rank)

        if not torch.distributed.is_initialized():
            if args.infer.op_impl == "cpu":
                torch.distributed.init_process_group("gloo")
            else:
                torch.distributed.init_process_group("nccl")
        if DiffusionBackend.use_gloo:
            DiffusionBackend.group_gloo = torch.distributed.new_group(backend="gloo")

        model_parallel_size = args.infer.tp_size
        pipeline_parallel_size = args.infer.pp_size
        non_expert_data_parallel_size = args.infer.dp_size
        expert_parallel_size = args.infer.ep_size

        # global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        assert model_parallel_size == 1, "DiffusionBackend only supports model_parallel_size=1"
        assert pipeline_parallel_size == 1, "DiffusionBackend only supports pipeline_parallel_size=1"
        assert expert_parallel_size == 1, "DiffusionBackend only supports expert_parallel_size=1"

        # Diffusion Parallelism
        non_expert_data_parallel_size = 1 # TODO: support batch generation with data parallelism

        # FIXME: a better cfg worldsize decision


        # if args.models.name in ["FLUX.2-klein-4B"]:
        #     DiffusionBackend.do_cfg = False
        # else:
        #     DiffusionBackend.do_cfg = all(x > 0 for x in args.models.sampler.guidance_scale)
        # cfg_size = 2 if (world_size >= 2 and  DiffusionBackend.do_cfg and args.infer.diffusion.cfg_size > 1) else 1

        DiffusionBackend.do_cfg = all(x > 0 for x in args.models.sampler.guidance_scale)
        #cfg_size = 2 if (world_size >= 2 and  DiffusionBackend.do_cfg) else 1
        fine_grained_parallel_size = args.infer.diffusion.fpp_size

        if hasattr(args.infer.diffusion, 'cfg_size') and args.infer.diffusion.cfg_size is not None:
            cfg_size = args.infer.diffusion.cfg_size
            logger.info(f"Using explicitly configured cfg_size={cfg_size}")
        else:#默认行为:如果有CFG且有多GPU且没有开启流水线并行，自动启用CFG并行
            cfg_size = 2 if (world_size >= 2 and DiffusionBackend.do_cfg and fine_grained_parallel_size == 1) else 1
            logger.info(f"Auto-determined cfg size={cfg_size} (world size-{world_size}, do_cfg={DiffusionBackend.do_cfg})")

        up_limit = args.infer.diffusion.up_limit
        context_parallel_size = args.infer.diffusion.cp_size

        assert (
            world_size
            == non_expert_data_parallel_size * cfg_size * context_parallel_size * fine_grained_parallel_size
        ), f"World size not match: {world_size} != {non_expert_data_parallel_size} * {cfg_size} * {context_parallel_size}"
        print (f"Initialized distributed with world size {world_size}, cfg size {cfg_size}, context parallel size {context_parallel_size}, fine-grained parallel size {fine_grained_parallel_size}")
        initialize_diffusion_parallel_groups(
            cfg_size= cfg_size,
            up_limit=up_limit,
            cp_size=context_parallel_size,
            fpp_size=fine_grained_parallel_size,
        )

    @staticmethod
    def _setup_environment(args):
        """
        Set up random seed, default dtype, and check prerequisites.

        Arguments:
            args: Configuration with seed and dtype settings
        """
        torch.manual_seed(args.infer.seed)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        DiffusionBackend.is_main_rank = (local_rank == 0)
        # Set default_dtype
        if args.float_16bit_variant == "float16":
            torch.set_default_dtype(torch.float16)
        elif args.float_16bit_variant == "bfloat16":
            torch.set_default_dtype(torch.bfloat16)
        else:
            raise NotImplementedError(f"Unsupported float_16bit_variant {args.dtype}")
        
    @staticmethod
    def _get_init_device(args):
        if torch.cuda.is_available() and args.infer.diffusion.low_mem_level<3:
            device = torch.cuda.current_device()
        else:
            device = 'cpu'
        return torch.device(device)


    @staticmethod
    def _init_text_encoder(args):
        """
        Initialize the multimodal processor for vision-language models.

        Arguments:
            args: Configuration with model settings

        Returns:
            Initialized processor or None if not a multimodal model
        """
        
        if args.infer.diffusion.low_mem_level >= 2:
            init_device = torch.device('cpu') 
        else:
            init_device = DiffusionBackend._get_init_device(args)

        if "Wan" in args.models.name:
            from chitu_diffusion.modules.encoders.t5 import T5EncoderModel
            logger.info(f"Initializing T5 encoder for {args.models.name}")
            text_encoder = T5EncoderModel(
                    text_len=args.models.encoder.text_len,
                    device = init_device,
                    checkpoint_path=os.path.join(args.models.ckpt_dir, args.models.encoder.t5_checkpoint),
                    tokenizer_path=os.path.join(args.models.ckpt_dir, args.models.encoder.t5_tokenizer),
                )
            logger.info(f"Initialized T5 encoder for {args.models.name}")
        elif "FLUX" in args.models.name:
            from chitu_diffusion.modules.encoders.qwen3 import Qwen3Embedder
            logger.info(f"Initializing T5 encoder for {args.models.name}")

            text_encoder = Qwen3Embedder(
                    model_spec=args.models.encoder.ckpt_dir,
                    device = init_device,
                )
            logger.info(f"Initialized T5 encoder for {args.models.name}")
        else:
            text_encoder = None

        logger.info(f"Initialized multimodal processor for {args.models.name}")
        return text_encoder

    @staticmethod
    def _init_vae(args):
        """
        Initialize the VAE model for diffusion.

        Arguments:
            args: Configuration with model settings
        """
        init_device = DiffusionBackend._get_init_device(args)

        if "Wan" in args.models.name:
            # TODO: Wan vae 不支持tiling或者slicing，因此只能offload。
            # 但是offload vae 貌似没有意义，主要显存都是activation。
            # 希望能替换为diffusers版本的vae。
            from chitu_diffusion.modules.vaes.wan_vae import WanVAE
            # from chitu_diffusion.modules.vaes.wan_vae_diffusers import AutoencoderKLWan
            logger.info(f"Initializing Wan VAE for {args.models.name}")
            vae = WanVAE(
                    vae_pth=os.path.join(args.models.ckpt_dir, args.models.vae.checkpoint),
                    device = init_device,
                )
            logger.info(f"Initialized Wan VAE for {args.models.name}")
        elif "FLUX" in args.models.name:
            from diffusers import AutoencoderKL
            logger.info(f"Initializing VAE for {args.models.name}")

            from chitu_diffusion.modules.vaes.flux_vae import FLUX2VAE
            vae = FLUX2VAE(
                    vae_path=args.models.vae.checkpoint,
                    device = init_device,
                )

            logger.info(f"Initialized VAE for {args.models.name}")
        else:
            # 将来其他vae优先支持slicing和tiling
            vae = None
        return vae
    
    @staticmethod
    def _init_cache_manager(args):
        """
        初始化缓存管理器
        将会支持多种Feature Cache策略
        """
        # 此处initialize的主要任务应该是开辟一段显存buffer（cpu/gpu）
        manager = FlexCacheManager(max_cache_memory=20) if args.infer.diffusion.enable_flexcache else None

        # 在模型初始化后启用缓存
        # 注意：这里需要在模型构建完成后调用 enable_cache_for_backend()
        logger.info("[CacheManager] Cache manager initialized")
        return manager

    @staticmethod
    def _init_attention_backend(args):
        """
        Initialize the attention backend implementation.
        
        Sets up the attention mechanism based on the configured attention type
        (e.g., flash_attn, sage, sparge) and wraps it with context parallelism
        if CP size > 1.
        
        Args:
            args: Global configuration containing attention settings.
            
        Returns:
            DiffusionAttnBackend or DiffusionAttention_with_CP: Initialized attention backend.
        """
        attn_type = args.infer.attn_type
        attn = DiffusionAttnBackend(attn_type)
        DiffusionBackend.attn = attn

        if args.infer.diffusion.cp_size > 1:
            attn = DiffusionAttention_with_CP(attn, args.infer.diffusion.up_limit)
        return attn
    
    @staticmethod
    def _get_rope_implementation(args):
        """
        Get the appropriate RoPE (Rotary Position Embedding) implementation.
        
        If context parallelism size > 1, returns a context-parallel aware RoPE
        implementation. Otherwise returns None to use the default implementation.
        
        Args:
            args: Global configuration containing context parallelism settings.
            
        Returns:
            callable or None: RoPE implementation function, or None for default.
        """

        if args.infer.diffusion.cp_size > 1:
            return partial(rope_apply_with_cp, cp_size=get_cp_group().group_size, cp_rank=get_cp_group().rank_in_group)

        elif args.infer.diffusion.fpp_size > 1:
            return rope_apply_with_position 
        
        else:
            return naive_rope_apply 


    # hmx: refactored for Wan2.2 because it has two noise models
    @staticmethod
    def _build_and_setup_model(args, attn_backend, rope_impl):
        """
        Build model architecture, load checkpoints, and apply quantization.

        Arguments:
            args: Configuration with model settings
            attn_backend: The initialized attention backend
            rope_impl: The rope implementation for the model
        """
        # convert args.models lists to tuples
        args.models = DiffusionBackend.check_and_convert_config(args.models)
        DiffusionBackend.args = args

        # build model
        if args.models.name in ["Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B"]:
            ckpt_path = os.path.join(args.models.ckpt_dir, "diffusion_pytorch_model.safetensors")
            model = DiffusionBackend._build_and_setup_single_model(
                args, 
                ckpt_path,
                attn_backend, 
                rope_impl
            ) 
            DiffusionBackend.model_pool.append(model) 
            
        elif args.models.name in ["Wan2.2-T2V-A14B"]:
            # build high noise model
            high_ckpt_path = os.path.join(args.models.ckpt_dir, args.models.high_noise_checkpoint)
            high_noise_model = DiffusionBackend._build_and_setup_single_model(
                args, 
                high_ckpt_path, 
                attn_backend, 
                rope_impl
            )
            # build low noise model
            low_ckpt_path = os.path.join(args.models.ckpt_dir, args.models.low_noise_checkpoint)
            low_noise_model = DiffusionBackend._build_and_setup_single_model(
                args, 
                low_ckpt_path, 
                attn_backend, 
                rope_impl
            )
            DiffusionBackend.model_pool = [high_noise_model, low_noise_model]
        elif args.models.name in ["FLUX.2-klein-4B"]:
            ckpt_path = os.path.join(args.models.ckpt_dir, "flux-2-klein-4b.safetensors")
            model = DiffusionBackend._build_and_setup_single_model(
                args,
                ckpt_path,
                attn_backend,
                rope_impl,
            )
            DiffusionBackend.model_pool.append(model)
        else:
            raise ValueError(f"Unsupported model name: {args.models.name}")

        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _build_and_setup_single_model(args, ckpt_path, attn_backend, rope_impl):
        """
        Build single model architecture, load checkpoints, and apply quantization.

        Arguments:
            args: Configuration with model settings
            ckpt_path: Checkpoint path for the model
            attn_backend: The initialized attention backend
            rope_impl: The rope implementation for the model

        Returns:
            Fully set up single model
        """
        target_device = DiffusionBackend._get_init_device(args)
        # Build the model. Don't allocate memory yet.
        with torch.device("meta"):
            model = DiffusionBackend._build_model_architecture(args.models, attn_backend, rope_impl)

        if not args.debug.skip_model_load:            
            # 调用加载逻辑，在内部完成 meta -> target_device 的转换
            DiffusionBackend._load_checkpoint(model, ckpt_path, args, target_device)
        else:
            # 如果跳过加载，也需要将 meta 模型转为实际模型（随机初始化）
            model.to_empty(device=target_device)
        
        # 如果启用流水并行，在这里进行模型分层，需要model支持分层接口
        # 如果在meta init时就完成分层，性能上最好，但load的时候需要对层idx做区分，尚未实现
        if args.infer.diffusion.fpp_size > 1:
            model.wrap_layers_for_fpp()
        model.eval().requires_grad_(False)
        return model


    @staticmethod
    def build(args):
        """
        Build and initialize the model, tokenizer, cache manager, and other components required for inference.

        Arguments:
            args: Configuration object containing model and training related configurations.
        """
        # Initialize distributed environment
        DiffusionBackend._init_distributed(args)

        # Setup environment and basic configuration
        DiffusionBackend._setup_environment(args)

        # Initialize tokenizer and formatter
        DiffusionBackend.text_encoder = DiffusionBackend._init_text_encoder(args)
        DiffusionBackend.vae = DiffusionBackend._init_vae(args)

        # Initialize feature cache manager
        DiffusionBackend.flexcache = DiffusionBackend._init_cache_manager(args)

        # Initialize attention backend
        # FIXME: 这里的实现有点丑陋了, CP相关应该在wrap model with cp里面做？
        attn_backend = DiffusionBackend._init_attention_backend(args)
        rope_impl = DiffusionBackend._get_rope_implementation(args)
       
        DiffusionBackend._build_and_setup_model(args, attn_backend, rope_impl)

        for model in DiffusionBackend.model_pool:
            model.cache_manager = DiffusionBackend.flexcache

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(
            f"rank {local_rank} Backend initialized with CUDA mem at {torch.cuda.memory_allocated()/1024**3:.2f} GB"
        )
        logger.info(
            f"Using {len(c10d._pg_map)} communication groups. If this number is too high, there may be too much memory reserved for underlying communication libraries."
        )

        return DiffusionBackend

    @staticmethod
    def stop():
        setattr(DiffusionBackend, "active_model", None)
        setattr(DiffusionBackend, "model_pool", None)
        setattr(DiffusionBackend, "cache_manager", None)
        gc.collect()
        torch.cuda.empty_cache()


