# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import operator
import os
from logging import getLogger
import psutil

import torch
import torch.distributed

from chitu_core.device_type import is_nvidia
from chitu_core.global_vars import (
    get_global_args,
    set_global_variables,
    set_quant_variables,
    set_backend_variables,
)
from chitu_core.utils import (
    gen_req_id,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
    ceil_div,
)
from chitu_core.schemas.utils import ModelConfigResolver
from chitu_core.utils import ceil_div
from chitu_core.distributed.parallel_state import get_dp_group
from chitu_core.logging_utils import setup_chitu_logging

from chitu_diffusion.task import DiffusionTask, DiffusionTaskPool, DiffusionTaskType
from chitu_diffusion.backend import DiffusionBackend, BackendState
from chitu_diffusion.generator import Generator
from chitu_diffusion.scheduler import DiffusionScheduler

numa, has_numa = try_import_opt_dep("numa", "cpu")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")


logger = getLogger(__name__)


from logging import getLogger
import logging

def init_logger(logging_level=logging.INFO):
    """
    Initialize the Chitu logging system.
    
    Sets up logging handlers and configures the base logger for the chitu_diffusion module.
    If root logger has handlers, they are copied to avoid duplicate logging.
    
    Args:
        logging_level: Logging level to set (default: logging.INFO).
        
    Returns:
        Logger: Configured logger instance for chitu_diffusion.
    """
    setup_chitu_logging()

    base_name = __name__.split(".")[0]
    base_logger = getLogger(base_name)
    base_logger.setLevel(logging_level)

    if base_logger.handlers:
        return base_logger
    root_logger = getLogger()
    if root_logger.handlers:
        base_logger.handlers = []  # 清空当前handlers
        base_logger.propagate = False  # 防止向上传播
        for handler in root_logger.handlers:
            base_logger.addHandler(handler)

    return base_logger


def init_cache_static():
    """
    Initialize CUDA cache statistics.
    
    Clears the CUDA cache and resets peak memory statistics for tracking
    memory usage during inference.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)


def warmup_diffusion_engine(args):
    """
    Warm up the diffusion engine (placeholder).
    
    TODO: Implement warmup logic if needed for performance optimization.
    
    Args:
        args: Global configuration arguments.
    """
    # TODO: Why we need warmup and how?
    pass


def check_checkpoint_path(args):
    """
    Validate that the checkpoint directory is provided.
    
    Checks if the checkpoint directory exists in the configuration and
    raises an error with helpful information if missing.
    
    Args:
        args: Global configuration containing models.ckpt_dir.
        
    Raises:
        ValueError: If checkpoint directory is not provided, with guidance
                   on how to set it and where to download the model.
    """

    if args.models.ckpt_dir is None:
        raise ValueError(
            f"No checkpoint path provided. You can set it in command line by adding `models.ckpt_dir=<path>`. The model {args.models.name} can be downloaded from {args.models.source}"
        )


def chitu_init(args, logging_level=None):
    """
    Initialize the Chitu diffusion system.
    
    This is the main initialization function that:
    1. Sets up logging
    2. Configures environment variables and parameters
    3. Initializes distributed training
    4. Validates and loads model checkpoints
    5. Sets up the backend, scheduler, and generator
    
    Args:
        args: Configuration object containing all system parameters.
        logging_level: Optional logging level override. If None, uses DEBUG
                      in debug mode, INFO otherwise.
    
    Note:
        This function must be called before any inference operations.
    """

    debug = os.getenv("CHITU_DEBUG", "0") == "1"

    if (
        is_nvidia()
        and torch.distributed.is_nccl_available()
        and torch.cuda.nccl.version() <= (2, 21, 5)
    ):
        os.environ["NCCL_NVLS_NCHANNELS"] = "32"

    if logging_level is None:
        logging_level = logging.DEBUG if debug else logging.INFO
    init_logger(logging_level)

    # Deal with legacy arguments
    if hasattr(args.infer, "soft_fp8") and args.infer.soft_fp8:
        logger.warning(
            "Argument `infer.soft_fp8=True` is deprecated. Use `infer.raise_lower_bit_float_to=bfloat16` instead."
        )
        args.infer.raise_lower_bit_float_to = "bfloat16"
    if hasattr(args, "dtype") and args.dtype is not None:
        logger.warning(
            "Argument `dtype` is deprecated. Use `float_16bit_variant` instead."
        )
        args.float_16bit_variant = args.dtype
    if hasattr(args.infer, "do_load") and not args.infer.do_load:
        logger.warning(
            "Argument `infer.do_load=False` is deprecated. Use `debug.skip_model_load=True` instead."
        )
        args.debug.skip_model_load = True


    # Bind process to CPU NUMA
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if args.infer.bind_process_to_cpu == "auto":
        if not has_cpuinfer and not has_numa:
            args.infer.bind_process_to_cpu = "none"
        elif not has_numa:
            logger.warning(
                "'cpuinfer' is found but 'numa' is missing. Disabling NUMA binding. "
                "For better CPU inference performance, please refer to README.md and "
                "install the full '[cpu]' optional dependency."
            )
            args.infer.bind_process_to_cpu = "none"
        elif not numa.available():
            logger.warning(
                "NUMA is not support on this OS or hardware platform. Disabling NUMA binding."
            )
            args.infer.bind_process_to_cpu = "none"
        elif numa.get_max_node() + 1 < local_world_size:
            logger.info("Disable NUMA binding due to insufficient NUMA nodes.")
            args.infer.bind_process_to_cpu = "none"
        else:
            args.infer.bind_process_to_cpu = "numa"
    if args.infer.bind_process_to_cpu == "numa":
        numa.bind({local_rank})
    elif args.infer.bind_process_to_cpu == "none":
        pass
    else:
        raise ValueError(
            f"Unsupported infer.bind_process_to_cpu={args.infer.bind_process_to_cpu}"
        )

    # TODO: Support cuda graph

    # Check checkpoint exists
    check_checkpoint_path(args)

    # Parse model configuration, supporting dynamic reading from config.json files
    # Uses $(config.json:field_name) syntax, e.g., n_heads: "$(config.json:head_dim)"
    model_resolver = ModelConfigResolver()
    args.models = model_resolver.process_config_dict(args.models, args.models.ckpt_dir)

    set_quant_variables(args)
    set_backend_variables(args)
    set_global_variables(args, debug=debug)

    args = get_global_args()
    DiffusionBackend.build(args)
    
    # Naive Diffusion scheduler
    rank = torch.distributed.get_rank()
    if rank == 0:
        scheduler = DiffusionScheduler.build(args.infer.diffusion)
        DiffusionBackend.scheduler = scheduler
    
    generator = Generator.build(args)
    
    DiffusionBackend.generator = generator
    
    # TODO: Support batched generation
    # PackedTasks.configure(max_num_tasks=args.infer.max_reqs)
    
    logger.info("Chitu has been initialized")


@torch.inference_mode()
def chitu_run_main_rank():
    """
    Execute one inference step on the main rank (rank 0).
    
    This function:
    1. Schedules the next task using the scheduler
    2. Processes the scheduled task through the generator
    3. Handles task completion
    
    Note:
        This should only be called on rank 0. Other ranks should call
        chitu_generate() which delegates to the generator directly.
    """
    task_ids = DiffusionBackend.scheduler.schedule()
    # logger.info(f"[Scheduler] scheduled task_ids={task_ids}")
    
    # 再基于task_ids给出打包
    if not DiffusionTaskPool.all_finished():
        # compute
        logger.debug(f"Processing {task_ids}")
        task = DiffusionTaskPool.pool[task_ids[0]]
        out = DiffusionBackend.generator.step(task)
        if out is not None:
            logger.debug(f"[run] executor.step returned. {out.shape=}")
        # postprocess        
    else:
        logger.debug("No tasks scheduled in this round.")

@torch.inference_mode()
def chitu_generate():
    """
    Execute one generation step across all ranks.
    
    This is the main generation function that should be called in a loop.
    Rank 0 schedules and processes tasks, while other ranks participate
    in distributed computation.
    
    Note:
        Must be called on all ranks in a synchronized manner for distributed inference.
    """
    rank = torch.distributed.get_rank()
    if rank != 0:
        DiffusionBackend.generator.step(None) 
        return
    chitu_run_main_rank()

def chitu_start():
    """
    Mark the backend as running and ready to process tasks.
    """
    DiffusionBackend.state = BackendState.Running

def chitu_terminate():
    """
    Gracefully terminate the Chitu backend.
    
    Signals all ranks to stop processing by setting the backend state to
    Terminated and sending a termination signal through the generator.
    """
    if torch.distributed.get_rank() == 0:
        DiffusionBackend.state = BackendState.Terminated
        terminated_task = DiffusionTask.create_terminate_signal("0x")
        DiffusionBackend.generator.step(terminated_task)

def chitu_run_eval():
    """
    Run evaluation on generated videos.
    
    Sets up the evaluation manager and runs the configured evaluation strategy
    (e.g., VBench) on all completed tasks. Evaluation is skipped if eval_type
    is not set.
    
    Raises:
        ValueError: If an unsupported eval_type is specified.
    """
    from chitu_diffusion.eval.eval_manager import EvalManager
    manager = EvalManager()
    args = get_global_args()

    if args.eval.eval_type == None:
        return
    elif args.eval.eval_type=='vbench':
        from chitu_diffusion.eval.strategy.Vbench import VbenchStrategy
        strategy = VbenchStrategy()
    else:
        raise ValueError(f"Unsupported eval type: {args.eval.eval_type}")
    manager.set_strategy(strategy)
    manager.run(args=args)
    


def chitu_is_terminated():
    """
    Check if the Chitu backend has been terminated.
    
    Returns:
        bool: True if the backend is in Terminated state, False otherwise.
    """
    return DiffusionBackend.state == BackendState.Terminated