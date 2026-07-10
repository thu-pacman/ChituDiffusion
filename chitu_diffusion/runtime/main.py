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

from chitu_diffusion.core.device_type import is_nvidia
from chitu_diffusion.core.global_vars import (
    get_global_args,
    set_global_variables,
    set_quant_variables,
    set_backend_variables,
)
from chitu_diffusion.core.utils import (
    gen_req_id,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
    ceil_div,
)
from chitu_diffusion.core.schemas.utils import ModelConfigResolver
from chitu_diffusion.core.utils import ceil_div
from chitu_diffusion.core.distributed.parallel_state import get_dp_group
from chitu_diffusion.core.logging_utils import setup_chitu_logging

from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskPool,
    DiffusionTaskType,
    DiffusionUserParams,
    DiffusionUserRequest,
)
from chitu_diffusion.runtime.backend import DiffusionBackend, BackendState
from chitu_diffusion.runtime.generator import Generator
from chitu_diffusion.runtime.scheduler import DiffusionScheduler
from chitu_diffusion.runtime.output_layout import quality_metrics_dir
from chitu_diffusion.observability import Timer

WARMUP_TASK_PREFIX = "__warmup__"

numa, has_numa = try_import_opt_dep("numa", "cpu")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")


logger = getLogger(__name__)


from logging import getLogger
import logging


def _env_flag(name: str, default: str = "0") -> bool:
    value = str(os.getenv(name, default)).strip().lower()
    return value in {"1", "true", "yes", "on"}

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


def _warmup_shapes(args) -> list[tuple[int, int]]:
    """Distinct (width, height) shapes to pre-run once at engine startup.

    Read from ``CHITU_WARMUP_SHAPES`` (e.g. ``"1024x1024,512x512"``); otherwise choose
    a conservative default for the active model family. A caller that knows its exact
    request mix (e.g. a trace-driven serve) can narrow/extend this via the env var
    without touching the engine."""
    spec = os.getenv("CHITU_WARMUP_SHAPES", "").strip()
    if not spec:
        model = str(getattr(args.models, "name", "") or "")
        if model.startswith("Wan"):
            spec = "832x480"
        elif model == "Qwen-Image":
            spec = "1328x1328"
        else:
            spec = "1024x1024,512x512"
    shapes: list[tuple[int, int]] = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok or "x" not in tok:
            continue
        try:
            w, h = (int(x) for x in tok.split("x", 1))
        except ValueError:
            continue
        if (w, h) not in shapes:
            shapes.append((w, h))
    return shapes


def _warmup_request_params(args, width: int, height: int, steps: int) -> DiffusionUserParams:
    """Build a model-family-safe warmup request.

    Keep this inside the engine instead of copying a benchmark/test request shape. Different
    adapters expect different harmless defaults (e.g. Wan is video-like and uses ``unipc``;
    image models usually use one frame and ``flowmatch_euler``). Environment variables can
    override the key fields when a deployment needs a model-specific warmup.
    """
    model = str(getattr(args.models, "name", "") or "")
    sampler = getattr(args.models, "sampler", None)
    solver_default = str(getattr(sampler, "sample_solver", "") or "")
    if not solver_default:
        solver_default = "unipc" if model.startswith("Wan") else "flowmatch_euler"

    if model.startswith("Wan"):
        prompt_default = "A cat walking on grass."
        negative_default = (
            "low quality, blurry, distorted, deformed, watermark, text, "
            "overexposed, underexposed, bad anatomy"
        )
        frame_default = int(os.getenv("CHITU_WAN_FRAMES", "81") or 81)
    elif model == "Qwen-Image":
        prompt_default = (
            'A coffee shop entrance features a chalkboard sign reading "Chitu Warmup", '
            "ultra HD, cinematic composition."
        )
        negative_default = " "
        frame_default = 1
    else:
        prompt_default = "A photorealistic cute cat against a simple background."
        negative_default = ""
        frame_default = 1

    return DiffusionUserParams(
        role=str(os.getenv("CHITU_WARMUP_ROLE", "client")),
        prompt=str(os.getenv("CHITU_WARMUP_PROMPT", prompt_default)),
        negative_prompt=os.getenv("CHITU_WARMUP_NEGATIVE_PROMPT", negative_default),
        seed=int(os.getenv("CHITU_WARMUP_SEED", "0") or 0),
        frame_num=int(os.getenv("CHITU_WARMUP_FRAMES", str(frame_default)) or frame_default),
        size=(int(width), int(height)),
        num_inference_steps=int(steps),
        n_sample=1,
        sample_solver=str(os.getenv("CHITU_WARMUP_SOLVER", solver_default)),
        flexcache_params=None,
    )


def warmup_diffusion_engine(args):
    """Warm the engine before serving so the first *real* request does not absorb the
    one-time cold-start (cuDNN autotune, CUDA-graph capture, first NCCL collective on
    each lane subgroup). Left unwarmed this cost lands on the first request and shifts
    the whole timeline by several seconds -- not a scheduling effect, but it pollutes
    A/B comparisons.

    Runs one throwaway request per distinct shape to completion through the real pool
    engine (a lone request gets the widest layout -- the same path the first real
    request hits), then resets all engine/instrumentation state so the served run starts
    from zero. Lockstep across ranks and fully exception-safe: a warmup hiccup can never
    desync or break serving. Gated by ``CHITU_WARMUP`` (default on)."""
    if str(os.getenv("CHITU_WARMUP", "on")).strip().lower() in {"0", "off", "no", "false", ""}:
        return
    generator = DiffusionBackend.generator
    # The self-driven warmup uses the pool-engine round driver (rank-0 plan + broadcast).
    # Other engine modes are left untouched.
    if generator is None or not getattr(generator, "pool_engine", False):
        return
    # ``chitu_init`` owns startup warmup. Keep this public helper idempotent for older
    # entry points that still call it explicitly after initialization.
    if bool(getattr(generator, "_startup_warmup_complete", False)):
        return
    generator._startup_warmup_complete = True
    shapes = _warmup_shapes(args)
    if not shapes:
        return
    steps = max(2, int(os.getenv("CHITU_WARMUP_STEPS", "6") or 6))
    rank = torch.distributed.get_rank()
    device = torch.cuda.current_device()
    ACTION_RUN, ACTION_STOP = 0, 2

    try:
        for si, (w, h) in enumerate(shapes):
            # Inject shapes one at a time so the lone request gets the widest layout,
            # warming the full-width (cfp2xcp4) path the first real request will hit.
            if rank == 0:
                req = DiffusionUserRequest(
                    request_id=f"{WARMUP_TASK_PREFIX}{si}",
                    params=_warmup_request_params(args, int(w), int(h), steps),
                )
                DiffusionTaskPool.add(DiffusionTask(task_id=req.request_id, req=req))
            # Lockstep drive: RUN until the pool drains, then STOP (bounded for safety).
            for _ in range(steps * 8 + 64):
                if rank == 0:
                    action = ACTION_RUN if DiffusionBackend.scheduler.can_schedule() else ACTION_STOP
                else:
                    action = ACTION_RUN
                action_t = torch.tensor([action if rank == 0 else 0], dtype=torch.int64, device=device)
                torch.distributed.broadcast(action_t, src=0)
                if int(action_t.item()) == ACTION_STOP:
                    break
                chitu_generate()
        logger.info("[warmup] engine warm: %d shape(s) x %d steps", len(shapes), steps)
    except Exception as exc:  # never let warmup break serving
        logger.warning("[warmup] skipped after error: %s", exc)
    finally:
        # Discard all warmup state so the served run's metrics/timeline start from zero.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[device])
        DiffusionTaskPool.reset()
        Timer.reset()
        gen = getattr(DiffusionBackend, "generator", None)
        if gen is not None:
            for attr, val in (
                ("_pool_round_index", 0), ("_pool_last_layout_sig", None),
                ("_pool_placement_reuse", 0), ("_pool_placement_change", 0),
                ("_pool_last_predicted_spin_ms", 0.0),
            ):
                if hasattr(gen, attr):
                    setattr(gen, attr, val)
            for attr in ("_pool_prev_placement", "pool_group"):
                container = getattr(gen, attr, None)
                if container is not None and hasattr(container, "clear"):
                    container.clear()


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

    debug = _env_flag("CHITU_DEBUG", "0")

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

    # Warm the fully-built engine before returning from initialization. This runs the
    # same pool-round path as real serving, then resets task/timer state, so every entry
    # point gets identical cold-start handling without benchmark-specific calls.
    warmup_diffusion_engine(args)
    
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
    generator = DiffusionBackend.generator

    # slo_elastic pool engine: rank 0 plans a full pool layout each round and
    # broadcasts it; every rank runs one denoise step for its lane. Admission and
    # retirement are owned by the engine round, so rank 0 just drives one round here.
    if getattr(generator, "pool_engine", False):
        generator.step()
        return

    decisions = DiffusionBackend.scheduler.schedule_decisions()
    task_ids = [decision.task_id for decision in decisions]
    # logger.info(f"[Scheduler] scheduled decisions={decisions}")

    if not task_ids:
        logger.debug("No tasks scheduled in this round.")
        return

    # Single-task path (unchanged).
    logger.debug(f"Processing {task_ids}")
    decision = decisions[0]
    task = DiffusionTaskPool.get_control_task(task_ids[0])
    if task is None:
        task = DiffusionTaskPool.pool[task_ids[0]]
    out = generator.step(task)
    if out is not None:
        logger.debug(f"[run] executor.step returned. {out.shape=}")

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
    # Rank 0 owns stage allocation and denoise scheduling; every rank enters the
    # same engine round and executes only the lane assigned by the broadcast plan.
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
    if DiffusionBackend.state == BackendState.Terminated:
        return
    if torch.distributed.get_rank() == 0:
        DiffusionTaskPool.request_shutdown("Chitu terminate requested")
    chitu_generate()


def chitu_cancel_current(reason: str = "Current generation cancelled"):
    if torch.distributed.get_rank() == 0:
        DiffusionTaskPool.request_cancel(reason)
    chitu_generate()

def chitu_run_eval():
    """
    Run one or multiple evaluation strategies on generated videos.

    The eval list is read from args.eval.eval_type. Empty/null means disabled.
    Unknown strategies and reference-dependent strategies without reference_path
    are skipped with warning.
    """
    from chitu_diffusion.evaluation.eval_manager import EvalManager

    manager = EvalManager()
    args = get_global_args()

    eval_types = manager.normalize_eval_types(getattr(args.eval, "eval_type", None))
    if not eval_types:
        return

    current_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
    eval_output_dir = quality_metrics_dir(current_output_dir) if current_output_dir else None
    manager.run(args=args, eval_types=eval_types, output_dir=eval_output_dir)
    


def chitu_is_terminated():
    """
    Check if the Chitu backend has been terminated.
    
    Returns:
        bool: True if the backend is in Terminated state, False otherwise.
    """
    return DiffusionBackend.state == BackendState.Terminated
