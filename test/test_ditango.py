import torch
import time
import os
import copy
import logging
from logging import getLogger

from chitu_diffusion.runtime.main import (
    chitu_init,
    chitu_generate,
    warmup_diffusion_engine,
    chitu_start,
    chitu_terminate,
    chitu_run_eval,
    chitu_is_terminated,
)

# from chitu_diffusion.core.task import UserRequest, TaskPool, Task
from chitu_diffusion.runtime.task import DiffusionUserRequest, DiffusionTask, DiffusionTaskPool, DiffusionUserParams, FlexCacheParams

from chitu_diffusion.core.schemas import ServeConfig
from chitu_diffusion.core.config_loader import load_config_from_cli
from chitu_diffusion.core.utils import gen_req_id
from chitu_diffusion.observability import Timer
from run_context import DiffusionTestRunContext, should_record_metrics_on_rank

logger = getLogger(__name__)

# T2V prompts
msgs = [
    DiffusionUserParams(
        role="Alex",
        prompt="A cat walking on grass.",
        seed=42,
        frame_num=81,
        size=(832, 480),  # 14b: 1280 720
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=50,
        sample_solver="unipc",
        flexcache_params=FlexCacheParams(
            strategy="ditango",
            cache_ratio=0,
            warmup=7,
            cooldown=3,
        ),
    ),
    DiffusionUserParams(
        role="Bob",
        prompt="A cat walking on grass.",
        seed=42,
        frame_num=81,
        size=(832, 480),  # 14b: 1280 720
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=50,
        sample_solver="unipc",
        flexcache_params=FlexCacheParams(
            strategy="ditango",
            cache_ratio=1,
            warmup=7,
            cooldown=3,
        ),
    ),
    # DiffusionUserParams(
    #     role="Bob",
    #     prompt="two cats walking on grass.",
    #     seed=42,
    #     frame_num=81,
    #     size=(832, 480),
    #     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    #     num_inference_steps=5,
    #     sample_solver="unipc",
    # ),
]


def gen_reqs(num_reqs, max_new_tokens, frequency_penalty, is_vl=False):
    reqs: list[DiffusionUserRequest] = []
    for i in range(num_reqs):
        params = copy.deepcopy(msgs[i])
        request_id = os.getenv("CHITU_RUN_TASK_ID") if i == 0 else None
        req = DiffusionUserRequest(
            request_id=request_id or f"{gen_req_id()}",
            params=params,
        )
        reqs.append(req)
    return reqs


def run_normal(args, run_context: DiffusionTestRunContext):
    rank = torch.distributed.get_rank()

    for i in range(1):
        reqs = []
        run_output_dir = None
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=len(msgs),  # args.infer.max_reqs
                max_new_tokens=args.request.max_new_tokens,
                frequency_penalty=args.request.frequency_penalty,
                is_vl=hasattr(args.models, "vision_config"),
            )
            run_output_dir = run_context.build_run_output_dir(reqs)
            run_context.activate_run(run_output_dir)
            run_context.apply_request_output_dirs(run_output_dir, reqs)

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            payload = [run_output_dir]
            torch.distributed.broadcast_object_list(payload, src=0)
            run_output_dir = payload[0]

        if run_output_dir:
            run_context.activate_run(run_output_dir)

        log_handler = None
        if (
            run_output_dir
            and getattr(args.output, "run_log", True)
            and os.getenv("CHITU_RUN_LOG_ATTACHED", "0") != "1"
        ):
            log_handler = run_context.attach_run_log_handler(run_output_dir)

        if rank == 0:
            run_context.dump_run_metadata(run_output_dir, reqs)
            logger.info(f"{reqs=}")

        run_context.dump_memory_snapshot(run_output_dir, "model_loaded")
        warmup_diffusion_engine(args)
        chitu_start()

        if rank == 0:
            for req in reqs:
                DiffusionTaskPool.add(DiffusionTask(task_id=req.request_id, req=req))

        logger.info(f"------ batch {i} ------")
        t_start = time.time()
        with Timer.get_timer("overall"):
            while not chitu_is_terminated():
                chitu_generate()
                if rank == 0 and DiffusionTaskPool.all_finished():
                    break

        if rank == 0:
            t_end = time.time()
            elapsed_s = t_end - t_start
            logger.info(f"Time cost {t_end - t_start}")
        run_context.log_final_memory()
        run_context.dump_memory_snapshot(run_output_dir, "final")

        if rank == 0 and getattr(args.output, "timer", False):
            Timer.print_statistics()
            run_context.dump_timing_summary(run_output_dir, elapsed_s=locals().get("elapsed_s"))

    chitu_terminate()
    chitu_run_eval()

    if log_handler is not None:
        run_context.close_run_log_handler(log_handler)

def main(args: ServeConfig):
    global local_args
    local_args = args
    logger.setLevel(logging.DEBUG)
    run_context = DiffusionTestRunContext(args, logger, per_task_results=True, capture_stdio_log=True)
    initial_run_output_dir = run_context.build_initial_run_output_dir()
    run_context.activate_run(initial_run_output_dir)

    early_log_handler = None
    if getattr(args.output, "run_log", True) and should_record_metrics_on_rank():
        early_log_handler = run_context.attach_run_log_handler(initial_run_output_dir)
        os.environ["CHITU_RUN_LOG_ATTACHED"] = "1"

    if os.getenv("RANK") == "0":
        logger.info(f"Run with args: {args}")

    # Initialize Backend: args / distributed / load models & kernels
    chitu_init(args, logging_level=logging.INFO)
    logger.info("initialized chitu_diffusion.core.")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    logger.debug("finish init")

    try:
        run_normal(args, run_context)
    finally:
        if early_log_handler is not None:
            run_context.close_run_log_handler(early_log_handler)


if __name__ == "__main__":
    main(load_config_from_cli())

    # Sometimes torch.distributed will hang during destruction if CUDA graph is enabled.
    # As a workaround, we `exec` a dummy process to kill the current process, without
    # returning an error.
    logger.info("Waiting for all ranks to finish...")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    # Don't exec bash because it loads startup scripts
    os.execl("/usr/bin/true", "true")  # /usr/bin/true does nothing but exits
