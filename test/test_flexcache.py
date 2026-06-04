import copy
import logging
import os
import time
from logging import getLogger

import torch

from chitu_diffusion.core.config_loader import load_config_from_cli
from chitu_diffusion.core.schemas import ServeConfig
from chitu_diffusion.core.utils import gen_req_id
from chitu_diffusion.observability import Timer
from chitu_diffusion.runtime.main import (
    chitu_generate,
    chitu_init,
    chitu_is_terminated,
    chitu_run_eval,
    chitu_start,
    chitu_terminate,
    warmup_diffusion_engine,
)
from chitu_diffusion.flexcache.params import BlockDanceParams, CubicParams, PABParams, TaylorSeerParams, TeaCacheParams
from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskPool,
    DiffusionUserParams,
    DiffusionUserRequest,
)
from flexcache_summary import write_flexcache_comparison
from run_context import DiffusionTestRunContext, should_record_metrics_on_rank

logger = getLogger(__name__)


def get_wan_steps() -> int:
    return int(os.getenv("CHITU_WAN_STEPS", "50"))


def get_wan_warmup_cooldown() -> tuple[int, int]:
    steps = get_wan_steps()
    if steps <= 10:
        return 1, 1
    return 7, 3


def build_wan_flexcache_params():
    strategy = os.getenv("CHITU_WAN_FLEXCACHE_STRATEGY", "taylorseer").strip().lower()
    warmup, cooldown = get_wan_warmup_cooldown()
    if strategy == "teacache":
        return TeaCacheParams(
            warmup=warmup,
            cooldown=cooldown,
            teacache_thresh=0.2,
            coefficients=None,
            use_ref_steps=True,
        )
    if strategy == "pab":
        return PABParams(warmup=warmup, cooldown=cooldown, skip_self_range=2, skip_cross_range=3)
    if strategy == "blockdance":
        return BlockDanceParams(warmup=warmup, cooldown=cooldown)
    if strategy == "cubic":
        return CubicParams(target_speedup=2.0, warmup=warmup, cooldown=cooldown)
    if strategy == "taylorseer":
        return TaylorSeerParams(
            warmup=warmup,
            cooldown=cooldown,
            fresh_threshold=5,
            max_order=1,
        )
    raise ValueError(f"Unsupported CHITU_WAN_FLEXCACHE_STRATEGY={strategy!r}")


msgs = [
    DiffusionUserParams(
        role="Alex",
        prompt="A cat walking on grass.",
        seed=42,
        frame_num=81,
        size=(832, 480),
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        num_inference_steps=get_wan_steps(),
        sample_solver="unipc",
        flexcache_params=build_wan_flexcache_params(),
    ),
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
    run_output_dir = None

    for i in range(1):
        reqs = []
        log_handler = None
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=len(msgs),
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

        if (
            run_output_dir
            and getattr(args.output, "run_log", True)
            and os.getenv("CHITU_RUN_LOG_ATTACHED", "0") != "1"
        ):
            log_handler = run_context.attach_run_log_handler(run_output_dir)

        try:
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
                logger.info(f"Time cost {elapsed_s}")
            run_context.log_final_memory()
            run_context.dump_memory_snapshot(run_output_dir, "final")

            if rank == 0 and getattr(args.output, "timer", False):
                Timer.print_statistics()
                run_context.dump_timing_summary(run_output_dir, elapsed_s=locals().get("elapsed_s"))
        finally:
            if log_handler is not None:
                run_context.close_run_log_handler(log_handler)

    chitu_terminate()
    chitu_run_eval()
    if run_output_dir and rank == 0:
        write_flexcache_comparison(run_output_dir)
        logger.info("FlexCache comparison written under %s/metrics", run_output_dir)


def main(args: ServeConfig):
    global local_args
    local_args = args
    logger.setLevel(logging.DEBUG)
    args.infer.diffusion.enable_flexcache = True

    run_context = DiffusionTestRunContext(args, logger, per_task_results=True, capture_stdio_log=True)
    initial_run_output_dir = run_context.build_initial_run_output_dir()
    run_context.activate_run(initial_run_output_dir)

    early_log_handler = None
    if getattr(args.output, "run_log", True) and should_record_metrics_on_rank():
        early_log_handler = run_context.attach_run_log_handler(initial_run_output_dir)
        os.environ["CHITU_RUN_LOG_ATTACHED"] = "1"

    if os.getenv("RANK") == "0":
        logger.info(f"Run with args: {args}")

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
    try:
        main(load_config_from_cli())
        logger.info("Waiting for all ranks to finish...")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
