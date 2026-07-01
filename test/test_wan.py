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
from chitu_diffusion.runtime.task import DiffusionTask, DiffusionTaskPool, DiffusionUserParams, DiffusionUserRequest
from run_context import DiffusionTestRunContext, should_record_metrics_on_rank

logger = getLogger(__name__)


def build_request() -> DiffusionUserRequest:
    request_id = os.getenv("CHITU_RUN_TASK_ID") or f"{gen_req_id()}"
    steps = int(os.getenv("CHITU_WAN_STEPS", "1"))
    frame_num = int(os.getenv("CHITU_WAN_FRAMES", "81"))
    width = int(os.getenv("CHITU_WAN_WIDTH", "832"))
    height = int(os.getenv("CHITU_WAN_HEIGHT", "480"))
    return DiffusionUserRequest(
        request_id=request_id,
        params=DiffusionUserParams(
            role="Alex",
            prompt="A cat walking on grass.",
            seed=42,
            frame_num=frame_num,
            size=(width, height),
            negative_prompt=(
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
                "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
                "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
                "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            ),
            num_inference_steps=steps,
            sample_solver=os.getenv("CHITU_WAN_SOLVER", "unipc"),
            flexcache_params=None,
        ),
    )


def main(args: ServeConfig):
    if args.models.name not in {"Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B", "Wan2.2-T2V-A14B"}:
        raise ValueError(f"test/test_wan.py expects a Wan model, got {args.models.name}.")

    logger.setLevel(logging.DEBUG)
    run_context = DiffusionTestRunContext(args, logger, per_task_results=True, capture_stdio_log=True)
    initial_run_output_dir = run_context.build_initial_run_output_dir()
    run_context.activate_run(initial_run_output_dir)

    early_log_handler = None
    if getattr(args.output, "run_log", True) and should_record_metrics_on_rank():
        early_log_handler = run_context.attach_run_log_handler(initial_run_output_dir)
        os.environ["CHITU_RUN_LOG_ATTACHED"] = "1"

    try:
        chitu_init(args, logging_level=logging.INFO)
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

        rank = torch.distributed.get_rank()
        run_output_dir = None
        if rank == 0:
            reqs = [build_request()]
            run_output_dir = run_context.build_run_output_dir(reqs)
            run_context.activate_run(run_output_dir)
            run_context.apply_request_output_dirs(run_output_dir, reqs)
            run_context.dump_run_metadata(run_output_dir, reqs)
            logger.info("Wan request: %s", reqs)
        else:
            reqs = []

        payload = [run_output_dir]
        torch.distributed.broadcast_object_list(payload, src=0)
        run_output_dir = payload[0]
        if run_output_dir:
            run_context.activate_run(run_output_dir)

        run_context.dump_memory_snapshot(run_output_dir, "model_loaded")
        warmup_diffusion_engine(args)
        chitu_start()

        if rank == 0:
            for req in reqs:
                DiffusionTaskPool.add(DiffusionTask(task_id=req.request_id, req=req))

        start = time.time()
        with Timer.get_timer("overall"):
            while not chitu_is_terminated():
                chitu_generate()
                if rank == 0 and DiffusionTaskPool.all_finished():
                    break

        elapsed_s = time.time() - start
        if rank == 0:
            logger.info("Wan time cost %.3fs", elapsed_s)
        run_context.dump_memory_snapshot(run_output_dir, "final")
        if rank == 0 and getattr(args.output, "timer", False):
            Timer.print_statistics()
            run_context.dump_timing_summary(run_output_dir, elapsed_s=elapsed_s)

        chitu_terminate()
        chitu_run_eval()
    finally:
        if early_log_handler is not None:
            run_context.close_run_log_handler(early_log_handler)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(load_config_from_cli())
