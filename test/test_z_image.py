import json
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


def _flexcache_params():
    raw = os.getenv("CHITU_Z_IMAGE_FLEXCACHE_PARAMS", "").strip()
    if not raw:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("CHITU_Z_IMAGE_FLEXCACHE_PARAMS must be a JSON object.")
    return payload


def build_request(args: ServeConfig) -> DiffusionUserRequest:
    request_id = os.getenv("CHITU_RUN_TASK_ID") or f"{gen_req_id()}"
    width, height = [
        int(item.strip())
        for item in os.getenv("CHITU_Z_IMAGE_SIZE", "512,512").lower().replace("x", ",").split(",", 1)
    ]
    steps = int(os.getenv("CHITU_Z_IMAGE_STEPS", str(args.models.sampler.sample_steps)))
    return DiffusionUserRequest(
        request_id=request_id,
        params=DiffusionUserParams(
            role="Alex",
            prompt=(
                'A compact workstation desk with a small sign reading "Z-Image x ChituDiffusion", '
                "soft morning light, crisp product photography, detailed cables and notebooks."
            ),
            seed=int(os.getenv("CHITU_Z_IMAGE_SEED", "42")),
            frame_num=1,
            size=(width, height),
            negative_prompt=os.getenv("CHITU_Z_IMAGE_NEGATIVE_PROMPT", ""),
            num_inference_steps=steps,
            sample_solver="flowmatch_euler",
            flexcache_params=_flexcache_params(),
        ),
    )


def main(args: ServeConfig):
    if args.models.name != "Z-Image":
        raise ValueError(f"test/test_z_image.py expects models=Z-Image, got {args.models.name}.")

    logger.setLevel(logging.DEBUG)
    args.models.sampler.sample_steps = int(os.getenv("CHITU_Z_IMAGE_STEPS", str(args.models.sampler.sample_steps)))
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
            reqs = [build_request(args)]
            run_output_dir = run_context.build_run_output_dir(reqs)
            run_context.activate_run(run_output_dir)
            run_context.apply_request_output_dirs(run_output_dir, reqs)
            run_context.dump_run_metadata(run_output_dir, reqs)
            logger.info("Z-Image request: %s", reqs)
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
            logger.info("Z-Image time cost %.3fs", elapsed_s)
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
