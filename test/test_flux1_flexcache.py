import copy
import logging
import os
import time
from logging import getLogger

import torch

from chitu_diffusion.core.config_loader import load_config_from_cli
from chitu_diffusion.core.schemas import ServeConfig
from chitu_diffusion.core.utils import gen_req_id
from chitu_diffusion.flexcache.params import BlockDanceParams, PABParams, TaylorSeerParams, TeaCacheParams
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
from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskPool,
    DiffusionUserParams,
    DiffusionUserRequest,
)
from flexcache_summary import write_flexcache_comparison
from run_context import DiffusionTestRunContext, should_record_metrics_on_rank


logger = getLogger(__name__)

FLUX1_PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
FLUX1_STEPS = 50


msgs = [
    DiffusionUserParams(
        role="Alex",
        prompt=FLUX1_PROMPT,
        seed=42,
        frame_num=1,
        size=(1024, 1024),
        negative_prompt=None,
        num_inference_steps=FLUX1_STEPS,
        sample_solver="flowmatch_euler",
        flexcache_params=TeaCacheParams(
            warmup=3,
            cooldown=3,
            teacache_thresh=0.6,
            coefficients=None,
            use_ref_steps=True,
        ),
    ),
    DiffusionUserParams(
        role="Alex",
        prompt=FLUX1_PROMPT,
        seed=42,
        frame_num=1,
        size=(1024, 1024),
        negative_prompt=None,
        num_inference_steps=FLUX1_STEPS,
        sample_solver="flowmatch_euler",
        flexcache_params=PABParams(
            warmup=3,
            cooldown=3,
            skip_self_range=2,
            skip_cross_range=3,
        ),
    ),
    DiffusionUserParams(
        role="Alex",
        prompt=FLUX1_PROMPT,
        seed=42,
        frame_num=1,
        size=(1024, 1024),
        negative_prompt=None,
        num_inference_steps=FLUX1_STEPS,
        sample_solver="flowmatch_euler",
        flexcache_params=BlockDanceParams(
            warmup=3,
            cooldown=3,
            boundary_block=18,
            group_size=2,
            start_fraction=0.25,
            end_fraction=0.90,
        ),
    ),
    DiffusionUserParams(
        role="Alex",
        prompt=FLUX1_PROMPT,
        seed=42,
        frame_num=1,
        size=(1024, 1024),
        negative_prompt=None,
        num_inference_steps=FLUX1_STEPS,
        sample_solver="flowmatch_euler",
        flexcache_params=TaylorSeerParams(
            warmup=3,
            cooldown=3,
            fresh_threshold=3,
            max_order=1,
            first_enhance=1,
        ),
    ),
]


def gen_reqs(num_reqs: int) -> list[DiffusionUserRequest]:
    reqs: list[DiffusionUserRequest] = []
    for i in range(num_reqs):
        params = copy.deepcopy(msgs[i])
        request_id = os.getenv("CHITU_RUN_TASK_ID") if i == 0 else None
        reqs.append(
            DiffusionUserRequest(
                request_id=request_id or f"{gen_req_id()}",
                params=params,
            )
        )
    return reqs


def run_flux1_flexcache(args: ServeConfig, run_context: DiffusionTestRunContext):
    rank = torch.distributed.get_rank()
    run_output_dir = None

    reqs = []
    log_handler = None
    if rank == 0:
        reqs = gen_reqs(num_reqs=len(msgs))
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
            logger.info("Flux1 FlexCache requests: %s", reqs)

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
            logger.info("Flux1 FlexCache time cost %.3fs", elapsed_s)
        run_context.log_final_memory()
        run_context.dump_memory_snapshot(run_output_dir, "final")

        if rank == 0 and getattr(args.output, "timer", False):
            Timer.print_statistics()
            run_context.dump_timing_summary(run_output_dir, elapsed_s=elapsed_s)
    finally:
        if log_handler is not None:
            run_context.close_run_log_handler(log_handler)

    chitu_terminate()
    chitu_run_eval()
    if run_output_dir and rank == 0:
        write_flexcache_comparison(run_output_dir)
        logger.info("Flux1 FlexCache comparison written under %s/metrics", run_output_dir)


def main(args: ServeConfig):
    if args.models.name != "FLUX.1-dev":
        raise ValueError(f"test/test_flux1_flexcache.py expects models=FLUX.1-dev, got {args.models.name}.")

    logger.setLevel(logging.DEBUG)
    args.infer.diffusion.enable_flexcache = True
    args.models.sampler.sample_steps = FLUX1_STEPS

    run_context = DiffusionTestRunContext(args, logger, per_task_results=True, capture_stdio_log=True)
    initial_run_output_dir = run_context.build_initial_run_output_dir()
    run_context.activate_run(initial_run_output_dir)

    early_log_handler = None
    if getattr(args.output, "run_log", True) and should_record_metrics_on_rank():
        early_log_handler = run_context.attach_run_log_handler(initial_run_output_dir)
        os.environ["CHITU_RUN_LOG_ATTACHED"] = "1"

    if os.getenv("RANK") == "0":
        logger.info("Run Flux1 FlexCache with args: %s", args)

    chitu_init(args, logging_level=logging.INFO)
    logger.info("initialized chitu_diffusion.core.")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    try:
        run_flux1_flexcache(args, run_context)
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
