#!/usr/bin/env python3
import json
import logging
import os
import time
from logging import getLogger
from pathlib import Path

import torch

from chitu_diffusion.core.config_loader import load_config_from_cli
from chitu_diffusion.core.schemas import ServeConfig
from chitu_diffusion.observability import Timer
from chitu_diffusion.runtime.main import (
    chitu_generate,
    chitu_init,
    chitu_is_terminated,
    chitu_start,
    chitu_terminate,
    warmup_diffusion_engine,
)
from chitu_diffusion.runtime.task import DiffusionTask, DiffusionTaskPool, DiffusionUserParams, DiffusionUserRequest
from test.run_context import DiffusionTestRunContext, should_record_metrics_on_rank


logger = getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "ChituBench" / "prompts" / "flux1_attention.json"


def _load_prompt_items(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"prompt file must contain a list: {path}")
    out = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict) or not item.get("prompt"):
            raise ValueError(f"invalid prompt item #{idx} in {path}")
        out.append({"id": str(item.get("id") or f"prompt{idx}"), "prompt": str(item["prompt"])})
    return out


def _seeds() -> list[int]:
    base_seed = int(os.getenv("CHITUBENCH_BASE_SEED", "42"))
    num_seeds = int(os.getenv("CHITUBENCH_NUM_SEEDS", "3"))
    return [base_seed + i for i in range(num_seeds)]


def _steps(args: ServeConfig) -> int:
    return int(os.getenv("CHITUBENCH_STEPS", str(args.models.sampler.sample_steps)))


def _size() -> tuple[int, int]:
    raw = os.getenv("CHITUBENCH_IMAGE_SIZE", "1024,1024")
    width, height = [int(item.strip()) for item in raw.lower().replace("x", ",").split(",", 1)]
    return width, height


def build_requests(args: ServeConfig) -> list[DiffusionUserRequest]:
    case_id = os.getenv("CHITUBENCH_CASE_ID", "").strip() or str(args.infer.attn_type)
    prompt_file = Path(os.getenv("CHITUBENCH_PROMPT_FILE", str(DEFAULT_PROMPT_FILE))).resolve()
    prompt_items = _load_prompt_items(prompt_file)
    seeds = _seeds()
    steps = _steps(args)
    size = _size()
    requests = []

    warmup_runs = int(os.getenv("CHITUBENCH_WARMUP_RUNS", "1"))
    for warmup_idx in range(warmup_runs):
        first = prompt_items[0]
        requests.append(
            DiffusionUserRequest(
                request_id=f"warmup_{case_id}_{first['id']}_seed{seeds[0]}_{warmup_idx}",
                params=DiffusionUserParams(
                    role="warmup",
                    prompt=first["prompt"],
                    seed=seeds[0],
                    frame_num=1,
                    size=size,
                    negative_prompt=None,
                    num_inference_steps=steps,
                    sample_solver="flowmatch_euler",
                    flexcache_params=None,
                ),
            )
        )

    for prompt_item in prompt_items:
        for seed in seeds:
            requests.append(
                DiffusionUserRequest(
                    request_id=f"{case_id}_{prompt_item['id']}_seed{seed}",
                    params=DiffusionUserParams(
                        role=case_id,
                        prompt=prompt_item["prompt"],
                        seed=seed,
                        frame_num=1,
                        size=size,
                        negative_prompt=None,
                        num_inference_steps=steps,
                        sample_solver="flowmatch_euler",
                        flexcache_params=None,
                    ),
                )
            )
    return requests


def run_benchmark(args: ServeConfig, run_context: DiffusionTestRunContext):
    rank = torch.distributed.get_rank()
    run_output_dir = None
    reqs = []
    log_handler = None

    if rank == 0:
        reqs = build_requests(args)
        run_output_dir = run_context.build_run_output_dir(reqs)
        run_context.activate_run(run_output_dir)
        run_context.apply_request_output_dirs(run_output_dir, reqs)

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        payload = [run_output_dir]
        torch.distributed.broadcast_object_list(payload, src=0)
        run_output_dir = payload[0]

    if run_output_dir:
        run_context.activate_run(run_output_dir)

    if run_output_dir and getattr(args.output, "run_log", True) and os.getenv("CHITU_RUN_LOG_ATTACHED", "0") != "1":
        log_handler = run_context.attach_run_log_handler(run_output_dir)

    try:
        if rank == 0:
            run_context.dump_run_metadata(run_output_dir, reqs)
            logger.info("ChituBench Flux requests: %s", reqs)

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
            logger.info("ChituBench Flux benchmark time cost %.3fs", elapsed_s)
        run_context.log_final_memory()
        run_context.dump_memory_snapshot(run_output_dir, "final")

        if rank == 0 and getattr(args.output, "timer", False):
            Timer.print_statistics()
            run_context.dump_timing_summary(run_output_dir, elapsed_s=elapsed_s)
    finally:
        if log_handler is not None:
            run_context.close_run_log_handler(log_handler)

    chitu_terminate()


def main(args: ServeConfig):
    if args.models.name not in {"Flux1-dev", "Flux2-klein-4B"}:
        raise ValueError(f"flux_image_benchmark expects a Flux image model, got {args.models.name}")
    logger.setLevel(logging.DEBUG)
    args.models.sampler.sample_steps = _steps(args)

    run_context = DiffusionTestRunContext(args, logger, per_task_results=True, capture_stdio_log=True)
    initial_run_output_dir = run_context.build_initial_run_output_dir()
    run_context.activate_run(initial_run_output_dir)

    early_log_handler = None
    if getattr(args.output, "run_log", True) and should_record_metrics_on_rank():
        early_log_handler = run_context.attach_run_log_handler(initial_run_output_dir)
        os.environ["CHITU_RUN_LOG_ATTACHED"] = "1"

    chitu_init(args, logging_level=logging.INFO)
    logger.info("initialized chitu_diffusion.core.")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    try:
        run_benchmark(args, run_context)
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
