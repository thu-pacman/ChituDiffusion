import copy
import logging
import os
import time
from logging import getLogger

import torch

from chitu_diffusion.core.config_loader import load_config_from_cli
from chitu_diffusion.core.schemas import ServeConfig
from chitu_diffusion.flexcache.params import BlockDanceParams, CubicParams, PABParams, TaylorSeerParams, TeaCacheParams
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
FLUX1_STEPS = int(os.getenv("CHITU_FLUX1_BENCH_STEPS", "50"))
FLUX1_BASE_SEED = int(os.getenv("CHITU_FLUX1_BENCH_BASE_SEED", "42"))
FLUX1_NUM_SEEDS = int(os.getenv("CHITU_FLUX1_BENCH_NUM_SEEDS", "3"))
FLUX1_WARMUP_RUNS = int(os.getenv("CHITU_FLUX1_BENCH_WARMUP_RUNS", "1"))
FLUX1_SIZE = (1024, 1024)


def _strategy_warmup_cooldown() -> tuple[int, int]:
    if FLUX1_STEPS <= 10:
        return 1, 1
    return 3, 3


STRATEGY_WARMUP, STRATEGY_COOLDOWN = _strategy_warmup_cooldown()


FLEXCACHE_STRATEGY_PARAMS = [
    TeaCacheParams(
        warmup=STRATEGY_WARMUP,
        cooldown=STRATEGY_COOLDOWN,
        teacache_thresh=0.6,
        coefficients=None,
        use_ref_steps=True,
    ),
    PABParams(
        warmup=STRATEGY_WARMUP,
        cooldown=STRATEGY_COOLDOWN,
        skip_self_range=2,
        skip_cross_range=3,
    ),
    BlockDanceParams(
        warmup=STRATEGY_WARMUP,
        cooldown=STRATEGY_COOLDOWN,
        boundary_block=18,
        group_size=2,
        start_fraction=0.25,
        end_fraction=0.90,
    ),
    TaylorSeerParams(
        warmup=STRATEGY_WARMUP,
        cooldown=STRATEGY_COOLDOWN,
        fresh_threshold=3,
        max_order=1,
        first_enhance=1,
    ),
    CubicParams(
        warmup=STRATEGY_WARMUP,
        cooldown=STRATEGY_COOLDOWN,
        target_speedup=2.0,
        tau_max=8,
    ),
]


def _selected_strategy_names() -> set[str] | None:
    raw = os.getenv("CHITU_FLUX1_BENCH_STRATEGIES", "").strip().lower()
    if not raw:
        return None
    names = {item.strip() for item in raw.replace(";", ",").replace(" ", ",").split(",") if item.strip()}
    return names or None


def _selected_strategy_params():
    selected = _selected_strategy_names()
    if selected is None:
        return FLEXCACHE_STRATEGY_PARAMS
    params = [item for item in FLEXCACHE_STRATEGY_PARAMS if _strategy_name(item) in selected]
    missing = selected - {_strategy_name(item) for item in params}
    if missing:
        raise ValueError(f"Unknown CHITU_FLUX1_BENCH_STRATEGIES items: {sorted(missing)}")
    return params


def _seeds() -> list[int]:
    return [FLUX1_BASE_SEED + i for i in range(FLUX1_NUM_SEEDS)]


def _base_params(seed: int, *, role: str = "Alex") -> DiffusionUserParams:
    return DiffusionUserParams(
        role=role,
        prompt=FLUX1_PROMPT,
        seed=seed,
        frame_num=1,
        size=FLUX1_SIZE,
        negative_prompt=None,
        num_inference_steps=FLUX1_STEPS,
        sample_solver="flowmatch_euler",
        flexcache_params=None,
    )


def _request_id(strategy: str, seed: int, *, warmup_idx: int | None = None) -> str:
    if warmup_idx is not None:
        return f"warmup_{strategy}_seed{seed}_{warmup_idx}"
    return f"{strategy}_seed{seed}"


def _strategy_name(params) -> str:
    return str(getattr(params, "strategy", None) or "none").strip().lower()


def gen_origin_reqs() -> list[DiffusionUserRequest]:
    reqs: list[DiffusionUserRequest] = []
    seeds = _seeds()
    for warmup_idx in range(FLUX1_WARMUP_RUNS):
        reqs.append(
            DiffusionUserRequest(
                request_id=_request_id("origin", seeds[0], warmup_idx=warmup_idx),
                params=_base_params(seeds[0], role="warmup"),
            )
        )
    reqs.extend([
        DiffusionUserRequest(
            request_id=_request_id("origin", seed),
            params=_base_params(seed),
        )
        for seed in seeds
    ])
    return reqs


def gen_flexcache_reqs() -> list[DiffusionUserRequest]:
    reqs: list[DiffusionUserRequest] = []
    seeds = _seeds()

    for warmup_idx in range(FLUX1_WARMUP_RUNS):
        warmup_params = _base_params(seeds[0], role="warmup")
        warmup_params.flexcache_params = copy.deepcopy(FLEXCACHE_STRATEGY_PARAMS[0])
        warmup_strategy = _strategy_name(warmup_params.flexcache_params)
        reqs.append(
            DiffusionUserRequest(
                request_id=_request_id(warmup_strategy, seeds[0], warmup_idx=warmup_idx),
                params=warmup_params,
            )
        )

    for strategy_params in _selected_strategy_params():
        strategy = _strategy_name(strategy_params)
        for seed in seeds:
            params = _base_params(seed)
            params.flexcache_params = copy.deepcopy(strategy_params)
            reqs.append(
                DiffusionUserRequest(
                    request_id=_request_id(strategy, seed),
                    params=params,
                )
            )
    return reqs


def _benchmark_mode(args: ServeConfig) -> str:
    mode = os.getenv("CHITU_FLUX1_BENCH_MODE", "").strip().lower()
    if mode:
        return mode
    reference_path = str(getattr(args.eval, "reference_path", "") or "").strip()
    return "flexcache" if reference_path else "origin"


def _build_reqs(mode: str) -> list[DiffusionUserRequest]:
    if mode == "origin":
        return gen_origin_reqs()
    if mode == "flexcache":
        return gen_flexcache_reqs()
    raise ValueError(f"Unsupported CHITU_FLUX1_BENCH_MODE={mode!r}; expected origin or flexcache.")


def run_flux1_benchmark(args: ServeConfig, run_context: DiffusionTestRunContext, mode: str):
    rank = torch.distributed.get_rank()
    run_output_dir = None

    reqs = []
    log_handler = None
    if rank == 0:
        reqs = _build_reqs(mode)
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
            logger.info("Flux1 %s benchmark requests: %s", mode, reqs)

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
            logger.info("Flux1 %s benchmark time cost %.3fs", mode, elapsed_s)
        run_context.log_final_memory()
        run_context.dump_memory_snapshot(run_output_dir, "final")

        if rank == 0 and getattr(args.output, "timer", False):
            Timer.print_statistics()
            run_context.dump_timing_summary(run_output_dir, elapsed_s=elapsed_s)
    finally:
        if log_handler is not None:
            run_context.close_run_log_handler(log_handler)

    chitu_terminate()
    if mode == "flexcache":
        chitu_run_eval()
    if run_output_dir and rank == 0:
        write_flexcache_comparison(run_output_dir)
        logger.info("Flux1 benchmark comparison written under %s/metrics", run_output_dir)


def main(args: ServeConfig):
    if args.models.name != "Flux1-dev":
        raise ValueError(f"test/test_flux1_flexcache.py expects models=Flux1-dev, got {args.models.name}.")

    logger.setLevel(logging.DEBUG)
    mode = _benchmark_mode(args)
    args.models.sampler.sample_steps = FLUX1_STEPS

    if mode == "flexcache":
        reference_path = str(getattr(args.eval, "reference_path", "") or "").strip()
        if not reference_path:
            raise ValueError("Flux1 flexcache benchmark requires eval.reference_path.")
        args.eval.reference_path = reference_path
    else:
        args.eval.eval_type = []
        args.eval.reference_path = None

    run_context = DiffusionTestRunContext(args, logger, per_task_results=True, capture_stdio_log=True)
    initial_run_output_dir = run_context.build_initial_run_output_dir()
    run_context.activate_run(initial_run_output_dir)

    early_log_handler = None
    if getattr(args.output, "run_log", True) and should_record_metrics_on_rank():
        early_log_handler = run_context.attach_run_log_handler(initial_run_output_dir)
        os.environ["CHITU_RUN_LOG_ATTACHED"] = "1"

    if os.getenv("RANK") == "0":
        logger.info("Run Flux1 %s benchmark with args: %s", mode, args)

    chitu_init(args, logging_level=logging.INFO)
    logger.info("initialized chitu_diffusion.core.")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    try:
        run_flux1_benchmark(args, run_context, mode)
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
