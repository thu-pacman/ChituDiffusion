import torch
import time
import os
import random
import logging
import json
import copy
import re
import sys
import subprocess
from subprocess import TimeoutExpired
from datetime import datetime
from logging import getLogger
import resource

from omegaconf import OmegaConf

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
from chitu_diffusion.runtime.output_layout import (
    build_run_output_dir,
    ensure_run_layout,
    logs_dir,
    metrics_dir,
    results_dir,
    write_json,
)

from chitu_diffusion.core.schemas import ServeConfig
from chitu_diffusion.core.config_loader import load_config_from_cli
from chitu_diffusion.core.utils import gen_req_id
from chitu_diffusion.observability import Timer

logger = getLogger(__name__)

PROMPT_SLUG_LEN = 32

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
            cache_ratio=0.4,
            warmup=5,
            cooldown=5,
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


def _slugify(raw: str, max_len: int = PROMPT_SLUG_LEN) -> str:
    if raw is None:
        raw = ""
    text = str(raw).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if not text:
        text = "prompt"
    return text[:max_len]


def _build_run_output_dir(args, reqs: list[DiffusionUserRequest]) -> str:
    env_run_dir = os.getenv("CHITU_RUN_DIR", "").strip()
    if env_run_dir:
        return env_run_dir
    root_dir = str(getattr(args.output, "root_dir", "outputs") or "outputs")
    run_tag = str(os.getenv("CHITU_RUN_TAG", "")).strip() or "run"
    task_id = reqs[0].request_id if reqs else "task"
    return build_run_output_dir(
        root_dir=root_dir,
        tag=run_tag,
        task_id=task_id,
        timestamp=os.getenv("CHITU_RUN_TIMESTAMP"),
    )


def _build_initial_run_output_dir(args) -> str:
    env_run_dir = os.getenv("CHITU_RUN_DIR", "").strip()
    if env_run_dir:
        return env_run_dir
    root_dir = str(getattr(args.output, "root_dir", "outputs") or "outputs")
    run_tag = str(os.getenv("CHITU_RUN_TAG", "")).strip() or "run"
    task_id = os.getenv("CHITU_RUN_TASK_ID", "").strip() or "task"
    return build_run_output_dir(
        root_dir=root_dir,
        tag=run_tag,
        task_id=task_id,
        timestamp=os.getenv("CHITU_RUN_TIMESTAMP"),
    )


def _attach_run_log_handler(run_output_dir: str):
    os.makedirs(logs_dir(run_output_dir), exist_ok=True)
    rank = int(os.getenv("RANK", "0"))
    log_name = "run.log" if rank == 0 else f"run.rank{rank}.log"
    run_log_path = os.path.join(logs_dir(run_output_dir), log_name)

    # Mirror full process stdout/stderr to run.log while keeping console output.
    # This captures logging, print, tqdm/progress bars, and most native writes.
    tee_proc = subprocess.Popen(
        ["tee", "-a", run_log_path],
        stdin=subprocess.PIPE,
        stdout=sys.__stdout__,
        stderr=sys.__stderr__,
        text=False,
    )

    if tee_proc.stdin is None:
        raise RuntimeError("failed to initialize run.log tee stdin")

    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    os.dup2(tee_proc.stdin.fileno(), 1)
    os.dup2(tee_proc.stdin.fileno(), 2)

    class _RunLogCapture:
        def __init__(self, proc, out_fd, err_fd):
            self._proc = proc
            self._out_fd = out_fd
            self._err_fd = err_fd
            self._closed = False

        def close(self):
            if self._closed:
                return
            self._closed = True

            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass

            os.dup2(self._out_fd, 1)
            os.dup2(self._err_fd, 2)
            os.close(self._out_fd)
            os.close(self._err_fd)

            if self._proc.stdin is not None:
                self._proc.stdin.close()
            try:
                self._proc.wait(timeout=10)
            except TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=3)

    return _RunLogCapture(tee_proc, old_stdout_fd, old_stderr_fd)


def _dump_run_metadata(run_output_dir: str, args, reqs: list[DiffusionUserRequest]):
    ensure_run_layout(run_output_dir)
    req_items = []
    for req in reqs:
        req_items.append(
            {
                "request_id": req.request_id,
                "params": req.params,
            }
        )

    write_json(os.path.join(run_output_dir, "request_params.json"), {"requests": req_items})
    write_json(
        os.path.join(run_output_dir, "system_params.json"),
        {
            "model_name": str(getattr(args.models, "name", "")),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_requests": len(reqs),
            "output_dir": os.path.abspath(run_output_dir),
            "results_dir": os.path.abspath(results_dir(run_output_dir)),
            "config": args,
        },
    )

    with open(os.path.join(run_output_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(args, resolve=True))


def _dump_metrics_summary(run_output_dir: str, elapsed_s: float | None = None):
    payload = {
        "timing": {
            "csv": os.path.join("metrics", "timing.csv"),
            "json": os.path.join("metrics", "timing.json"),
            "overall_elapsed_s": elapsed_s,
        },
        "quality": {
            "summary": os.path.join("metrics", "quality", "summary.json"),
        },
        "memory": {},
    }
    if torch.cuda.is_available():
        payload["memory"] = {
            "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }
    write_json(os.path.join(metrics_dir(run_output_dir), "summary.json"), payload)


def run_normal(args):
    rank = torch.distributed.get_rank()
    warmup_diffusion_engine(args)

    for i in range(1):
        chitu_start()
        reqs = []
        run_output_dir = None
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=len(msgs),  # args.infer.max_reqs
                max_new_tokens=args.request.max_new_tokens,
                frequency_penalty=args.request.frequency_penalty,
                is_vl=hasattr(args.models, "vision_config"),
            )
            run_output_dir = _build_run_output_dir(args, reqs)
            ensure_run_layout(run_output_dir)
            run_results_dir = results_dir(run_output_dir)
            for req in reqs:
                req.params.save_dir = run_results_dir

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            payload = [run_output_dir]
            torch.distributed.broadcast_object_list(payload, src=0)
            run_output_dir = payload[0]

        if run_output_dir:
            os.environ["CHITU_CURRENT_OUTPUT_DIR"] = run_output_dir
            os.environ["CHITU_CURRENT_RESULTS_DIR"] = results_dir(run_output_dir)
            os.environ["CHITU_CURRENT_METRICS_DIR"] = metrics_dir(run_output_dir)
            os.environ["CHITU_CURRENT_LOGS_DIR"] = logs_dir(run_output_dir)

        log_handler = None
        if (
            run_output_dir
            and getattr(args.output, "enable_run_log", True)
            and os.getenv("CHITU_RUN_LOG_ATTACHED", "0") != "1"
        ):
            log_handler = _attach_run_log_handler(run_output_dir)

        if rank == 0:
            _dump_run_metadata(run_output_dir, args, reqs)
            logger.info(f"{reqs=}")
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
        logger.info(
            f"[Final] | GPU-Alloc:{torch.cuda.memory_allocated()/1024**3:.3f} Max:{torch.cuda.max_memory_allocated()/1024**3:.3f} Rsrv:{torch.cuda.memory_reserved()/1024**3:.3f} GB  | CPU:{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
        )

        if rank == 0 and getattr(args.output, "enable_timer_dump", False):
            Timer.print_statistics()
            Timer.save_statistics(os.path.join(metrics_dir(run_output_dir), "timing.csv"))
            Timer.save_statistics_json(os.path.join(metrics_dir(run_output_dir), "timing.json"))

        if rank == 0:
            _dump_metrics_summary(run_output_dir, elapsed_s=locals().get("elapsed_s"))

    chitu_terminate()
    chitu_run_eval()

    if log_handler is not None:
        log_handler.close()

def main(args: ServeConfig):
    global local_args
    local_args = args
    logger.setLevel(logging.DEBUG)
    initial_run_output_dir = _build_initial_run_output_dir(args)
    ensure_run_layout(initial_run_output_dir)
    os.environ["CHITU_CURRENT_OUTPUT_DIR"] = initial_run_output_dir
    os.environ["CHITU_CURRENT_RESULTS_DIR"] = results_dir(initial_run_output_dir)
    os.environ["CHITU_CURRENT_METRICS_DIR"] = metrics_dir(initial_run_output_dir)
    os.environ["CHITU_CURRENT_LOGS_DIR"] = logs_dir(initial_run_output_dir)

    early_log_handler = None
    if getattr(args.output, "enable_run_log", True):
        early_log_handler = _attach_run_log_handler(initial_run_output_dir)
        os.environ["CHITU_RUN_LOG_ATTACHED"] = "1"

    if os.getenv("RANK") == "0":
        logger.info(f"Run with args: {args}")

    # Initialize Backend: args / distributed / load models & kernels
    chitu_init(args, logging_level=logging.INFO)
    logger.info("initialized chitu_diffusion.core.")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    logger.debug("finish init")

    try:
        run_normal(args)
    finally:
        if early_log_handler is not None:
            early_log_handler.close()


if __name__ == "__main__":
    main(load_config_from_cli())

    # Sometimes torch.distributed will hang during destruction if CUDA graph is enabled.
    # As a workaround, we `exec` a dummy process to kill the current process, without
    # returning an error.
    logger.info("Waiting for all ranks to finish...")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    # Don't exec bash because it loads startup scripts
    os.execl("/usr/bin/true", "true")  # /usr/bin/true does nothing but exits
