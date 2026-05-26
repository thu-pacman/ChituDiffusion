import logging
import os
import resource
import subprocess
import sys
from datetime import datetime
from subprocess import TimeoutExpired
from typing import Iterable

import torch
from omegaconf import OmegaConf

from chitu_diffusion.observability import Timer
from chitu_diffusion.core.logging_utils import should_log_on_rank
from chitu_diffusion.runtime.output_layout import (
    append_json_list_item,
    build_run_output_dir,
    ensure_run_layout,
    logs_dir,
    memory_metrics_dir,
    metrics_dir,
    quality_metrics_dir,
    results_dir,
    task_results_dir,
    timing_metrics_dir,
    write_json,
)
from chitu_diffusion.runtime.task import DiffusionUserRequest


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.getenv("RANK", "0"))


def should_record_metrics_on_rank(rank: int | None = None) -> bool:
    return should_log_on_rank(_rank() if rank is None else rank)


def should_record_memory(args) -> bool:
    return bool(getattr(args.output, "memory", True))


def _format_log_ranks(value) -> str:
    if isinstance(value, str):
        return value.strip().lower() or "0"
    if OmegaConf.is_list(value):
        return ",".join(str(int(item)) for item in value)
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(item)) for item in value)
    return "0"


def _memory_payload(stage: str, task_id: str | None = None) -> dict:
    payload = {
        "stage": stage,
        "rank": _rank(),
        "cpu_max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    if task_id is not None:
        payload["task_id"] = task_id
    if torch.cuda.is_available():
        payload.update(
            {
                "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            }
        )
    return payload


class _TeeRunLog:
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


class DiffusionTestRunContext:
    def __init__(
        self,
        args,
        logger: logging.Logger,
        *,
        per_task_results: bool = False,
        capture_stdio_log: bool = False,
    ):
        self.args = args
        self.logger = logger
        self.per_task_results = per_task_results
        self.capture_stdio_log = capture_stdio_log
        os.environ.setdefault("CHITU_LOG_RANKS", _format_log_ranks(getattr(args.output, "log_ranks", [0])))

    def build_initial_run_output_dir(self) -> str:
        env_run_dir = os.getenv("CHITU_RUN_DIR", "").strip()
        if env_run_dir:
            return env_run_dir
        return self._build_run_output_dir(os.getenv("CHITU_RUN_TASK_ID", "").strip() or "task")

    def build_run_output_dir(self, reqs: list[DiffusionUserRequest]) -> str:
        env_run_dir = os.getenv("CHITU_RUN_DIR", "").strip()
        if env_run_dir:
            return env_run_dir
        task_id = reqs[0].request_id if reqs else "task"
        return self._build_run_output_dir(task_id)

    def _build_run_output_dir(self, task_id: str) -> str:
        root_dir = str(getattr(self.args.output, "root_dir", "outputs") or "outputs")
        run_tag = str(os.getenv("CHITU_RUN_TAG", "")).strip() or "run"
        return build_run_output_dir(
            root_dir=root_dir,
            tag=run_tag,
            task_id=task_id,
            timestamp=os.getenv("CHITU_RUN_TIMESTAMP"),
        )

    def activate_run(self, run_output_dir: str) -> None:
        ensure_run_layout(run_output_dir)
        os.environ["CHITU_CURRENT_OUTPUT_DIR"] = run_output_dir
        os.environ["CHITU_CURRENT_RESULTS_DIR"] = results_dir(run_output_dir)
        os.environ["CHITU_CURRENT_METRICS_DIR"] = metrics_dir(run_output_dir)
        os.environ["CHITU_CURRENT_LOGS_DIR"] = logs_dir(run_output_dir)

    def apply_request_output_dirs(self, run_output_dir: str, reqs: Iterable[DiffusionUserRequest]) -> None:
        run_results_dir = results_dir(run_output_dir)
        for req in reqs:
            if self.per_task_results:
                req.params.save_dir = task_results_dir(run_output_dir, req.request_id)
            else:
                req.params.save_dir = run_results_dir

    def attach_run_log_handler(self, run_output_dir: str):
        if not should_record_metrics_on_rank():
            return None
        os.makedirs(logs_dir(run_output_dir), exist_ok=True)
        rank = _rank()
        log_name = "run.log" if rank == 0 else f"run.rank{rank}.log"
        run_log_path = os.path.join(logs_dir(run_output_dir), log_name)

        if self.capture_stdio_log:
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
            return _TeeRunLog(tee_proc, old_stdout_fd, old_stderr_fd)

        handler = logging.FileHandler(run_log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        return handler

    def close_run_log_handler(self, handler) -> None:
        if handler is None:
            return
        if hasattr(handler, "close") and not isinstance(handler, logging.Handler):
            handler.close()
            return
        logging.getLogger().removeHandler(handler)
        handler.close()

    def dump_run_metadata(self, run_output_dir: str, reqs: list[DiffusionUserRequest]) -> None:
        ensure_run_layout(run_output_dir)
        write_json(
            os.path.join(run_output_dir, "request_params.json"),
            {
                "requests": [
                    {
                        "request_id": req.request_id,
                        "params": req.params,
                    }
                    for req in reqs
                ]
            },
        )
        write_json(
            os.path.join(run_output_dir, "system_params.json"),
            {
                "model_name": str(getattr(self.args.models, "name", "")),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "num_requests": len(reqs),
                "output_dir": os.path.abspath(run_output_dir),
                "results_dir": os.path.abspath(results_dir(run_output_dir)),
                "metrics_dirs": {
                    "timing": os.path.abspath(timing_metrics_dir(run_output_dir)),
                    "memory": os.path.abspath(memory_metrics_dir(run_output_dir)),
                    "quality": os.path.abspath(quality_metrics_dir(run_output_dir)),
                },
                "config": self.args,
            },
        )

        with open(os.path.join(run_output_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(self.args, resolve=True))

    def dump_timing_summary(self, run_output_dir: str, elapsed_s: float | None = None) -> None:
        write_json(
            os.path.join(timing_metrics_dir(run_output_dir), "summary.json"),
            {
                "overall_elapsed_s": elapsed_s,
                "timers": Timer.statistics_dict(),
                "records": Timer.records_dict(),
            },
        )

    def dump_memory_snapshot(
        self,
        run_output_dir: str,
        stage: str,
        *,
        filename: str | None = None,
        task_id: str | None = None,
    ) -> None:
        if not should_record_memory(self.args):
            return
        rank = _rank()
        if not should_record_metrics_on_rank(rank):
            return
        filename = filename or f"rank{rank}.json"
        append_json_list_item(
            os.path.join(memory_metrics_dir(run_output_dir), filename),
            "events",
            _memory_payload(stage, task_id=task_id),
            base_payload={"rank": rank},
        )

    def log_final_memory(self) -> None:
        if not should_record_memory(self.args):
            return
        if not should_record_metrics_on_rank():
            return
        if torch.cuda.is_available():
            self.logger.info(
                f"[Final] | GPU-Alloc:{torch.cuda.memory_allocated()/1024**3:.3f} "
                f"Max:{torch.cuda.max_memory_allocated()/1024**3:.3f} "
                f"Rsrv:{torch.cuda.memory_reserved()/1024**3:.3f} GB  | "
                f"CPU:{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
            )
        else:
            self.logger.info(
                f"[Final] | CPU:{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
            )
