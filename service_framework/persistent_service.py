import json
import logging
import os
import queue
import re
import signal
import socket
import sys
import threading
import time
import uuid
import dataclasses
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = SERVICE_ROOT / "runs"
TEST_DIR = PROJECT_ROOT / "test"
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from chitu_diffusion.core.config_loader import load_config_from_cli
from chitu_diffusion.core.schemas import ServeConfig
from chitu_diffusion.flexcache.params import FLEXCACHE_PARAM_CLASSES
from chitu_diffusion.observability import Timer
from chitu_diffusion.runtime.main import (
    chitu_generate,
    chitu_init,
    chitu_run_eval,
    chitu_start,
    chitu_is_terminated,
)
from chitu_diffusion.runtime.output_layout import (
    build_run_output_dir,
    write_json,
)
from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskPool,
    DiffusionUserParams,
    DiffusionUserRequest,
)
from run_context import DiffusionTestRunContext, should_record_metrics_on_rank


logger = logging.getLogger(__name__)


def _handle_shutdown_signal(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def _safe_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-")
    return value[:80] or uuid.uuid4().hex[:8]


def _tail(path: Path, max_bytes: int = 65536) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)
    os.replace(tmp, path)


class WorkerControlServer:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.token = uuid.uuid4().hex
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", 0))
        sock.listen(16)
        sock.settimeout(0.5)
        self._socket = sock
        host = socket.gethostname()
        port = int(sock.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, name="chitu-service-control", daemon=True)
        self._thread.start()
        logger.info("Worker control endpoint listening on %s:%s", host, port)

    def endpoint(self) -> dict[str, Any]:
        if self._socket is None:
            raise RuntimeError("worker control endpoint has not been started")
        return {
            "host": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "master_addr": os.getenv("MASTER_ADDR", ""),
            "port": int(self._socket.getsockname()[1]),
            "token": self.token,
            "created_at": time.time(),
        }

    def stop(self) -> None:
        self._stop.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass

    def drain(self) -> list[dict[str, Any]]:
        messages = []
        while True:
            try:
                messages.append(self.messages.get_nowait())
            except queue.Empty:
                return messages

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                assert self._socket is not None
                conn, addr = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                try:
                    data = b""
                    conn.settimeout(2.0)
                    while b"\n" not in data and len(data) < 1048576:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                    message = json.loads(data.decode("utf-8").strip())
                    if not isinstance(message, dict):
                        raise ValueError("control message must be a JSON object")
                    if message.pop("token", None) != self.token:
                        raise PermissionError("invalid control token")
                    action = str(message.get("action") or "")
                    if action != "cancel":
                        raise ValueError(f"unsupported control action: {action}")
                    message["received_at"] = time.time()
                    message["remote_addr"] = addr[0] if addr else None
                    self.messages.put(message)
                    conn.sendall(b'{"ok": true}\n')
                except Exception as exc:
                    logger.warning("Rejected worker control request: %s", exc)
                    response = {"ok": False, "error": str(exc)}
                    conn.sendall((json.dumps(response, ensure_ascii=True) + "\n").encode("utf-8"))


def _tuple_size(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError("size must be a two-item list, for example [1024, 1024]")


def _apply_output_options(args: ServeConfig, output: dict[str, Any]) -> None:
    for key in ("root_dir", "run_log", "memory", "timer", "log_ranks"):
        if key in output:
            setattr(args.output, key, output[key])
    if bool(getattr(args.output, "timer", False)):
        Timer.enable()
    else:
        Timer.disable()
    if "log_ranks" in output:
        value = output["log_ranks"]
        if isinstance(value, str):
            os.environ["CHITU_LOG_RANKS"] = value.strip().lower() or "0"
        else:
            os.environ["CHITU_LOG_RANKS"] = ",".join(str(int(item)) for item in value)


def _reset_request_observability() -> None:
    Timer.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _job_output_options(job: dict[str, Any]) -> dict[str, Any]:
    request = job.get("request") if isinstance(job.get("request"), dict) else {}
    output = request.get("output") if isinstance(request.get("output"), dict) else job.get("output")
    return output if isinstance(output, dict) else {}


def _build_request(args: ServeConfig, request: dict[str, Any]) -> DiffusionUserRequest:
    request_id = _safe_id(str(request.get("request_id") or uuid.uuid4().hex[:8]))
    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    sample_solver = request.get("sample_solver")
    if not sample_solver:
        sample_solver = "flowmatch_euler" if str(args.models.name).startswith("FLUX") else "ddpm"

    flexcache_params = request.get("flexcache_params")
    if not flexcache_params and request.get("flexcache_strategy"):
        strategy = str(request.get("flexcache_strategy") or "").strip().lower()
        if strategy not in {"", "origin", "none", "off", "disabled"}:
            if strategy not in FLEXCACHE_PARAM_CLASSES:
                raise ValueError(f"unsupported flexcache strategy: {strategy}")
            flexcache_params = {"strategy": strategy}
            for field in dataclasses.fields(FLEXCACHE_PARAM_CLASSES[strategy]):
                key = field.name
                if key == "strategy":
                    continue
                if request.get(key) not in (None, ""):
                    flexcache_params[key] = request[key]

    params = DiffusionUserParams(
        role=str(request.get("role") or "user"),
        size=_tuple_size(request.get("size", [512, 512])),
        frame_num=int(request.get("frame_num", 1)),
        prompt=prompt,
        negative_prompt=request.get("negative_prompt") or None,
        seed=None if request.get("seed") in (None, "") else int(request.get("seed")),
        sample_solver=str(sample_solver),
        num_inference_steps=(
            int(request["num_inference_steps"])
            if request.get("num_inference_steps") not in (None, "")
            else args.models.sampler.sample_steps
        ),
        flexcache_params=flexcache_params or None,
    )
    return DiffusionUserRequest(request_id=request_id, params=params)


def _build_service_output_dir(args: ServeConfig, task_id: str) -> str:
    return build_run_output_dir(
        root_dir=str(getattr(args.output, "root_dir", "outputs") or "outputs"),
        tag=os.getenv("CHITU_RUN_TAG", "").strip() or "service",
        task_id=task_id,
    )


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _broadcast_control(control: dict[str, Any] | None) -> dict[str, Any]:
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        payload = [control]
        torch.distributed.broadcast_object_list(payload, src=0)
        return payload[0]
    if control is None:
        raise RuntimeError("rank0 control cannot be None in single-process mode")
    return control


def _cancel_queued_job(jobs_dir: Path, job_id: str, reason: str) -> bool:
    if not job_id:
        return False
    path = jobs_dir / job_id / "job.json"
    job = _read_json(path, None)
    if not isinstance(job, dict):
        return False
    if job.get("status") in {"completed", "failed", "cancelled", "stopped"}:
        return False
    if job.get("status") not in {"queued", "dispatching", "cancelling"}:
        return False
    job["status"] = "cancelled"
    job["returncode"] = 130
    job["error"] = reason
    _write_json_file(path, job)
    logger.info("Cancelled queued job_id=%s: %s", job_id, reason)
    return True


def _run_one_job(
    args: ServeConfig,
    run_context: DiffusionTestRunContext,
    job: dict[str, Any],
    control: dict[str, Any],
    control_server: WorkerControlServer | None,
) -> None:
    rank = _rank()
    run_dir = Path(os.getenv("CHITU_SERVICE_RUN_DIR", str(RUNS_ROOT / "service")))
    stop_path = run_dir / "stop.json"
    jobs_dir = run_dir / "jobs"
    cancelled = False
    cancel_reason = None
    _apply_output_options(args, control["output"])
    run_output_dir = control["run_output_dir"]
    run_context.activate_run(run_output_dir)

    log_handler = None
    if (
        getattr(args.output, "run_log", True)
        and should_record_metrics_on_rank()
        and os.getenv("CHITU_RUN_LOG_ATTACHED", "0") != "1"
    ):
        log_handler = run_context.attach_run_log_handler(run_output_dir)

    try:
        if rank == 0:
            req = _build_request(args, control["request"])
            req.request_id = control["request_id"]
            reqs = [req]
            run_context.apply_request_output_dirs(run_output_dir, reqs)
            run_context.dump_run_metadata(run_output_dir, reqs)
            write_json(os.path.join(run_output_dir, "service_request.json"), control["request"])
            _write_json_file(Path(job["request_path"]), control["request"])
            DiffusionTaskPool.reset()
            DiffusionTaskPool.add(DiffusionTask(task_id=req.request_id, req=req))
            job["status"] = "running"
            job["request_id"] = req.request_id
            job["output_dir"] = run_output_dir
            _write_json_file(Path(job["job_path"]), job)

        _reset_request_observability()
        run_context.dump_memory_snapshot(run_output_dir, "model_loaded")
        chitu_start()
        start = time.time()
        with Timer.get_timer("overall"):
            while True:
                if rank == 0:
                    stop = _read_stop_request(stop_path)
                    if stop is not None:
                        DiffusionTaskPool.request_shutdown(str(stop.get("reason") or "service shutdown requested"))
                    if stop is None and control_server is not None:
                        for message in control_server.drain():
                            if message.get("action") != "cancel":
                                continue
                            target_job_id = str(message.get("job_id") or "")
                            message_reason = str(message.get("reason") or "Current generation cancelled")
                            if target_job_id in ("", str(job.get("job_id") or "")):
                                cancel_reason = message_reason
                                current_task = DiffusionTaskPool.pool.get(control["request_id"])
                                current_step = None
                                if current_task is not None and getattr(current_task, "buffer", None) is not None:
                                    current_step = getattr(current_task.buffer, "current_step", None)
                                logger.info(
                                    "Received worker control cancel for job_id=%s request_id=%s at current_step=%s: %s",
                                    job.get("job_id"),
                                    control["request_id"],
                                    current_step,
                                    cancel_reason,
                                )
                                DiffusionTaskPool.request_cancel(cancel_reason)
                            else:
                                _cancel_queued_job(jobs_dir, target_job_id, message_reason)
                chitu_generate()
                terminated = chitu_is_terminated()
                done = DiffusionTaskPool.all_finished() if rank == 0 else False
                if rank == 0 and not terminated and done:
                    current_task = DiffusionTaskPool.pool.get(control["request_id"])
                    cancelled = current_task is not None and current_task.status.name == "Failed"
                    if cancelled and cancel_reason is None:
                        cancel_reason = current_task.error_message or "Current generation cancelled"
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    done_payload = [done or terminated, cancelled, cancel_reason]
                    torch.distributed.broadcast_object_list(done_payload, src=0)
                    done = bool(done_payload[0])
                    cancelled = bool(done_payload[1])
                    cancel_reason = done_payload[2]
                else:
                    done = done or terminated
                if done:
                    break

        elapsed_s = time.time() - start
        run_context.log_final_memory()
        run_context.dump_memory_snapshot(run_output_dir, "final")
        if rank == 0 and getattr(args.output, "timer", False):
            Timer.print_statistics()
            run_context.dump_timing_summary(run_output_dir, elapsed_s=elapsed_s)
        if rank == 0:
            if chitu_is_terminated():
                job["status"] = "stopped"
                job["returncode"] = 130
                job["error"] = "Service shutdown requested during generation."
            elif cancelled:
                job["status"] = "cancelled"
                job["returncode"] = 130
                job["error"] = cancel_reason or "Current generation cancelled."
            else:
                job["status"] = "completed"
                job["returncode"] = 0
            _write_json_file(Path(job["job_path"]), job)
        if not chitu_is_terminated() and not cancelled:
            chitu_run_eval()
    except Exception as exc:
        if rank == 0:
            job["status"] = "failed"
            job["returncode"] = 1
            job["error"] = str(exc)
            _write_json_file(Path(job["job_path"]), job)
        raise
    finally:
        if log_handler is not None:
            run_context.close_run_log_handler(log_handler)


def _claim_next_job(jobs_dir: Path) -> dict[str, Any] | None:
    jobs = []
    for path in jobs_dir.glob("*/job.json"):
        job = _read_json(path, None)
        if isinstance(job, dict) and job.get("status") == "queued":
            jobs.append((float(job.get("created_at", 0)), path, job))
    if not jobs:
        return None
    _, path, job = sorted(jobs, key=lambda item: item[0])[0]
    job["status"] = "dispatching"
    job["job_path"] = str(path)
    _write_json_file(path, job)
    return job


def _read_stop_request(stop_path: Path) -> dict[str, Any] | None:
    data = _read_json(stop_path, None)
    if isinstance(data, dict) and data.get("stop"):
        return data
    return None


def _service_loop(
    args: ServeConfig,
    run_context: DiffusionTestRunContext,
    jobs_dir: Path,
    stop_path: Path,
    control_server: WorkerControlServer | None,
) -> None:
    rank = _rank()
    while True:
        if rank == 0:
            job = None
            while job is None:
                stop = _read_stop_request(stop_path)
                if stop is not None:
                    control = {
                        "action": "stop",
                        "reason": str(stop.get("reason") or "service shutdown requested"),
                    }
                    break
                if control_server is not None:
                    for message in control_server.drain():
                        if message.get("action") != "cancel":
                            continue
                        target_job_id = str(message.get("job_id") or "")
                        reason = str(message.get("reason") or "Current generation cancelled")
                        _cancel_queued_job(jobs_dir, target_job_id, reason)
                job = _claim_next_job(jobs_dir)
                if job is None:
                    time.sleep(0.5)
            if job is not None:
                try:
                    req = _build_request(args, job["request"])
                    job["request_id"] = req.request_id
                    output = _job_output_options(job)
                    _apply_output_options(args, output)
                    run_output_dir = _build_service_output_dir(args, req.request_id)
                    control = {
                        "action": "run",
                        "job_id": job["job_id"],
                        "request_id": req.request_id,
                        "request": {**job["request"], "request_id": req.request_id},
                        "output": output,
                        "run_output_dir": run_output_dir,
                        "job": job,
                    }
                except Exception as exc:
                    job["status"] = "failed"
                    job["returncode"] = 1
                    job["error"] = str(exc)
                    _write_json_file(Path(job["job_path"]), job)
                    continue
        else:
            job = None
            control = None

        control = _broadcast_control(control)
        if control.get("action") == "stop":
            if rank == 0:
                logger.info("Stopping persistent service worker: %s", control.get("reason", "service shutdown"))
                DiffusionTaskPool.request_shutdown(str(control.get("reason") or "service shutdown"))
            chitu_generate()
            break
        if control.get("action") == "run":
            job = control["job"]
            _run_one_job(args, run_context, job, control, control_server)


def main(args: ServeConfig) -> None:
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    logging.getLogger("chitu_diffusion").setLevel(logging.INFO)
    rank = int(os.getenv("RANK", "0"))

    run_context = DiffusionTestRunContext(args, logger, per_task_results=True, capture_stdio_log=True)
    initial_run_output_dir = run_context.build_initial_run_output_dir()
    run_context.activate_run(initial_run_output_dir)

    early_log_handler = None
    if getattr(args.output, "run_log", True) and should_record_metrics_on_rank():
        early_log_handler = run_context.attach_run_log_handler(initial_run_output_dir)
        os.environ["CHITU_RUN_LOG_ATTACHED"] = "1"

    print("Loading ChituDiffusion model before accepting HTTP requests...", flush=True)
    chitu_init(args, logging_level=logging.INFO)

    run_dir = Path(os.getenv("CHITU_SERVICE_RUN_DIR", str(RUNS_ROOT / "service")))
    jobs_dir = run_dir / "jobs"
    ready_path = run_dir / "ready.json"
    stop_path = run_dir / "stop.json"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    control_server = None
    if rank == 0:
        control_server = WorkerControlServer(run_dir)
        control_server.start()
        _write_json_file(
            ready_path,
            {
                "ready": True,
                "status": "ready",
                "message": "Model is loaded and the worker is waiting for requests.",
                "updated_at": time.time(),
                "model": str(getattr(args.models, "name", "")),
                "control": "tcp",
                "control_endpoint": control_server.endpoint(),
            },
        )
        print("ChituDiffusion worker is ready for UI requests.", flush=True)

    try:
        _service_loop(args, run_context, jobs_dir, stop_path, control_server)
    finally:
        if control_server is not None:
            control_server.stop()
        if rank == 0:
            _write_json_file(
                ready_path,
                {
                    "ready": False,
                    "status": "stopped",
                    "message": "Worker has stopped.",
                    "updated_at": time.time(),
                    "model": str(getattr(args.models, "name", "")),
                },
            )
        if early_log_handler is not None:
            run_context.close_run_log_handler(early_log_handler)


if __name__ == "__main__":
    try:
        main(load_config_from_cli())
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
