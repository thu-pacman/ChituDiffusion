import argparse
import dataclasses
import errno
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import yaml
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = PROJECT_ROOT / "service_framework"
STATIC_ROOT = SERVICE_ROOT / "static"
RUNS_ROOT = SERVICE_ROOT / "runs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chitu_diffusion.flexcache.params import FLEXCACHE_PARAM_CLASSES
from chitu_diffusion.evaluation.eval_manager import EvalManager
from chitu_diffusion.runtime.output_layout import quality_metrics_dir, results_dir
from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskPool,
    DiffusionTaskStatus,
    DiffusionUserParams,
    DiffusionUserRequest,
)


CANCELLABLE_JOB_STATUSES = {"queued", "dispatching", "running", "cancelling"}
QUALITY_METRICS = {"vbench", "fid", "fvd", "psnr", "ssim", "lpips"}


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)
    os.replace(tmp, path)


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


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", errors="replace") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _worker_stage_from_log(log_text: str, ready: dict[str, Any]) -> dict[str, Any]:
    if ready.get("ready"):
        return {"stage": "ready", "message": ready.get("message") or "Model is loaded and ready."}
    if ready.get("status") == "failed":
        return {"stage": "failed", "message": ready.get("message") or "GPU worker failed."}

    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    recent = lines[-12:]
    joined = "\n".join(lines[-80:])
    command = next((line for line in reversed(lines) if line.startswith("Executing:")), "")

    if "srun: job" in joined and "queued and waiting for resources" in joined and "has been allocated resources" not in joined:
        message = next((line for line in reversed(lines) if "queued and waiting for resources" in line), "")
        return {"stage": "squeue", "message": message or "Slurm job is queued.", "command": command, "recent": recent}
    if "has been allocated resources" in joined and "Loading ChituDiffusion model" not in joined:
        return {"stage": "allocated", "message": "Slurm resources are allocated; worker process is starting.", "command": command, "recent": recent}
    if "Loading ChituDiffusion model" in joined and "Chitu has been initialized" not in joined:
        return {"stage": "building_model", "message": "Building/loading ChituDiffusion model.", "command": command, "recent": recent}
    if "Chitu has been initialized" in joined:
        return {"stage": "initializing_service", "message": "Chitu initialized; waiting for service readiness.", "command": command, "recent": recent}
    if command:
        return {"stage": "launching", "message": "Launching GPU worker through chitu run.", "command": command, "recent": recent}
    return {"stage": "loading", "message": ready.get("message") or "GPU worker is starting.", "recent": recent}


def _flexcache_schema() -> dict[str, list[dict[str, Any]]]:
    schema = {}
    for strategy, cls in sorted(FLEXCACHE_PARAM_CLASSES.items()):
        fields = []
        for field in dataclasses.fields(cls):
            if field.name == "strategy":
                continue
            default = None
            if field.default is not dataclasses.MISSING:
                default = field.default
            fields.append(
                {
                    "name": field.name,
                    "default": default,
                    "type": str(field.type),
                }
            )
        schema[strategy] = fields
    return schema


def _service_cp_size(cfg: dict[str, Any]) -> int:
    launch = cfg.get("launch", {}) or {}
    parallel = cfg.get("parallel", {}) or {}
    num_nodes = int(launch.get("num_nodes", 1) or 1)
    gpus_per_node = int(launch.get("gpus_per_node", 1) or 1)
    cfp = int(parallel.get("cfp", 1) or 1)
    total_gpus = num_nodes * gpus_per_node
    return max(1, total_gpus // max(1, cfp))


def _flexcache_strategy_available(strategy: str, cp_size: int) -> tuple[bool, str | None]:
    strategy = str(strategy or "").strip().lower()
    if strategy in {"", "origin", "none", "off", "disabled"}:
        return True, None
    if strategy == "ditango" and cp_size <= 1:
        return False, "DiTango requires cp world size > 1."
    if strategy == "cubic" and cp_size != 1:
        return False, "Cubic requires cp world size = 1."
    return True, None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _quality_metric_availability() -> dict[str, dict[str, Any]]:
    availability = {
        "psnr": {"available": _module_available("skimage"), "reason": None},
        "ssim": {"available": _module_available("skimage"), "reason": None},
        "lpips": {"available": _module_available("lpips"), "reason": None},
        "fid": {"available": _module_available("pytorch_fid"), "reason": None},
        "fvd": {
            "available": _module_available("torchmetrics") and _module_available("torch_fidelity"),
            "reason": None,
        },
        "vbench": {"available": _module_available("vbench"), "reason": None},
    }
    reasons = {
        "psnr": "scikit-image is not installed.",
        "ssim": "scikit-image is not installed.",
        "lpips": "lpips is not installed.",
        "fid": "pytorch-fid is not installed.",
        "fvd": "torchmetrics and torch-fidelity are required for FVD.",
        "vbench": "vbench is not installed.",
    }
    for metric, item in availability.items():
        if not item["available"]:
            item["reason"] = reasons[metric]
    return availability


def _request_defaults(model_name: str) -> dict[str, Any]:
    name = str(model_name or "")
    if name.startswith("Wan"):
        return {
            "role": "Alex",
            "prompt": "A cat walking on grass.",
            "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "seed": 42,
            "width": 832,
            "height": 480,
            "frame_num": 81,
            "num_inference_steps": 50,
            "sample_solver": "unipc",
        }
    if name == "FLUX.2-klein-4B":
        return {
            "role": "Alex",
            "prompt": "A cat holding a sign that says hello world",
            "negative_prompt": "",
            "seed": 42,
            "width": 1024,
            "height": 1024,
            "frame_num": 1,
            "num_inference_steps": None,
            "sample_solver": "flowmatch_euler",
        }
    return {
        "role": "Alex",
        "prompt": "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background.",
        "negative_prompt": "",
        "seed": 42,
        "width": 1024,
        "height": 1024,
        "frame_num": 1,
        "num_inference_steps": None,
        "sample_solver": "flowmatch_euler",
    }


class ServiceState:
    def __init__(self, config_path: Path, host: str, port: int, auto_port: bool):
        self.base_config_path = config_path.resolve()
        self.base_config = _read_yaml(self.base_config_path)
        self.host = host
        self.port = port
        self.auto_port = auto_port
        self.run_dir = RUNS_ROOT / f"service_{time.strftime('%Y%m%d_%H%M%S')}"
        self.jobs_dir = self.run_dir / "jobs"
        self.ready_path = self.run_dir / "ready.json"
        self.stop_path = self.run_dir / "stop.json"
        self.service_log = self.run_dir / "service.log"
        self.config_path = self.run_dir / "system_config.yaml"
        self.proc: subprocess.Popen | None = None
        self.worker_log_handle = None
        self.shutdown_lock = threading.Lock()
        self.eval_lock = threading.Lock()
        self.shutdown_started = False

    def prepare(self) -> None:
        cfg = dict(self.base_config)
        cfg.setdefault("launch", {})
        cfg.setdefault("infer", {})
        cfg["launch"]["python_script"] = "service_framework/persistent_service.py"
        _write_yaml(self.config_path, cfg)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            self.ready_path,
            {
                "ready": False,
                "status": "loading",
                "message": "GPU worker is starting and loading the model.",
                "created_at": time.time(),
            },
        )

    def launch_worker(self) -> None:
        env = os.environ.copy()
        env["CHITU_SERVICE_RUN_DIR"] = str(self.run_dir)
        env["CHITU_SERVICE_CONFIG"] = str(self.config_path)
        chitu_cmd = ["chitu"]
        raw_cli_cmd = env.get("CHITU_CLI_CMD", "").strip()
        if raw_cli_cmd:
            try:
                parsed_cli_cmd = json.loads(raw_cli_cmd)
            except json.JSONDecodeError:
                parsed_cli_cmd = []
            if isinstance(parsed_cli_cmd, list) and all(isinstance(item, str) for item in parsed_cli_cmd):
                chitu_cmd = parsed_cli_cmd
        cmd = [*chitu_cmd, "run", str(self.config_path)]
        self.service_log.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(self.service_log, "a", encoding="utf-8", errors="replace")
        self.worker_log_handle = log_f
        log_f.write("Executing: " + " ".join(cmd) + "\n")
        log_f.flush()
        self.proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def request_stop(self, reason: str) -> None:
        _write_json(
            self.stop_path,
            {
                "stop": True,
                "reason": reason,
                "requested_at": time.time(),
            },
        )
        ready = self.ready()
        ready.update(
            {
                "ready": False,
                "status": "stopping",
                "message": f"Service shutdown requested: {reason}",
                "updated_at": time.time(),
            }
        )
        _write_json(self.ready_path, ready)

    def _send_worker_control(self, payload: dict[str, Any], timeout_s: float = 3.0) -> dict[str, Any]:
        ready = _read_json(self.ready_path, None)
        control = ready.get("control_endpoint") if isinstance(ready, dict) else None
        if not isinstance(control, dict):
            raise RuntimeError("worker control endpoint is not ready yet")
        port = int(control.get("port") or 0)
        token = str(control.get("token") or "")
        hosts = []
        for key in ("host", "fqdn", "master_addr"):
            value = str(control.get(key) or "").strip()
            if value and value not in hosts:
                hosts.append(value)
        if not hosts or port <= 0 or not token:
            raise RuntimeError("worker control endpoint is invalid")

        request = dict(payload)
        request["token"] = token
        data = (json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8")
        errors = []
        for host in hosts:
            try:
                with socket.create_connection((host, port), timeout=timeout_s) as sock:
                    sock.settimeout(timeout_s)
                    sock.sendall(data)
                    chunks = []
                    while True:
                        chunk = sock.recv(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        if b"\n" in chunk:
                            break
                raw = b"".join(chunks).splitlines()[0]
                response = json.loads(raw.decode("utf-8"))
                if not isinstance(response, dict):
                    raise RuntimeError("worker control endpoint returned a non-object response")
                if not response.get("ok"):
                    raise RuntimeError(str(response.get("error") or "worker rejected control request"))
                return response
            except Exception as exc:
                errors.append(f"{host}:{port}: {exc}")
        raise RuntimeError("failed to reach worker control endpoint; " + "; ".join(errors))

    def request_cancel(self, job_id: str | None = None, reason: str = "requested from web UI") -> dict[str, Any]:
        if job_id:
            job = _read_json(self.job_path(job_id), None)
            if not isinstance(job, dict):
                raise ValueError("job not found")
            status = str(job.get("status") or "")
            if status not in CANCELLABLE_JOB_STATUSES:
                raise ValueError(f"job is already {status or 'not running'} and cannot be cancelled")

        payload = {
            "action": "cancel",
            "job_id": job_id,
            "reason": reason,
            "requested_at": time.time(),
        }
        response = self._send_worker_control(payload)
        _append_text(
            self.service_log,
            f"HTTP cancel request forwarded to worker: job_id={job_id} reason={reason} at={time.time()}",
        )
        if job_id:
            current = _read_json(self.job_path(job_id), None)
            if isinstance(current, dict) and str(current.get("status") or "") in CANCELLABLE_JOB_STATUSES:
                current["status"] = "cancelling"
                current["error"] = reason
                _write_json(self.job_path(job_id), current)
        return {"request": payload, "worker": response}

    def shutdown_worker(self, reason: str = "service shutdown", graceful_timeout_s: float = 30.0) -> None:
        with self.shutdown_lock:
            if self.shutdown_started:
                return
            self.shutdown_started = True
            self.request_stop(reason)
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is not None:
            self._close_worker_log()
            return
        deadline = time.time() + graceful_timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                self._close_worker_log()
                return
            time.sleep(0.2)
        self._signal_worker_group(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._signal_worker_group(signal.SIGKILL)
            proc.wait(timeout=10)
        finally:
            self._close_worker_log()

    def _signal_worker_group(self, sig: signal.Signals) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            return

    def _close_worker_log(self) -> None:
        if self.worker_log_handle is not None:
            self.worker_log_handle.close()
            self.worker_log_handle = None

    def ready(self) -> dict[str, Any]:
        data = _read_json(self.ready_path, {}) or {}
        if self.proc is not None and self.proc.poll() is not None and not data.get("ready"):
            data.update(
                {
                    "ready": False,
                    "status": "failed",
                    "message": f"GPU worker exited with code {self.proc.returncode}.",
                }
            )
        return data

    def static_summary(self) -> dict[str, Any]:
        cfg = self.base_config
        ready = self.ready()
        service_log_text = _tail(self.service_log)
        model = cfg.get("model", {})
        model_name = model.get("name", "")
        cp_size = _service_cp_size(cfg)
        strategies = ["origin", *sorted(FLEXCACHE_PARAM_CLASSES.keys())]
        availability = {}
        for strategy in strategies:
            available, reason = _flexcache_strategy_available(strategy, cp_size)
            availability[strategy] = {"available": available, "reason": reason}
        return {
            "config_path": str(self.config_path),
            "ready": bool(ready.get("ready")),
            "worker_status": ready,
            "worker_stage": _worker_stage_from_log(service_log_text, ready),
            "service_log": str(self.service_log),
            "launch": cfg.get("launch", {}),
            "model": model,
            "request_defaults": _request_defaults(model_name),
            "parallel": cfg.get("parallel", {}),
            "infer": cfg.get("infer", {}),
            "derived": {"cp_size": cp_size},
            "flexcache": {
                "strategies": strategies,
                "schema": _flexcache_schema(),
                "availability": availability,
            },
            "quality": {
                "metrics": ["fid", "fvd", "psnr", "ssim", "lpips", "vbench"],
                "availability": _quality_metric_availability(),
            },
            "output": cfg.get("output", {}),
        }

    def job_path(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "job.json"

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.ready().get("ready"):
            raise RuntimeError("GPU worker is not ready yet.")
        request = payload.get("request") or {}
        if not isinstance(request, dict):
            raise ValueError("payload must contain object field: request")
        output = request.get("output")
        if output is None:
            output = payload.get("output") or {}
            request["output"] = output
        if not isinstance(output, dict):
            raise ValueError("request.output must be an object")
        if not str(request.get("prompt") or "").strip():
            raise ValueError("request.prompt is required")
        flexcache_params = request.get("flexcache_params") if isinstance(request.get("flexcache_params"), dict) else {}
        strategy = str(flexcache_params.get("strategy") or request.get("flexcache_strategy") or "").strip().lower()
        available, reason = _flexcache_strategy_available(strategy, _service_cp_size(self.base_config))
        if not available:
            raise ValueError(reason or f"FlexCache strategy '{strategy}' is not available for this instance.")

        job_id = uuid.uuid4().hex[:12]
        request_id = _safe_id(str(request.get("request_id") or job_id))
        request["request_id"] = request_id
        job = {
            "job_id": job_id,
            "request_id": request_id,
            "status": "queued",
            "returncode": None,
            "created_at": time.time(),
            "kind": "generation",
            "request": request,
            "run_dir": str(self.jobs_dir / job_id),
            "request_path": str(self.jobs_dir / job_id / "request.json"),
            "output_dir": None,
            "error": None,
        }
        job_dir = self.jobs_dir / job_id
        _write_json(job_dir / "request.json", request)
        _write_json(self.job_path(job_id), job)
        return self.job_to_dict(job)

    def submit_evaluation(self, test_task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        test_task_id = str(test_task_id or "").strip()
        reference_task_id = str(payload.get("reference_task_id") or "").strip()
        metrics = payload.get("metrics") or []
        if not test_task_id or not reference_task_id:
            raise ValueError("reference_task_id is required")
        if not isinstance(metrics, list):
            raise ValueError("metrics must be a list")
        eval_types = []
        for metric in metrics:
            name = str(metric or "").strip().lower()
            if not name:
                continue
            if name not in QUALITY_METRICS:
                raise ValueError(f"unsupported quality metric: {name}")
            if name not in eval_types:
                eval_types.append(name)
        if not eval_types:
            raise ValueError("select at least one quality metric")
        metric_availability = _quality_metric_availability()
        unavailable = [name for name in eval_types if not metric_availability.get(name, {}).get("available")]
        if unavailable:
            details = "; ".join(
                f"{name}: {metric_availability.get(name, {}).get('reason') or 'unavailable'}"
                for name in unavailable
            )
            raise ValueError(f"selected eval metrics are unavailable: {details}")
        test_job = self.find_job(test_task_id)
        reference_job = self.find_job(reference_task_id)
        if test_job is None:
            raise ValueError(f"test task not found: {test_task_id}")
        if reference_job is None:
            raise ValueError(f"reference task not found: {reference_task_id}")
        if not test_job.get("output_dir"):
            raise ValueError(f"test task has no output_dir yet: {test_task_id}")
        if not reference_job.get("output_dir"):
            raise ValueError(f"reference task has no output_dir yet: {reference_task_id}")
        if str(test_job.get("status") or "") not in {"completed", "failed", "cancelled", "stopped"}:
            raise ValueError(f"test task is not finished yet: {test_task_id}")

        path = self.job_path(test_job["job_id"])
        current = _read_json(path, None)
        if not isinstance(current, dict):
            raise ValueError("test job not found")
        eval_state = {
            "status": "running",
            "reference_task_id": reference_task_id,
            "reference_job_id": reference_job["job_id"],
            "metrics": eval_types,
            "started_at": time.time(),
            "summary_path": str(Path(test_job["output_dir"]) / "metrics" / "quality" / "summary.json"),
            "error": None,
        }
        current["eval"] = eval_state
        _write_json(path, current)
        threading.Thread(
            target=self._run_cpu_evaluation,
            args=(current["job_id"], reference_job["job_id"], eval_types),
            daemon=True,
        ).start()
        return self.job_to_dict(current)

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for path in self.jobs_dir.glob("*/job.json"):
            job = _read_json(path, None)
            if isinstance(job, dict):
                jobs.append(self.job_to_dict(job))
        return sorted(jobs, key=lambda item: item.get("created_at", 0), reverse=True)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        job = _read_json(self.job_path(job_id), None)
        return self.job_to_dict(job) if isinstance(job, dict) else None

    def find_job(self, task_id: str) -> dict[str, Any] | None:
        job = self.get_job(task_id)
        if job is not None:
            return job
        for candidate in self.list_jobs():
            if str(candidate.get("request_id") or "") == task_id:
                return candidate
        return None

    def _run_cpu_evaluation(self, test_job_id: str, reference_job_id: str, metrics: list[str]) -> None:
        test_path = self.job_path(test_job_id)
        try:
            test_job = _read_json(test_path, None)
            reference_job = _read_json(self.job_path(reference_job_id), None)
            if not isinstance(test_job, dict) or not isinstance(reference_job, dict):
                raise ValueError("evaluation job disappeared")
            with self.eval_lock:
                result = _run_cpu_quality_eval(
                    self.base_config,
                    test_job,
                    reference_job,
                    metrics,
                )
            updated = _read_json(test_path, None)
            if isinstance(updated, dict):
                updated["eval"] = {
                    **(updated.get("eval") if isinstance(updated.get("eval"), dict) else {}),
                    "status": "completed",
                    "completed_at": time.time(),
                    "result": result,
                    "summary_path": str(Path(updated["output_dir"]) / "metrics" / "quality" / "summary.json"),
                    "error": None,
                }
                _write_json(test_path, updated)
        except Exception as exc:
            updated = _read_json(test_path, None)
            if isinstance(updated, dict):
                updated["eval"] = {
                    **(updated.get("eval") if isinstance(updated.get("eval"), dict) else {}),
                    "status": "failed",
                    "completed_at": time.time(),
                    "error": str(exc),
                }
                _write_json(test_path, updated)
            _append_text(self.service_log, f"CPU eval failed for job_id={test_job_id}: {exc}")

    def job_to_dict(self, job: dict[str, Any]) -> dict[str, Any]:
        item = dict(job)
        item["preview"] = _job_preview(item)
        item["timing"] = _job_timing(item)
        item["memory"] = _job_memory(item)
        item["quality"] = _job_quality(item)
        item["strategy_summary"] = _job_strategy_summary(item)
        return item


def _job_preview_path(job: dict[str, Any]) -> tuple[Path, Path, str] | None:
    output_dir = job.get("output_dir")
    if not output_dir:
        return None
    root = Path(output_dir).resolve()
    if not root.is_dir():
        return None
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm"}
    candidates = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    path = candidates[0]
    kind = "video" if path.suffix.lower() in {".mp4", ".webm"} else "image"
    return root, path, kind


def _job_preview(job: dict[str, Any]) -> dict[str, str] | None:
    preview = _job_preview_path(job)
    if preview is None:
        return None
    root, path, kind = preview
    rel = path.relative_to(root).as_posix()
    return {
        "kind": kind,
        "path": rel,
        "url": f"/api/jobs/{job['job_id']}/files/{rel}",
        "download_url": f"/api/jobs/{job['job_id']}/download/result",
    }


def _job_timing(job: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = job.get("output_dir")
    if not output_dir:
        return None
    summary = _read_json(Path(output_dir) / "metrics" / "timing" / "summary.json", None)
    if not isinstance(summary, dict):
        return None
    timers = summary.get("timers") or {}
    dit = timers.get("dit_forward") or timers.get("dit_forward_step") or {}
    overall_ms = None
    if isinstance(summary.get("overall_elapsed_s"), (int, float)):
        overall_ms = float(summary["overall_elapsed_s"]) * 1000.0
    elif isinstance((timers.get("overall") or {}).get("total_ms"), (int, float)):
        overall_ms = float(timers["overall"]["total_ms"])
    return {
        "e2e_ms": overall_ms,
        "dit_forward_total_ms": dit.get("total_ms"),
        "dit_forward_avg_ms": dit.get("avg_ms"),
        "dit_forward_samples": dit.get("samples"),
    }


def _job_memory(job: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = job.get("output_dir")
    if not output_dir:
        return None
    memory_dir = Path(output_dir) / "metrics" / "memory"
    if not memory_dir.is_dir():
        return None

    rank_summaries = []
    peak_gpu_allocated_gb = None
    peak_gpu_reserved_gb = None
    peak_cpu_rss_gb = None
    peak_flexcache_gb = None
    for path in sorted(memory_dir.glob("rank*.json")):
        data = _read_json(path, None)
        if not isinstance(data, dict):
            continue
        rank = data.get("rank")
        rank_peak_allocated = None
        rank_peak_reserved = None
        rank_peak_cpu = None
        rank_peak_flexcache = None
        events = data.get("events") or []
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            allocated = event.get("gpu_max_allocated_gb", event.get("gpu_allocated_gb"))
            reserved = event.get("gpu_reserved_gb")
            cpu_rss = event.get("cpu_max_rss_gb")
            flexcache = event.get("flexcache_peak_cache_gb", event.get("flexcache_cache_gb"))
            if isinstance(allocated, (int, float)):
                rank_peak_allocated = max(rank_peak_allocated or 0.0, float(allocated))
                peak_gpu_allocated_gb = max(peak_gpu_allocated_gb or 0.0, float(allocated))
            if isinstance(reserved, (int, float)):
                rank_peak_reserved = max(rank_peak_reserved or 0.0, float(reserved))
                peak_gpu_reserved_gb = max(peak_gpu_reserved_gb or 0.0, float(reserved))
            if isinstance(cpu_rss, (int, float)):
                rank_peak_cpu = max(rank_peak_cpu or 0.0, float(cpu_rss))
                peak_cpu_rss_gb = max(peak_cpu_rss_gb or 0.0, float(cpu_rss))
            if isinstance(flexcache, (int, float)):
                rank_peak_flexcache = max(rank_peak_flexcache or 0.0, float(flexcache))
                peak_flexcache_gb = max(peak_flexcache_gb or 0.0, float(flexcache))
        if rank_peak_allocated is not None or rank_peak_reserved is not None or rank_peak_flexcache is not None:
            rank_summaries.append(
                {
                    "rank": rank,
                    "peak_gpu_allocated_gb": rank_peak_allocated,
                    "peak_gpu_reserved_gb": rank_peak_reserved,
                    "peak_cpu_rss_gb": rank_peak_cpu,
                    "peak_flexcache_gb": rank_peak_flexcache,
                }
            )

    if (
        peak_gpu_allocated_gb is None
        and peak_gpu_reserved_gb is None
        and peak_cpu_rss_gb is None
        and peak_flexcache_gb is None
    ):
        return None
    return {
        "peak_gpu_allocated_gb": peak_gpu_allocated_gb,
        "peak_gpu_reserved_gb": peak_gpu_reserved_gb,
        "peak_cpu_rss_gb": peak_cpu_rss_gb,
        "peak_flexcache_gb": peak_flexcache_gb,
        "ranks": rank_summaries,
    }


def _job_quality(job: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = job.get("output_dir")
    if not output_dir:
        return None
    summary = _read_json(Path(output_dir) / "metrics" / "quality" / "summary.json", None)
    return summary if isinstance(summary, dict) else None


def _build_eval_args(base_config: dict[str, Any], reference_output_dir: str, metrics: list[str]) -> SimpleNamespace:
    model = base_config.get("model", {}) or {}
    eval_cfg = SimpleNamespace(eval_type=metrics, reference_path=reference_output_dir)
    models = SimpleNamespace(name=str(model.get("name") or ""))
    return SimpleNamespace(eval=eval_cfg, models=models)


def _run_cpu_quality_eval(
    base_config: dict[str, Any],
    test_job: dict[str, Any],
    reference_job: dict[str, Any],
    metrics: list[str],
) -> dict[str, Any]:
    test_output_dir = str(test_job.get("output_dir") or "").strip()
    reference_output_dir = str(reference_job.get("output_dir") or "").strip()
    request_params = _read_json(Path(test_output_dir) / "request_params.json", None)
    requests = request_params.get("requests") if isinstance(request_params, dict) else None
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"test output has no request_params.json requests: {test_output_dir}")
    req_payload = requests[0]
    params_payload = req_payload.get("params") if isinstance(req_payload, dict) else None
    if not isinstance(params_payload, dict):
        raise ValueError(f"test output request params are invalid: {test_output_dir}")

    params = DiffusionUserParams(**params_payload)
    request_id = str(req_payload.get("request_id") or test_job.get("request_id") or test_job.get("job_id"))
    req = DiffusionUserRequest(request_id=request_id, params=params)
    req.params.save_dir = os.path.join(results_dir(test_output_dir), req.request_id)
    task = DiffusionTask(task_id=req.request_id, req=req)
    task.status = DiffusionTaskStatus.Completed

    old_env = {
        key: os.environ.get(key)
        for key in (
            "CHITU_CURRENT_OUTPUT_DIR",
            "CHITU_CURRENT_RESULTS_DIR",
            "CHITU_CURRENT_METRICS_DIR",
            "CHITU_CURRENT_LOGS_DIR",
        )
    }
    try:
        os.environ["CHITU_CURRENT_OUTPUT_DIR"] = test_output_dir
        os.environ["CHITU_CURRENT_RESULTS_DIR"] = results_dir(test_output_dir)
        os.environ["CHITU_CURRENT_METRICS_DIR"] = os.path.join(test_output_dir, "metrics")
        os.environ["CHITU_CURRENT_LOGS_DIR"] = os.path.join(test_output_dir, "logs")
        DiffusionTaskPool.reset()
        DiffusionTaskPool.add(task)
        manager = EvalManager()
        return manager.run(
            args=_build_eval_args(base_config, reference_output_dir, metrics),
            eval_types=metrics,
            output_dir=quality_metrics_dir(test_output_dir),
        )
    finally:
        DiffusionTaskPool.reset()
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _job_strategy_summary(job: dict[str, Any]) -> str:
    request = job.get("request") or {}
    params_dict = request.get("flexcache_params") if isinstance(request.get("flexcache_params"), dict) else {}
    strategy = str(params_dict.get("strategy") or request.get("flexcache_strategy") or "").strip().lower()
    if strategy in {"", "origin", "none", "off", "disabled"}:
        return "origin"
    if strategy not in FLEXCACHE_PARAM_CLASSES:
        return strategy
    params = []
    for field in dataclasses.fields(FLEXCACHE_PARAM_CLASSES[strategy]):
        key = field.name
        if key == "strategy":
            continue
        value = params_dict.get(key, request.get(key))
        if value not in (None, ""):
            params.append(f"{key}={value}")
    if not params:
        return strategy
    return f"{strategy}({', '.join(params[:4])})"


def _json_response(handler: SimpleHTTPRequestHandler, code: int, data: Any) -> None:
    body = json.dumps(data, ensure_ascii=True, indent=2).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_bytes(
    handler: SimpleHTTPRequestHandler,
    data: bytes,
    content_type: str,
    filename: str | None = None,
) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    if filename:
        safe_filename = Path(filename).name or "download"
        handler.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(safe_filename)}")
    handler.end_headers()
    handler.wfile.write(data)


def _file_response(
    handler: SimpleHTTPRequestHandler,
    job: dict[str, Any],
    rel_path: str,
    download: bool = False,
) -> None:
    output_dir = job.get("output_dir")
    if not output_dir:
        handler.send_error(404, "job has no output directory")
        return
    root = Path(output_dir).resolve()
    target = (root / unquote(rel_path)).resolve()
    if target != root and root not in target.parents:
        handler.send_error(403, "path escapes job output directory")
        return
    if not target.is_file():
        handler.send_error(404, "file not found")
        return
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }
    data = target.read_bytes()
    filename = f"{job.get('request_id') or job.get('job_id')}{target.suffix}" if download else None
    _send_bytes(handler, data, content_types.get(target.suffix.lower(), "application/octet-stream"), filename)


def _download_log_response(handler: SimpleHTTPRequestHandler, job: dict[str, Any], service_log: Path) -> None:
    text = _job_log(job, service_log)
    filename = f"{job.get('request_id') or job.get('job_id')}.log"
    _send_bytes(handler, text.encode("utf-8"), "text/plain; charset=utf-8", filename)


def _download_result_response(handler: SimpleHTTPRequestHandler, job: dict[str, Any]) -> None:
    preview = _job_preview_path(job)
    if preview is None:
        handler.send_error(404, "job has no preview result")
        return
    root, path, _kind = preview
    rel = path.relative_to(root).as_posix()
    _file_response(handler, job, rel, download=True)


def _shutdown_server(server: ThreadingHTTPServer, state: ServiceState, reason: str) -> None:
    state.shutdown_worker(reason)
    server.shutdown()


class Handler(SimpleHTTPRequestHandler):
    state: ServiceState

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        rel = unquote(parsed.path.lstrip("/")) or "index.html"
        return str((STATIC_ROOT / rel).resolve())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            return _json_response(self, 200, self.state.static_summary())
        if parsed.path == "/api/jobs":
            return _json_response(self, 200, {"jobs": self.state.list_jobs()})
        if parsed.path == "/api/service/log":
            return _json_response(self, 200, {"text": _tail(self.state.service_log)})
        if parsed.path.startswith("/api/jobs/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 3:
                job = self.state.get_job(parts[2])
                if job is None:
                    return _json_response(self, 404, {"error": "job not found"})
                if len(parts) == 4 and parts[3] == "log":
                    return _json_response(self, 200, {"text": _job_log(job, self.state.service_log)})
                if len(parts) == 5 and parts[3] == "download" and parts[4] == "log":
                    return _download_log_response(self, job, self.state.service_log)
                if len(parts) == 5 and parts[3] == "download" and parts[4] == "result":
                    return _download_result_response(self, job)
                if len(parts) == 4 and parts[3] == "cancel":
                    return _json_response(self, 405, {"error": "use POST to cancel a job"})
                if len(parts) >= 5 and parts[3] == "files":
                    return _file_response(self, job, "/".join(parts[4:]))
                return _json_response(self, 200, job)
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/service/shutdown":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                reason = str(payload.get("reason") or "requested from web UI")
                self.state.request_stop(reason)
                threading.Thread(
                    target=_shutdown_server,
                    args=(self.server, self.state, reason),
                    daemon=True,
                ).start()
                return _json_response(self, 202, {"ok": True, "message": "service shutdown started"})
            except Exception as exc:
                return _json_response(self, 400, {"error": str(exc)})
        if parsed.path.startswith("/api/jobs/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 4 and parts[3] == "cancel":
                try:
                    job = self.state.get_job(parts[2])
                    if job is None:
                        return _json_response(self, 404, {"error": "job not found"})
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                    reason = str(payload.get("reason") or "requested from web UI")
                    cancel = self.state.request_cancel(parts[2], reason)
                    return _json_response(self, 202, {"ok": True, "cancel": cancel})
                except Exception as exc:
                    return _json_response(self, 400, {"error": str(exc)})
            if len(parts) == 4 and parts[3] == "eval":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                    job = self.state.submit_evaluation(parts[2], payload)
                    return _json_response(self, 202, job)
                except RuntimeError as exc:
                    return _json_response(self, 409, {"error": str(exc)})
                except Exception as exc:
                    return _json_response(self, 400, {"error": str(exc)})
        if parsed.path != "/api/jobs":
            return _json_response(self, 404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            job = self.state.submit(payload)
            return _json_response(self, 202, job)
        except RuntimeError as exc:
            return _json_response(self, 409, {"error": str(exc)})
        except Exception as exc:
            return _json_response(self, 400, {"error": str(exc)})


def _job_log(job: dict[str, Any], service_log: Path) -> str:
    output_dir = job.get("output_dir")
    if output_dir:
        run_log = Path(output_dir) / "logs" / "run.log"
        if run_log.is_file():
            return _tail(run_log)
    return _tail(service_log)


def _serve(state: ServiceState) -> None:
    Handler.state = state
    server = None
    last_error = None
    ports = range(state.port, state.port + 20) if state.auto_port else [state.port]
    for port in ports:
        try:
            server = ThreadingHTTPServer((state.host, port), Handler)
            state.port = port
            break
        except OSError as exc:
            last_error = exc
            if exc.errno != errno.EADDRINUSE or not state.auto_port:
                break
    if server is None:
        if last_error and last_error.errno == errno.EADDRINUSE:
            raise SystemExit(f"Port {state.port} is already in use. Choose another --port or add --auto-port.")
        raise last_error or RuntimeError("failed to start HTTP server")

    print(f"Service UI: http://{state.host}:{state.port}", flush=True)
    print("VSCode Remote should offer to open/forward this URL.", flush=True)
    print(f"If your local browser cannot load it, manually forward remote port {state.port} in VSCode Ports.", flush=True)
    print(f"Remote health check: curl http://127.0.0.1:{state.port}/api/config", flush=True)
    print("The page will show loading until the GPU worker finishes model initialization.", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ChituDiffusion web UI plus a persistent GPU worker")
    parser.add_argument("--config", default="system_config.yaml", help="base system config")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host on the VSCode remote host")
    parser.add_argument("--port", type=int, default=7860, help="HTTP bind port on the VSCode remote host")
    parser.add_argument("--auto-port", action="store_true", help="try later ports if the requested port is busy")
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    state = ServiceState(PROJECT_ROOT / args.config, args.host, args.port, args.auto_port)
    state.prepare()
    print("Starting GPU worker through chitu run.", flush=True)
    print(f"Generated service config: {state.config_path}", flush=True)
    print(f"Worker log: {state.service_log}", flush=True)
    state.launch_worker()
    try:
        _serve(state)
    except KeyboardInterrupt:
        print("\nCtrl+C received; shutting down GPU worker and Slurm allocation.", flush=True)
        state.shutdown_worker("server received Ctrl+C")
    finally:
        state.shutdown_worker("server exiting", graceful_timeout_s=5.0)


if __name__ == "__main__":
    main(parse_args())
