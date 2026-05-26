import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def slugify_run_part(value: Any, fallback: str = "run", max_len: int = 64) -> str:
    text = "" if value is None else str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-").lower()
    if not text:
        text = fallback
    return text[:max_len]


def build_run_output_dir(root_dir: str, tag: str | None, task_id: str, timestamp: str | None = None) -> str:
    root = str(root_dir or "outputs").strip() or "outputs"
    run_tag = slugify_run_part(tag, fallback="run", max_len=32)
    run_task_id = slugify_run_part(task_id, fallback="task", max_len=32)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(root, f"{run_tag}-{timestamp}-{run_task_id}")


def ensure_run_layout(run_output_dir: str) -> dict[str, str]:
    paths = {
        "root": run_output_dir,
        "results": os.path.join(run_output_dir, "results"),
        "metrics": os.path.join(run_output_dir, "metrics"),
        "timing_metrics": os.path.join(run_output_dir, "metrics", "timing"),
        "memory_metrics": os.path.join(run_output_dir, "metrics", "memory"),
        "quality_metrics": os.path.join(run_output_dir, "metrics", "quality"),
        "logs": os.path.join(run_output_dir, "logs"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def results_dir(run_output_dir: str) -> str:
    return os.path.join(run_output_dir, "results")


def task_results_dir(run_output_dir: str, task_id: str) -> str:
    return os.path.join(results_dir(run_output_dir), slugify_run_part(task_id, fallback="task", max_len=64))


def metrics_dir(run_output_dir: str) -> str:
    return os.path.join(run_output_dir, "metrics")


def timing_metrics_dir(run_output_dir: str) -> str:
    return os.path.join(metrics_dir(run_output_dir), "timing")


def memory_metrics_dir(run_output_dir: str) -> str:
    return os.path.join(metrics_dir(run_output_dir), "memory")


def quality_metrics_dir(run_output_dir: str) -> str:
    return os.path.join(run_output_dir, "metrics", "quality")


def logs_dir(run_output_dir: str) -> str:
    return os.path.join(run_output_dir, "logs")


def task_logs_dir(run_output_dir: str, task_id: str) -> str:
    return os.path.join(logs_dir(run_output_dir), slugify_run_part(task_id, fallback="task", max_len=64))


def debug_output_dir(run_output_dir: str) -> str:
    return logs_dir(run_output_dir)


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


def write_json(path: str, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_plain_data(payload), f, ensure_ascii=False, indent=2)


def append_json_list_item(path: str, key: str, item: Any, base_payload: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            payload = loaded

    if base_payload:
        for base_key, base_value in base_payload.items():
            payload.setdefault(base_key, to_plain_data(base_value))

    values = payload.setdefault(key, [])
    if not isinstance(values, list):
        values = []
        payload[key] = values
    values.append(to_plain_data(item))
    write_json(path, payload)
