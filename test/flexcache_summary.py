import json
import os
from typing import Any


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _logs_dir(run_output_dir: str) -> str:
    return os.path.join(run_output_dir, "logs")


def _metrics_dir(run_output_dir: str) -> str:
    return os.path.join(run_output_dir, "metrics")


def _timing_metrics_dir(run_output_dir: str) -> str:
    return os.path.join(_metrics_dir(run_output_dir), "timing")


def _memory_metrics_dir(run_output_dir: str) -> str:
    return os.path.join(_metrics_dir(run_output_dir), "memory")


def _quality_metrics_dir(run_output_dir: str) -> str:
    return os.path.join(_metrics_dir(run_output_dir), "quality")


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _strategy_label(params: dict[str, Any]) -> str:
    spec = params.get("flexcache_params") or {}
    strategy = spec.get("strategy") or params.get("flexcache") or "none"
    if strategy in {"teacache", "pab"}:
        core = spec.get("baseline_params") or {}
    else:
        core = {
            "cache_ratio": spec.get("cache_ratio"),
            "warmup": spec.get("warmup"),
            "cooldown": spec.get("cooldown"),
            "tau_max": spec.get("tau_max"),
            "power": spec.get("curvature_interval_power"),
        }
    compact = ", ".join(f"{key}={value}" for key, value in core.items() if value is not None)
    return f"{strategy} ({compact})" if compact else str(strategy)


def _task_timing(run_output_dir: str, task_id: str) -> dict[str, float | None]:
    path = os.path.join(_timing_metrics_dir(run_output_dir), f"{task_id}.json")
    if not os.path.exists(path):
        return {"dit_forward_ms": None, "dit_forward_step_ms": None}
    payload = _read_json(path)
    timers = payload.get("timers", {})
    dit_forward = timers.get("dit_forward", {})
    dit_forward_step = timers.get("dit_forward_step", {})
    return {
        "dit_forward_ms": dit_forward.get("total_ms"),
        "dit_forward_step_ms": dit_forward_step.get("total_ms"),
    }


def _quality_by_task(run_output_dir: str) -> dict[str, dict[str, float]]:
    path = os.path.join(_quality_metrics_dir(run_output_dir), "summary.json")
    if not os.path.exists(path):
        return {}
    payload = _read_json(path)
    by_task: dict[str, dict[str, float]] = {}
    for metric_name, metric_payload in payload.items():
        result = metric_payload.get("result", {})
        for task_id, task_payload in (result.get("by_task_id") or {}).items():
            by_task.setdefault(task_id, {})[metric_name] = task_payload.get("mean_score")
    return by_task


def _memory_by_task(run_output_dir: str) -> dict[str, dict[str, float | int | None]]:
    path = os.path.join(_memory_metrics_dir(run_output_dir), "rank0.json")
    if not os.path.exists(path):
        return {}
    payload = _read_json(path)
    by_task: dict[str, dict[str, float | int | None]] = {}
    for event in payload.get("events", []):
        task_id = event.get("task_id")
        if not task_id:
            continue
        item = by_task.setdefault(
            task_id,
            {
                "flexcache_cache_gb_max": 0.0,
                "flexcache_cache_entries_max": 0,
                "flexcache_cache_tensors_max": 0,
                "gpu_reserved_gb_max": 0.0,
            },
        )
        item["gpu_reserved_gb_max"] = max(float(item["gpu_reserved_gb_max"] or 0.0), float(event.get("gpu_reserved_gb") or 0.0))
        cache_gb = event.get("flexcache_peak_cache_gb", event.get("flexcache_cache_gb"))
        cache_entries = event.get("flexcache_peak_cache_entries", event.get("flexcache_cache_entries"))
        cache_tensors = event.get("flexcache_peak_cache_tensors", event.get("flexcache_cache_tensors"))
        if cache_gb is not None:
            item["flexcache_cache_gb_max"] = max(float(item["flexcache_cache_gb_max"] or 0.0), float(cache_gb or 0.0))
        if cache_entries is not None:
            item["flexcache_cache_entries_max"] = max(int(item["flexcache_cache_entries_max"] or 0), int(cache_entries or 0))
        if cache_tensors is not None:
            item["flexcache_cache_tensors_max"] = max(int(item["flexcache_cache_tensors_max"] or 0), int(cache_tensors or 0))
    for item in by_task.values():
        reserved = float(item.get("gpu_reserved_gb_max") or 0.0)
        cache = float(item.get("flexcache_cache_gb_max") or 0.0)
        item["flexcache_memory_reserved_pct"] = None if reserved <= 0 else cache / reserved * 100.0
    return by_task


def _policy_paths(run_output_dir: str) -> dict[str, str]:
    log_dir = _logs_dir(run_output_dir)
    candidates = {
        "model": "flexcache_model_policy.ppm",
        "layer": "flexcache_layer_policy.ppm",
        "attn": "flexcache_attn_policy.ppm",
        "teacache": "teacache_policy_timestep_pos.ppm",
        "pab": "pab_policy_timestep_pos.ppm",
    }
    return {
        strategy: os.path.join(log_dir, filename)
        for strategy, filename in candidates.items()
        if os.path.exists(os.path.join(log_dir, filename))
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_flexcache_comparison(run_output_dir: str) -> None:
    request_path = os.path.join(run_output_dir, "request_params.json")
    if not os.path.exists(request_path):
        return

    requests = _read_json(request_path).get("requests", [])
    quality = _quality_by_task(run_output_dir)
    memory = _memory_by_task(run_output_dir)
    policies = _policy_paths(run_output_dir)

    rows = []
    baseline_dit_ms = None
    for request in requests:
        task_id = request.get("request_id")
        params = request.get("params", {})
        spec = params.get("flexcache_params") or {}
        strategy = spec.get("strategy") or params.get("flexcache") or "none"
        timing = _task_timing(run_output_dir, task_id)
        dit_ms = timing.get("dit_forward_step_ms") or timing.get("dit_forward_ms")
        if baseline_dit_ms is None and dit_ms:
            baseline_dit_ms = float(dit_ms)
        speedup = None if not baseline_dit_ms or not dit_ms else baseline_dit_ms / float(dit_ms)
        mem = memory.get(task_id, {})
        row = {
            "task_id": task_id,
            "strategy": strategy,
            "core_params": _strategy_label(params),
            "dit_forward_ms": dit_ms,
            "speedup_vs_first": speedup,
            "quality": quality.get(task_id, {}),
            "flexcache_cache_gb_max": mem.get("flexcache_cache_gb_max"),
            "flexcache_memory_reserved_pct": mem.get("flexcache_memory_reserved_pct"),
            "policy_ppm": policies.get(strategy),
        }
        rows.append(row)

    summary = {
        "baseline_task_id": rows[0]["task_id"] if rows else None,
        "baseline_note": "speedup_vs_first uses the first request in request_params.json as baseline",
        "rows": rows,
    }
    _write_json(os.path.join(_metrics_dir(run_output_dir), "flexcache_comparison.json"), summary)
    _write_markdown(run_output_dir, rows)


def _write_markdown(run_output_dir: str, rows: list[dict[str, Any]]) -> None:
    metric_names = sorted({name for row in rows for name in row.get("quality", {}).keys()})
    header = [
        "task_id",
        "method",
        "params",
        "dit_fw_s",
        "speedup",
        *metric_names,
        "cache_gb",
        "cache/reserved",
        "policy",
    ]
    lines = [
        "# FlexCache Comparison",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        quality = row.get("quality", {})
        policy = row.get("policy_ppm")
        policy_cell = "-" if not policy else os.path.relpath(policy, run_output_dir)
        cells = [
            row.get("task_id"),
            row.get("strategy"),
            row.get("core_params"),
            _fmt(None if row.get("dit_forward_ms") is None else row["dit_forward_ms"] / 1000.0),
            _fmt(row.get("speedup_vs_first")),
            *[_fmt(quality.get(name)) for name in metric_names],
            _fmt(row.get("flexcache_cache_gb_max")),
            "-" if row.get("flexcache_memory_reserved_pct") is None else f"{row['flexcache_memory_reserved_pct']:.2f}%",
            policy_cell,
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines.append("")
    lines.append("Speedup is computed against the first request in `request_params.json`.")

    path = os.path.join(_metrics_dir(run_output_dir), "flexcache_comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
