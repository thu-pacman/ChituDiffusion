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


def _flexcache_metrics_dir(run_output_dir: str) -> str:
    return os.path.join(_metrics_dir(run_output_dir), "flexcache")


def _quality_metrics_dir(run_output_dir: str) -> str:
    return os.path.join(_metrics_dir(run_output_dir), "quality")


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _strategy_label(params: dict[str, Any]) -> str:
    spec = params.get("flexcache_params") or {}
    strategy = spec.get("strategy") or params.get("flexcache") or "none"
    fields_by_strategy = {
        "teacache": ("warmup", "cooldown", "teacache_thresh", "use_ref_steps"),
        "pab": ("warmup", "cooldown", "skip_self_range", "skip_cross_range"),
        "blockdance": ("warmup", "cooldown", "boundary_block", "group_size", "start_fraction", "end_fraction"),
        "cubic": ("cache_ratio", "warmup", "cooldown", "tau_max", "target_speedup", "anchor_interval", "partition_mode"),
        "taylorseer": ("warmup", "cooldown", "fresh_threshold", "max_order", "first_enhance"),
        "ditango": ("cache_ratio", "warmup", "cooldown", "tau_max", "curvature_interval_power"),
    }
    names = fields_by_strategy.get(strategy, tuple(key for key in spec.keys() if key != "strategy"))
    core = {key: spec.get(key) for key in names}
    if "curvature_interval_power" in core:
        core["power"] = core.pop("curvature_interval_power")
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


def _compute_by_task(run_output_dir: str) -> dict[str, dict[str, Any]]:
    path = os.path.join(_flexcache_metrics_dir(run_output_dir), "rank0.json")
    if not os.path.exists(path):
        return {}
    payload = _read_json(path)
    by_task: dict[str, dict[str, Any]] = {}
    for event in payload.get("events", []):
        if event.get("stage") != "task_summary":
            continue
        task_id = event.get("task_id")
        if not task_id:
            continue
        by_task[task_id] = {
            "baseline_units": event.get("baseline_units"),
            "actual_units": event.get("actual_units"),
            "saved_units": event.get("saved_units"),
            "saving_ratio": event.get("saving_ratio"),
            "event_count": event.get("event_count"),
            "scope_summary": event.get("scope_summary") or {},
        }
    return by_task


def _policy_paths(run_output_dir: str) -> dict[str, str]:
    log_dir = _logs_dir(run_output_dir)
    candidates = {
        "blockdance": "flexcache_blockdance_policy.ppm",
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
    compute = _compute_by_task(run_output_dir)
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
        comp = compute.get(task_id, {})
        row = {
            "task_id": task_id,
            "strategy": strategy,
            "core_params": _strategy_label(params),
            "dit_forward_ms": dit_ms,
            "speedup_vs_first": speedup,
            "quality": quality.get(task_id, {}),
            "flexcache_cache_gb_max": mem.get("flexcache_cache_gb_max"),
            "flexcache_memory_reserved_pct": mem.get("flexcache_memory_reserved_pct"),
            "compute_saving_ratio": comp.get("saving_ratio"),
            "compute_saved_units": comp.get("saved_units"),
            "compute_baseline_units": comp.get("baseline_units"),
            "compute_unit_note": "runtime proxy units from metrics/flexcache",
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
        "compute_saved",
        "saved_units",
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
            "-" if row.get("compute_saving_ratio") is None else f"{row['compute_saving_ratio'] * 100.0:.2f}%",
            _fmt(row.get("compute_saved_units"), digits=1),
            policy_cell,
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines.append("")
    lines.append("Speedup is computed against the first request in `request_params.json`.")
    lines.append("Compute_saved is a runtime FlexCache metric: strategies report baseline proxy units and actual computed units during inference.")

    path = os.path.join(_metrics_dir(run_output_dir), "flexcache_comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
