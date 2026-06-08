import json
import os
from statistics import mean
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
    strategy = spec.get("strategy") or params.get("flexcache") or "origin"
    fields_by_strategy = {
        "teacache": ("warmup", "cooldown", "teacache_thresh", "use_ref_steps"),
        "pab": ("warmup", "cooldown", "skip_self_range", "skip_cross_range"),
        "blockdance": ("warmup", "cooldown", "boundary_block", "group_size", "start_fraction", "end_fraction"),
        "cubic": ("target_speedup", "warmup", "cooldown", "tau_max", "block_size"),
        "taylorseer": ("warmup", "cooldown", "fresh_threshold", "max_order", "first_enhance"),
        "ditango": (
            "cache_ratio",
            "warmup",
            "cooldown",
            "tau_max",
            "curvature_interval_power",
            "intra_group_size_limit",
            "locality_group_compute_boost",
            "anchor_interval",
            "groupwise_stagger_period",
            "groupwise_stagger_fresh_count",
            "groupwise_stagger_layer_start",
            "groupwise_stagger_layer_end",
            "groupwise_keep_local",
            "groupwise_force_tail_full_layers",
            "groupwise_reuse_stale_kv",
        ),
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
    if not isinstance(payload, dict):
        return {}
    by_task: dict[str, dict[str, float]] = {}
    for metric_name, metric_payload in payload.items():
        if not isinstance(metric_payload, dict):
            continue
        result = metric_payload.get("result") or {}
        if not isinstance(result, dict):
            continue
        by_task_id = result.get("by_task_id") or {}
        if not isinstance(by_task_id, dict):
            continue
        for task_id, task_payload in by_task_id.items():
            if not isinstance(task_payload, dict):
                continue
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


def _reference_path(run_output_dir: str) -> str | None:
    path = os.path.join(run_output_dir, "system_params.json")
    if not os.path.exists(path):
        return None
    payload = _read_json(path)
    config = payload.get("config") or {}
    eval_cfg = config.get("eval") or {}
    reference_path = str(eval_cfg.get("reference_path") or "").strip()
    return reference_path or None


def _origin_baseline_dit_values(run_output_dir: str) -> list[float]:
    reference_path = _reference_path(run_output_dir)
    if not reference_path or not os.path.isdir(reference_path):
        return []

    request_path = os.path.join(reference_path, "request_params.json")
    if not os.path.exists(request_path):
        return []

    values = []
    for request in _read_json(request_path).get("requests", []):
        if _is_warmup_request(request):
            continue
        task_id = request.get("request_id")
        if not task_id:
            continue
        timing = _task_timing(reference_path, task_id)
        dit_ms = timing.get("dit_forward_step_ms") or timing.get("dit_forward_ms")
        if dit_ms:
            values.append(float(dit_ms))
    return values


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


def _is_warmup_request(request: dict[str, Any]) -> bool:
    params = request.get("params", {})
    role = str(params.get("role") or "").strip().lower()
    task_id = str(request.get("request_id") or "")
    return role == "warmup" or task_id.startswith("warmup_")


def _mean_or_none(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return mean(numeric) if numeric else None


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("strategy") or "none"), []).append(row)

    aggregates = []
    for strategy, items in grouped.items():
        quality_names = sorted({name for item in items for name in item.get("quality", {}).keys()})
        quality_mean = {
            name: _mean_or_none([item.get("quality", {}).get(name) for item in items])
            for name in quality_names
        }
        aggregate = {
            "strategy": strategy,
            "num_seeds": len(items),
            "task_ids": [item.get("task_id") for item in items],
            "dit_forward_ms_mean": _mean_or_none([item.get("dit_forward_ms") for item in items]),
            "speedup_vs_origin_mean": _mean_or_none([item.get("speedup_vs_origin") for item in items]),
            "quality_mean": quality_mean,
            "flexcache_cache_gb_max": _mean_or_none([item.get("flexcache_cache_gb_max") for item in items]),
            "compute_saving_ratio_mean": _mean_or_none([item.get("compute_saving_ratio") for item in items]),
            "compute_saved_units_mean": _mean_or_none([item.get("compute_saved_units") for item in items]),
            "core_params": items[0].get("core_params") if items else strategy,
        }
        aggregates.append(aggregate)
    return aggregates


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
    origin_dit_values = []
    for request in requests:
        if _is_warmup_request(request):
            continue
        task_id = request.get("request_id")
        params = request.get("params", {})
        spec = params.get("flexcache_params") or {}
        strategy = spec.get("strategy") or params.get("flexcache") or "origin"
        timing = _task_timing(run_output_dir, task_id)
        dit_ms = timing.get("dit_forward_step_ms") or timing.get("dit_forward_ms")
        if strategy in {"origin", "none"} and dit_ms:
            origin_dit_values.append(float(dit_ms))
        mem = memory.get(task_id, {})
        comp = compute.get(task_id, {})
        row = {
            "task_id": task_id,
            "strategy": strategy,
            "core_params": _strategy_label(params),
            "dit_forward_ms": dit_ms,
            "speedup_vs_origin": None,
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

    baseline_source = "current_origin_rows"
    if not origin_dit_values:
        origin_dit_values = _origin_baseline_dit_values(run_output_dir)
        baseline_source = "reference_path_origin"
    baseline_dit_ms = mean(origin_dit_values) if origin_dit_values else None
    if not origin_dit_values:
        baseline_source = None
    for row in rows:
        dit_ms = row.get("dit_forward_ms")
        row["speedup_vs_origin"] = None if not baseline_dit_ms or not dit_ms else baseline_dit_ms / float(dit_ms)

    aggregates = _aggregate_rows(rows)
    summary = {
        "baseline_task_id": None,
        "baseline_dit_forward_ms_mean": baseline_dit_ms,
        "baseline_source": baseline_source,
        "baseline_note": "speedup_vs_origin uses mean origin dit_forward_step timing from this run or eval.reference_path; warmup requests are excluded.",
        "rows": rows,
        "by_strategy": aggregates,
    }
    _write_json(os.path.join(_metrics_dir(run_output_dir), "flexcache_comparison.json"), summary)
    _write_markdown(run_output_dir, rows, aggregates)


def _write_markdown(run_output_dir: str, rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    metric_names = sorted({name for row in rows for name in row.get("quality", {}).keys()})
    aggregate_metric_names = sorted({name for row in aggregates for name in row.get("quality_mean", {}).keys()})
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
        "## Strategy Mean",
        "",
        "| " + " | ".join(["method", "num_seeds", "dit_fw_s", "speedup", *aggregate_metric_names, "cache_gb", "compute_saved", "saved_units"]) + " |",
        "| " + " | ".join(["---"] * (7 + len(aggregate_metric_names))) + " |",
    ]
    for row in aggregates:
        quality = row.get("quality_mean", {})
        cells = [
            row.get("strategy"),
            row.get("num_seeds"),
            _fmt(None if row.get("dit_forward_ms_mean") is None else row["dit_forward_ms_mean"] / 1000.0),
            _fmt(row.get("speedup_vs_origin_mean")),
            *[_fmt(quality.get(name)) for name in aggregate_metric_names],
            _fmt(row.get("flexcache_cache_gb_max")),
            "-" if row.get("compute_saving_ratio_mean") is None else f"{row['compute_saving_ratio_mean'] * 100.0:.2f}%",
            _fmt(row.get("compute_saved_units_mean"), digits=1),
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")

    lines.extend(
        [
            "",
            "## Per Seed",
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
    )
    for row in rows:
        quality = row.get("quality", {})
        policy = row.get("policy_ppm")
        policy_cell = "-" if not policy else os.path.relpath(policy, run_output_dir)
        cells = [
            row.get("task_id"),
            row.get("strategy"),
            row.get("core_params"),
            _fmt(None if row.get("dit_forward_ms") is None else row["dit_forward_ms"] / 1000.0),
            _fmt(row.get("speedup_vs_origin")),
            *[_fmt(quality.get(name)) for name in metric_names],
            _fmt(row.get("flexcache_cache_gb_max")),
            "-" if row.get("flexcache_memory_reserved_pct") is None else f"{row['flexcache_memory_reserved_pct']:.2f}%",
            "-" if row.get("compute_saving_ratio") is None else f"{row['compute_saving_ratio'] * 100.0:.2f}%",
            _fmt(row.get("compute_saved_units"), digits=1),
            policy_cell,
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines.append("")
    lines.append("Speedup is computed against mean origin timing from this run or `eval.reference_path`; warmup requests are excluded.")
    lines.append("Compute_saved is a runtime FlexCache metric: strategies report baseline proxy units and actual computed units during inference.")

    path = os.path.join(_metrics_dir(run_output_dir), "flexcache_comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
