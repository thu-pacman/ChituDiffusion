#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_json(path: Path) -> Any | None:
    return read_json(path) if path.exists() else None


def is_warmup_request(request: dict[str, Any]) -> bool:
    task_id = str(request.get("request_id") or "")
    role = str(((request.get("params") or {}).get("role")) or "").lower()
    return role == "warmup" or task_id.startswith("warmup_")


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]


def task_timing(run_dir: Path, task_id: str) -> dict[str, Any]:
    payload = maybe_json(run_dir / "metrics" / "timing" / f"{task_id}.json") or {}
    timers = payload.get("timers") or {}
    dit = timers.get("dit_forward_step") or timers.get("dit_forward") or {}
    return {
        "dit_forward_ms": dit.get("total_ms"),
        "dit_forward_samples": dit.get("samples"),
    }


def run_config(run_dir: Path) -> dict[str, Any]:
    system = maybe_json(run_dir / "system_params.json") or {}
    cfg = system.get("config") or {}
    infer = cfg.get("infer") or {}
    diffusion = infer.get("diffusion") or {}
    models = cfg.get("models") or {}
    return {
        "model": models.get("name") or system.get("model_name"),
        "attn_type": infer.get("attn_type"),
        "cp_size": diffusion.get("cp_size"),
    }


def rows_from_run(run_dir: Path, experiment_id: str) -> list[dict[str, Any]]:
    request_payload = maybe_json(run_dir / "request_params.json")
    if not request_payload:
        return []
    cfg = run_config(run_dir)
    case = str(run_dir.name).split("-")[-1]
    case = str(cfg.get("attn_type") or case)
    if case == "flash":
        case = "origin_flash"
    rows = []
    for request in request_payload.get("requests", []):
        if is_warmup_request(request):
            continue
        task_id = str(request.get("request_id") or "")
        params = request.get("params") or {}
        timing = task_timing(run_dir, task_id)
        if timing.get("dit_forward_ms") is None:
            continue
        rows.append(
            {
                "experiment_id": experiment_id,
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "case": case,
                "task_id": task_id,
                "prompt": params.get("prompt"),
                "seed": params.get("seed"),
                "steps": params.get("num_inference_steps"),
                "attn_type": cfg.get("attn_type"),
                "cp_size": cfg.get("cp_size"),
                "dit_forward_ms": timing["dit_forward_ms"],
                "dit_forward_s": timing["dit_forward_ms"] / 1000.0,
                "dit_forward_samples": timing.get("dit_forward_samples"),
            }
        )
    return rows


def quality_by_run_task(experiment_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = maybe_json(experiment_dir / "quality" / "quality_rows.json") or []
    out = {}
    if not isinstance(payload, list):
        return out
    for row in payload:
        if not isinstance(row, dict):
            continue
        run_name = str(row.get("run_name") or "")
        task_id = str(row.get("task_id") or "")
        if run_name and task_id:
            out[(run_name, task_id)] = row
    return out


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    out = []
    for case, items in sorted(grouped.items()):
        dit_values = numeric([item.get("dit_forward_s") for item in items])
        out.append(
            {
                "case": case,
                "num_tasks": len(items),
                "dit_forward_s_mean": mean(dit_values) if dit_values else None,
                "dit_forward_s_min": min(dit_values) if dit_values else None,
                "dit_forward_s_max": max(dit_values) if dit_values else None,
                "psnr_mean": mean(numeric([item.get("psnr") for item in items])) if numeric([item.get("psnr") for item in items]) else None,
                "ssim_mean": mean(numeric([item.get("ssim") for item in items])) if numeric([item.get("ssim") for item in items]) else None,
                "one_minus_lpips_mean": mean(numeric([item.get("one_minus_lpips") for item in items])) if numeric([item.get("one_minus_lpips") for item in items]) else None,
                "hpsv3_score_mean": mean(numeric([item.get("hpsv3_score") for item in items])) if numeric([item.get("hpsv3_score") for item in items]) else None,
            }
        )
    baseline = next((item["dit_forward_s_mean"] for item in out if item["case"] == "origin_flash" and item["dit_forward_s_mean"]), None)
    for item in out:
        value = item.get("dit_forward_s_mean")
        item["speedup_vs_origin"] = None if not baseline or not value else baseline / value
    return out


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(experiment_dir: Path, summary_rows: list[dict[str, Any]], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (experiment_dir / "plots").mkdir(parents=True, exist_ok=True)
        (experiment_dir / "plots" / "PLOT_SKIPPED.txt").write_text(str(exc), encoding="utf-8")
        return

    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("one_minus_lpips_mean", "Perceptual Fidelity", "1-LPIPS vs origin"),
        ("hpsv3_score_mean", "Human Preference", "HPSv3 score"),
    ]
    colors = {
        "origin_flash": "#111827",
        "torch_sdpa_math": "#2563eb",
        "sage": "#d97706",
        "sparge": "#16a34a",
    }
    rows = [row for row in summary_rows if row.get("speedup_vs_origin") is not None and math.isfinite(float(row["speedup_vs_origin"]))]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        points = [row for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
        if not points:
            ax.text(
                0.5,
                0.5,
                "metric missing",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6b7280",
            )
            ax.set_title(title)
            ax.set_xlabel("Speedup vs origin_flash")
            ax.set_ylabel(ylabel)
            ax.grid(True, color="#e5e7eb", linewidth=0.8)
            continue
        for row in points:
            x = float(row["speedup_vs_origin"])
            y = float(row[metric])
            case = str(row["case"])
            ax.scatter(x, y, s=96, color=colors.get(case, "#6b7280"), edgecolor="white", linewidth=1.0, zorder=3)
            ax.annotate(case, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8)
        ax.axvline(1.0, color="#9ca3af", linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_xlabel("Speedup vs origin_flash")
    fig.suptitle(title, y=0.99, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
    fig.savefig(plot_dir / "speed_quality.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--experiment-id", default="flux1_dev_attention")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--title", default="Flux Attention Backend Trade-off")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    runs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and (path / "request_params.json").exists() and not (path / ".skip_perf_collect").exists()
    ] if experiment_dir.exists() else []
    rows = []
    for run_dir in sorted(runs):
        rows.extend(rows_from_run(run_dir, args.experiment_id))

    quality = quality_by_run_task(experiment_dir)
    for row in rows:
        row_quality = quality.get((row["run_name"], str(row["task_id"]))) or {}
        for name in ["psnr", "ssim", "one_minus_lpips", "hpsv3_score"]:
            row[name] = row_quality.get(name)

    if not rows and not args.allow_partial:
        raise SystemExit(f"No benchmark runs found under {experiment_dir}")

    summary_rows = aggregate(rows)
    write_table(experiment_dir / "raw_rows.csv", rows)
    write_table(experiment_dir / "summary.csv", summary_rows)
    (experiment_dir / "raw_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (experiment_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    plot(experiment_dir, summary_rows, args.title)
    print(f"Wrote {experiment_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
