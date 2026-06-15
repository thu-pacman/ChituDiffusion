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
        request_case = str(params.get("role") or "").strip()
        row_case = request_case if request_case and request_case != "warmup" else str(cfg.get("attn_type") or case)
        if row_case == "flash":
            row_case = "origin_flash"
        rows.append(
            {
                "experiment_id": experiment_id,
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "case": row_case,
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


def _baseline_case(summary_rows: list[dict[str, Any]]) -> str:
    for candidate in ("origin_flash", "torch_sdpa", "baseline_1gpu", "1gpu"):
        if any(row.get("case") == candidate and row.get("dit_forward_s_mean") for row in summary_rows):
            return candidate
    for row in summary_rows:
        if row.get("dit_forward_s_mean"):
            return str(row.get("case"))
    return ""


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
    baseline_case = _baseline_case(out)
    baseline = next((item["dit_forward_s_mean"] for item in out if item["case"] == baseline_case and item["dit_forward_s_mean"]), None)
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


def read_table(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge_rows(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing:
        run_name = str(row.get("run_name") or "")
        task_id = str(row.get("task_id") or "")
        if run_name and task_id:
            merged[(run_name, task_id)] = row
    for row in new_rows:
        merged[(str(row.get("run_name") or ""), str(row.get("task_id") or ""))] = row
    return list(merged.values())


def plot(experiment_dir: Path, summary_rows: list[dict[str, Any]], title: str, experiment_id: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (experiment_dir / "plots").mkdir(parents=True, exist_ok=True)
        (experiment_dir / "plots" / "PLOT_SKIPPED.txt").write_text(str(exc), encoding="utf-8")
        return

    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("psnr_mean", "Pixel Fidelity", "PSNR vs origin"),
        ("one_minus_lpips_mean", "Perceptual Fidelity", "1-LPIPS vs origin"),
    ]
    family_style = {
        "origin": ("#222222", "o"),
        "teacache": ("#7c3aed", "P"),
        "pab": ("#dc2626", "s"),
        "blockdance": ("#2563eb", "^"),
        "cubic": ("#059669", "D"),
        "taylorseer": ("#d97706", "v"),
        "sage": ("#0f766e", "X"),
        "sparge": ("#9333ea", "*"),
        "torch_sdpa": ("#64748b", "h"),
        "other": ("#6b7280", "o"),
    }

    def family_for_case(case: str) -> str:
        if case == "origin_flash":
            return "origin"
        for family in ("teacache", "pab", "blockdance", "cubic", "taylorseer", "sage", "sparge"):
            if case.startswith(family):
                return family
        if case.startswith("torch_sdpa"):
            return "torch_sdpa"
        return "other"
    x_metric = "dit_forward_s_mean" if experiment_id == "qwen_image_attention" else "speedup_vs_origin"
    x_label = "DiT forward latency (s, lower is better)" if experiment_id == "qwen_image_attention" else "Speedup vs origin_flash"
    rows = [row for row in summary_rows if row.get(x_metric) is not None and math.isfinite(float(row[x_metric]))]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.4), sharex=True)
    legend_handles = {}
    for ax, (metric, panel_title, ylabel) in zip(axes, metrics):
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
            ax.set_title(panel_title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.grid(True, color="#e5e7eb", linewidth=0.8)
            continue
        by_family: dict[str, list[dict[str, Any]]] = {}
        for row in points:
            by_family.setdefault(family_for_case(str(row["case"])), []).append(row)
        for family, family_points in sorted(by_family.items()):
            color, marker = family_style.get(family, family_style["other"])
            family_points = sorted(family_points, key=lambda row: float(row[x_metric]))
            x_values = [float(row[x_metric]) for row in family_points]
            y_values = [float(row[metric]) for row in family_points]
            line = ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=1.7 if len(family_points) > 1 else 0,
                alpha=0.78,
                zorder=2,
            )[0]
            scatter = ax.scatter(
                x_values,
                y_values,
                s=88,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=1.1,
                zorder=3,
            )
            legend_handles.setdefault(family, scatter if len(family_points) == 1 else line)
            for row, x, y in zip(family_points, x_values, y_values):
                case = str(row["case"])
                label = case
                for prefix in ("teacache_", "blockdance_", "taylorseer_", "cubic_"):
                    if label.startswith(prefix):
                        label = label[len(prefix) :]
                ax.annotate(label, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=7.5, color="#374151")
        if experiment_id != "qwen_image_attention":
            ax.axvline(1.0, color="#9ca3af", linewidth=1, linestyle="--")
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.set_xlabel(x_label)
    if legend_handles:
        family_labels = {
            "origin": "Origin",
            "teacache": "TeaCache",
            "pab": "PAB",
            "blockdance": "BlockDance",
            "cubic": "Cubic",
            "taylorseer": "TaylorSeer",
            "sage": "SageAttention",
            "sparge": "SpargeAttn",
            "torch_sdpa": "Torch SDPA",
            "other": "Other",
        }
        order = ["origin", "blockdance", "cubic", "pab", "taylorseer", "teacache", "sage", "sparge", "torch_sdpa", "other"]
        handles = [legend_handles[key] for key in order if key in legend_handles]
        labels = [family_labels[key] for key in order if key in legend_handles]
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False, fontsize=9)
    fig.suptitle(title, y=0.99, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=2.0)
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
    if args.allow_partial:
        rows = merge_rows(read_table(experiment_dir / "raw_rows.csv"), rows)

    summary_rows = aggregate(rows)
    write_table(experiment_dir / "raw_rows.csv", rows)
    write_table(experiment_dir / "summary.csv", summary_rows)
    (experiment_dir / "raw_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (experiment_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    plot(experiment_dir, summary_rows, args.title, args.experiment_id)
    print(f"Wrote {experiment_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
