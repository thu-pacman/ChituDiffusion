#!/usr/bin/env python3
"""Grouped-bar comparison plots for the Qwen-Image parallel ablations.

Two paired ablations share this plotter:

* ``--kind cp_backend``: AGCP vs UCP context-parallel backend at a fixed GPU
  layout (the variable is the attention CP backend).
* ``--kind sampler``: parallel DiT only vs parallel DiT + parallel sampler at a
  fixed GPU layout (the variable is whether latent shards stay local across
  scheduler steps).

For each layout we report two latencies: ``dit_forward`` (DiT-forward time, where
the per-step latent gather and the attention CP backend live) and ``denoise`` (the
whole denoise loop, which additionally includes the sharded scheduler math that
parallel sampler keeps local). ``dit_forward`` comes from ``summary.csv`` written
by ``collect.py``; ``denoise`` is read per run from ``metrics/timing/summary.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# layout key -> (label, gpu_count, {variant: case_id})
LAYOUTS = {
    "cp_backend": [
        ("cp2", "CP2 (2 GPU)", 2, {"agcp": "cp2_agcp", "ucp": "cp2_ucp"}),
        ("cfp2cp2", "CFP2+CP2 (4 GPU)", 4, {"agcp": "cfp2cp2_agcp", "ucp": "cfp2cp2_ucp"}),
        ("cfp2cp4", "CFP2+CP4 (8 GPU)", 8, {"agcp": "cfp2cp4_agcp", "ucp": "cfp2cp4_ucp"}),
    ],
    "sampler": [
        ("cp2", "CP2 (2 GPU)", 2, {"dit": "cp2_dit_only", "dit_sampler": "cp2_dit_sampler"}),
        ("cfp2cp2", "CFP2+CP2 (4 GPU)", 4, {"dit": "cfp2cp2_dit_only", "dit_sampler": "cfp2cp2_dit_sampler"}),
        ("cfp2cp4", "CFP2+CP4 (8 GPU)", 8, {"dit": "cfp2cp4_dit_only", "dit_sampler": "cfp2cp4_dit_sampler"}),
    ],
    "sampler_flux2": [
        ("cp2", "CP2 (2 GPU)", 2, {"dit": "cp2_dit_only", "dit_sampler": "cp2_dit_sampler"}),
        ("cp4", "CP4 (4 GPU)", 4, {"dit": "cp4_dit_only", "dit_sampler": "cp4_dit_sampler"}),
        ("cp8", "CP8 (8 GPU)", 8, {"dit": "cp8_dit_only", "dit_sampler": "cp8_dit_sampler"}),
    ],
}
VARIANT_STYLE = {
    "agcp": ("AGCP (all-gather KV)", "#ef8a47"),
    "ucp": ("UCP (ring / Ulysses)", "#3b82f6"),
    "dit": ("Parallel DiT only", "#94a3b8"),
    "dit_sampler": ("Parallel DiT + Sampler", "#10a37f"),
}
TITLES = {
    "cp_backend": "Qwen-Image Parallel DiT: AGCP vs UCP",
    "sampler": "Qwen-Image: Parallel DiT vs Parallel DiT + Sampler",
    "sampler_flux2": "Flux2-klein-4B: Parallel DiT vs Parallel DiT + Sampler",
}


def read_summary(experiment_dir: Path) -> dict[str, dict[str, str]]:
    path = experiment_dir / "summary.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def denoise_seconds_by_case(experiment_dir: Path) -> dict[str, float]:
    """Map case_id -> per-image denoise seconds from each run's timing summary.

    The ``denoise`` timer records one sample per denoise *step*, so ``total_ms`` is
    the full per-run denoise time; divide by the number of generated images
    (non-warmup requests) to get a per-image figure comparable to ``dit_forward``.
    """
    out: dict[str, float] = {}
    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = run_dir / "metrics" / "timing" / "summary.json"
        if not summary.exists():
            continue
        case = run_dir.name.split("-")[-1]
        payload = json.loads(summary.read_text(encoding="utf-8"))
        denoise = (payload.get("timers") or {}).get("denoise") or {}
        total_ms = denoise.get("total_ms")
        if total_ms is None:
            continue
        requests = json.loads((run_dir / "request_params.json").read_text(encoding="utf-8")).get("requests", [])
        n_images = sum(
            1
            for r in requests
            if str(((r.get("params") or {}).get("role")) or "").lower() != "warmup"
            and not str(r.get("request_id") or "").startswith("warmup_")
        ) or 1
        out[case] = float(total_ms) / 1000.0 / n_images
    return out


def build_rows(experiment_dir: Path, kind: str) -> list[dict[str, object]]:
    summary = read_summary(experiment_dir)
    denoise = denoise_seconds_by_case(experiment_dir)
    baseline = summary.get("baseline_1gpu", {})
    baseline_dit = float(baseline["dit_forward_s_mean"]) if baseline.get("dit_forward_s_mean") else None
    rows: list[dict[str, object]] = []
    for layout_key, layout_label, gpus, variants in LAYOUTS[kind]:
        for variant, case_id in variants.items():
            row = summary.get(case_id)
            if not row or not row.get("dit_forward_s_mean"):
                continue
            dit_s = float(row["dit_forward_s_mean"])
            rows.append(
                {
                    "layout": layout_key,
                    "layout_label": layout_label,
                    "gpus": gpus,
                    "variant": variant,
                    "case": case_id,
                    "dit_forward_s": dit_s,
                    "denoise_s": denoise.get(case_id),
                    "dit_speedup_vs_1gpu": (baseline_dit / dit_s) if baseline_dit else None,
                }
            )
    return rows


def write_markdown(experiment_dir: Path, kind: str, rows: list[dict[str, object]]) -> None:
    lines = ["| layout | GPUs | variant | DiT forward (s) | denoise (s) | DiT speedup vs 1 GPU | rel. to pair |"]
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    by_layout: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_layout.setdefault(str(row["layout"]), []).append(row)
    for layout_key, _label, _gpus, variants in LAYOUTS[kind]:
        items = by_layout.get(layout_key, [])
        ref = next((r for r in items if r["variant"] in ("agcp", "dit")), None)
        ref_dit = float(ref["dit_forward_s"]) if ref else None
        for variant in variants:
            row = next((r for r in items if r["variant"] == variant), None)
            if row is None:
                continue
            rel = ""
            if ref_dit and variant not in ("agcp", "dit"):
                delta = (ref_dit - float(row["dit_forward_s"])) / ref_dit * 100.0
                rel = f"{delta:+.1f}% DiT"
            denoise_s = row["denoise_s"]
            speedup = row["dit_speedup_vs_1gpu"]
            lines.append(
                "| {layout} | {gpus} | {variant} | {dit:.3f} | {den} | {sp} | {rel} |".format(
                    layout=row["layout_label"],
                    gpus=row["gpus"],
                    variant=VARIANT_STYLE[str(row["variant"])][0],
                    dit=float(row["dit_forward_s"]),
                    den=f"{float(denoise_s):.3f}" if denoise_s is not None else "-",
                    sp=f"{float(speedup):.3f}" if speedup is not None else "-",
                    rel=rel,
                )
            )
    (experiment_dir / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (experiment_dir / "ablation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot(experiment_dir: Path, kind: str, rows: list[dict[str, object]], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:  # noqa: BLE001
        (experiment_dir / "plots").mkdir(parents=True, exist_ok=True)
        (experiment_dir / "plots" / "PLOT_SKIPPED.txt").write_text(str(exc), encoding="utf-8")
        return

    variants = [v for v in (LAYOUTS[kind][0][3].keys())]
    layout_keys = [lk for lk, *_ in LAYOUTS[kind]]
    layout_labels = {lk: lbl for lk, lbl, _g, _v in LAYOUTS[kind]}
    by_case_layout = {(str(r["layout"]), str(r["variant"])): r for r in rows}

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfaf7",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    metrics = [("dit_forward_s", "Mean DiT forward time (s)", "DiT forward latency"),
               ("denoise_s", "Mean denoise time (s)", "Full denoise latency")]
    n_var = len(variants)
    bar_w = 0.8 / n_var
    for ax, (metric, ylabel, panel_title) in zip(axes, metrics):
        ax.grid(axis="y", color="#e8e3db", linewidth=0.9)
        ax.set_axisbelow(True)
        x_base = list(range(len(layout_keys)))
        for vi, variant in enumerate(variants):
            label, color = VARIANT_STYLE[variant]
            xs, ys = [], []
            for li, lk in enumerate(layout_keys):
                row = by_case_layout.get((lk, variant))
                val = row.get(metric) if row else None
                if val is None:
                    continue
                xs.append(li + (vi - (n_var - 1) / 2) * bar_w)
                ys.append(float(val))
            bars = ax.bar(xs, ys, width=bar_w * 0.92, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            for rect, y in zip(bars, ys):
                ax.annotate(f"{y:.1f}", (rect.get_x() + rect.get_width() / 2, y),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=8.2, color="#3f3a34")
        ax.set_xticks(x_base)
        ax.set_xticklabels([layout_labels[lk] for lk in layout_keys], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
    legend_items = [Line2D([0], [0], marker="s", color="none", markerfacecolor=VARIANT_STYLE[v][1],
                           markeredgecolor="white", markersize=11, label=VARIANT_STYLE[v][0]) for v in variants]
    fig.legend(handles=legend_items, loc="lower center", ncol=len(variants), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=14.5, fontweight="semibold", color="#25211d")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93), w_pad=2.0)
    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"ablation_{kind}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--kind", required=True, choices=sorted(LAYOUTS.keys()))
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    title = args.title or TITLES[args.kind]
    rows = build_rows(experiment_dir, args.kind)
    if not rows:
        raise SystemExit(f"No rows found in {experiment_dir}/summary.csv for kind={args.kind}")
    write_markdown(experiment_dir, args.kind, rows)
    plot(experiment_dir, args.kind, rows, title)
    print(f"Wrote {experiment_dir / 'ablation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
