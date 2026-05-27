#!/usr/bin/env python3
"""Plot DiTango policy trace JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch


DECISION_LABELS = {
    0: "warmup/cooldown",
    1: "anchor",
    2: "compute",
    3: "reuse",
}
DECISION_COLORS = {
    0: "#9e9e9e",
    1: "#1fa34a",
    2: "#2f6fdf",
    3: "#e0b72f",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_json", type=Path, help="Path to ditango_policy_trace.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated PNG files")
    parser.add_argument("--branch", choices=("pos", "neg", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_trace(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])
    if not records:
        raise ValueError(f"No records found in {path}")
    return data, records


def selected_branches(records: Iterable[dict], branch: str) -> list[str]:
    branches = sorted({str(record["branch"]) for record in records})
    if branch == "both":
        return branches
    if branch not in branches:
        raise ValueError(f"Branch {branch!r} is not present. Available: {branches}")
    return [branch]


def build_matrix(records: list[dict], branch: str, field: str, total_layers: int, group_num: int):
    branch_records = [record for record in records if record["branch"] == branch]
    steps = sorted({int(record["step"]) for record in branch_records})
    if not steps:
        raise ValueError(f"No records for branch {branch}")
    step_to_col = {step: idx for idx, step in enumerate(steps)}
    rows = total_layers * group_num
    mat = np.full((rows, len(steps)), np.nan, dtype=np.float64)

    for record in branch_records:
        layer = int(record["layer"])
        group = int(record["group"])
        row = layer * group_num + group
        col = step_to_col[int(record["step"])]
        value = record.get(field)
        if value is not None:
            mat[row, col] = float(value)
    return steps, mat


def add_axis_labels(ax, steps, total_layers: int, group_num: int):
    if len(steps) <= 20:
        tick_cols = list(range(len(steps)))
    else:
        tick_cols = np.linspace(0, len(steps) - 1, 12, dtype=int).tolist()
    ax.set_xticks(tick_cols)
    ax.set_xticklabels([str(steps[idx]) for idx in tick_cols], rotation=45, ha="right")
    ax.set_xlabel("step")

    layer_rows = [layer * group_num + (group_num - 1) / 2 for layer in range(total_layers)]
    if total_layers <= 30:
        ax.set_yticks(layer_rows)
        ax.set_yticklabels([f"L{layer}" for layer in range(total_layers)])
    else:
        tick_layers = np.linspace(0, total_layers - 1, 16, dtype=int).tolist()
        ax.set_yticks([layer * group_num + (group_num - 1) / 2 for layer in tick_layers])
        ax.set_yticklabels([f"L{layer}" for layer in tick_layers])
    ax.set_ylabel("layer / group rows")

    for layer in range(1, total_layers):
        ax.axhline(layer * group_num - 0.5, color="white", linewidth=0.35, alpha=0.8)


def plot_decisions(records, branch, total_layers, group_num, output_dir, dpi):
    steps, mat = build_matrix(records, branch, "decision_code", total_layers, group_num)
    cmap = ListedColormap([DECISION_COLORS[idx] for idx in range(4)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(max(10, len(steps) * 0.18), max(6, total_layers * group_num * 0.12)))
    ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    add_axis_labels(ax, steps, total_layers, group_num)
    ax.set_title(f"DiTango decisions ({branch})")
    legend = [Patch(facecolor=DECISION_COLORS[key], label=label) for key, label in DECISION_LABELS.items()]
    ax.legend(handles=legend, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"decisions_{branch}.png", dpi=dpi)
    plt.close(fig)


def plot_numeric(records, branch, total_layers, group_num, field, output_dir, dpi, log_scale=False):
    steps, mat = build_matrix(records, branch, field, total_layers, group_num)
    masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(max(10, len(steps) * 0.18), max(6, total_layers * group_num * 0.12)))
    kwargs = {}
    if log_scale:
        positive = masked.compressed()
        positive = positive[positive > 0]
        if positive.size:
            kwargs["norm"] = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap="viridis", **kwargs)
    add_axis_labels(ax, steps, total_layers, group_num)
    ax.set_title(f"DiTango {field} ({branch})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(field)
    fig.tight_layout()
    fig.savefig(output_dir / f"{field}_{branch}.png", dpi=dpi)
    plt.close(fig)


def plot_step_summary(records, branch, output_dir, dpi):
    branch_records = [record for record in records if record["branch"] == branch]
    steps = sorted({int(record["step"]) for record in branch_records})
    counts = {code: [] for code in DECISION_LABELS}
    mean_interval = []
    mean_curvature = []

    for step in steps:
        step_records = [record for record in branch_records if int(record["step"]) == step]
        codes = [int(record["decision_code"]) for record in step_records]
        for code in DECISION_LABELS:
            counts[code].append(codes.count(code) / max(len(codes), 1))

        intervals = [record["interval"] for record in step_records if record.get("interval") is not None]
        curvatures = [record["curvature"] for record in step_records if record.get("curvature") is not None]
        mean_interval.append(float(np.mean(intervals)) if intervals else np.nan)
        mean_curvature.append(float(np.mean(curvatures)) if curvatures else np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax = axes[0]
    bottom = np.zeros(len(steps), dtype=np.float64)
    for code, label in DECISION_LABELS.items():
        values = np.asarray(counts[code], dtype=np.float64)
        ax.bar(steps, values, bottom=bottom, color=DECISION_COLORS[code], label=label, width=0.85)
        bottom += values
    ax.set_ylabel("decision fraction")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", ncol=4)
    ax.set_title(f"DiTango step summary ({branch})")

    axes[1].plot(steps, mean_interval, marker="o", linewidth=1.2, label="mean interval")
    axes[1].set_ylabel("mean interval")
    axes[1].set_xlabel("step")
    ax2 = axes[1].twinx()
    ax2.plot(steps, mean_curvature, marker=".", linewidth=1.0, color="#444444", label="mean curvature")
    ax2.set_ylabel("mean curvature")

    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / f"step_summary_{branch}.png", dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    data, records = load_trace(args.trace_json)
    output_dir = args.output_dir or args.trace_json.parent / "ditango_policy_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_layers = int(data["total_layers"])
    group_num = int(data["group_num"])
    branches = selected_branches(records, args.branch)
    for branch in branches:
        plot_decisions(records, branch, total_layers, group_num, output_dir, args.dpi)
        plot_numeric(records, branch, total_layers, group_num, "interval", output_dir, args.dpi)
        plot_numeric(records, branch, total_layers, group_num, "curvature", output_dir, args.dpi, log_scale=True)
        plot_step_summary(records, branch, output_dir, args.dpi)

    print(f"Saved DiTango trace plots to {output_dir}")


if __name__ == "__main__":
    main()
