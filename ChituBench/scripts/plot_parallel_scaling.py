#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


CASE_ORDER = [
    "baseline_1gpu",
    "ring_2gpu",
    "ulysses_2gpu",
    "ring_4gpu",
    "usp_r2u2_4gpu",
    "ulysses_4gpu",
    "ring_8gpu",
    "usp_r4u2_8gpu",
    "ulysses_8gpu",
    "cfp1up2_2gpu",
    "cfp2_2gpu",
    "cfp2up2_4gpu",
    "cfp2ring2_4gpu",
    "cfp2up4_8gpu",
]
CASE_LABELS = {
    "baseline_1gpu": "1 GPU",
    "ring_2gpu": "Ring",
    "ulysses_2gpu": "Ulysses",
    "ring_4gpu": "Ring",
    "usp_r2u2_4gpu": "USP r2u2",
    "ulysses_4gpu": "Ulysses",
    "ring_8gpu": "Ring",
    "usp_r4u2_8gpu": "USP r4u2",
    "ulysses_8gpu": "Ulysses",
    "cfp1up2_2gpu": "CFP1+UP2",
    "cfp2_2gpu": "CFP2",
    "cfp2up2_4gpu": "CFP2+UP2",
    "cfp2ring2_4gpu": "CFP2+Ring2",
    "cfp2up4_8gpu": "CFP2+UP4",
}
GPU_COUNT = {
    "baseline_1gpu": 1,
    "ring_2gpu": 2,
    "ulysses_2gpu": 2,
    "ring_4gpu": 4,
    "usp_r2u2_4gpu": 4,
    "ulysses_4gpu": 4,
    "ring_8gpu": 8,
    "usp_r4u2_8gpu": 8,
    "ulysses_8gpu": 8,
    "cfp1up2_2gpu": 2,
    "cfp2_2gpu": 2,
    "cfp2up2_4gpu": 4,
    "cfp2ring2_4gpu": 4,
    "cfp2up4_8gpu": 8,
}
COLORS = {
    "baseline": "#2f3437",
    "ring": "#3b82f6",
    "ulysses": "#ef8a47",
    "usp": "#10a37f",
    "cfg": "#10a37f",
    "cfg_ring": "#3b82f6",
    "cfg_up": "#ef8a47",
}
MARKERS = {"baseline": "o", "ring": "o", "ulysses": "s", "usp": "^", "cfg": "^", "cfg_ring": "o", "cfg_up": "s"}
METHOD_LABELS = {
    "baseline": "1 GPU baseline",
    "ring": "Ring",
    "ulysses": "Ulysses",
    "usp": "USP",
    "cfg": "CFP",
    "cfg_ring": "CFP + Ring",
    "cfg_up": "CFP + UP",
}
X_OFFSETS = {
    "baseline": 0.0,
    "ring": -0.08,
    "usp": 0.0,
    "ulysses": 0.08,
    "cfg": 0.0,
    "cfg_ring": -0.08,
    "cfg_up": 0.08,
}
LABEL_OFFSETS = {
    "baseline": (0, 8),
    "ring": (-8, -18),
    "usp": (0, 9),
    "ulysses": (8, 10),
    "cfg": (0, 9),
    "cfg_ring": (-8, -18),
    "cfg_up": (8, 10),
}


def method_for_case(case: str) -> str:
    if case == "cfp2_2gpu":
        return "cfg"
    if "ring" in case:
        return "cfg_ring" if case.startswith("cfp") else "ring"
    if "up" in case:
        return "cfg_up" if case.startswith("cfp") else "ulysses"
    if case.startswith("ring"):
        return "ring"
    if case.startswith("ulysses"):
        return "ulysses"
    if case.startswith("usp"):
        return "usp"
    return "baseline"


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def format_float(value: float) -> str:
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--title", default="Flux1-dev Sequence Parallel Scaling")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    rows = read_summary(experiment_dir / "summary.csv")
    by_case = {row["case"]: row for row in rows}
    ordered = [by_case[name] for name in CASE_ORDER if name in by_case]
    if not ordered:
        raise SystemExit(f"No sequence parallel rows found in {experiment_dir / 'summary.csv'}")

    baseline = next((row for row in ordered if row["case"] == "baseline_1gpu"), ordered[0])
    baseline_time = float(baseline["dit_forward_s_mean"])
    for row in ordered:
        speedup = baseline_time / float(row["dit_forward_s_mean"])
        row["speedup_vs_1gpu"] = speedup
        row["parallel_efficiency"] = speedup / GPU_COUNT.get(row["case"], 1)

    (experiment_dir / "summary_parallel.json").write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    with (experiment_dir / "summary_parallel.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(ordered[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        plot_dir = experiment_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        (plot_dir / "SCALING_PLOT_SKIPPED.txt").write_text(str(exc), encoding="utf-8")
        return 0

    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfaf7",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.grid(axis="y", color="#e8e3db", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#3f3a34")
        ax.spines["left"].set_color("#cfc7bc")
        ax.spines["bottom"].set_color("#cfc7bc")

    gpu_ticks = sorted({GPU_COUNT[row["case"]] for row in ordered})
    for row in ordered:
        case = row["case"]
        method = method_for_case(case)
        gpu = GPU_COUNT[case]
        x = gpu + X_OFFSETS[method]
        time_value = float(row["dit_forward_s_mean"])
        speedup_value = float(row["speedup_vs_1gpu"])
        color = COLORS[method]
        marker = MARKERS[method]
        label = CASE_LABELS.get(case, case)
        text_offset = LABEL_OFFSETS[method]
        axes[0].scatter(x, time_value, s=92, color=color, marker=marker, edgecolor="white", linewidth=1.4, zorder=3)
        axes[1].scatter(x, speedup_value, s=92, color=color, marker=marker, edgecolor="white", linewidth=1.4, zorder=3)
        axes[0].annotate(label, (x, time_value), xytext=text_offset, textcoords="offset points", ha="center", fontsize=8.5, color="#3f3a34")
        axes[1].annotate(f"{format_float(speedup_value)}x", (x, speedup_value), xytext=text_offset, textcoords="offset points", ha="center", fontsize=8.5, color="#3f3a34")

    ideal_x = gpu_ticks
    axes[1].plot(ideal_x, ideal_x, color="#a79f92", linewidth=1.2, linestyle=(0, (4, 4)), label="Ideal scaling")
    axes[0].plot(
        ideal_x,
        [baseline_time / x for x in ideal_x],
        color="#a79f92",
        linewidth=1.2,
        linestyle=(0, (4, 4)),
        label="Ideal latency",
    )

    axes[0].set_xticks(gpu_ticks)
    axes[1].set_xticks(gpu_ticks)
    axes[0].set_xlim(min(gpu_ticks) - 0.45, max(gpu_ticks) + 0.45)
    axes[1].set_xlim(min(gpu_ticks) - 0.45, max(gpu_ticks) + 0.45)
    axes[0].set_xlabel("GPUs")
    axes[1].set_xlabel("GPUs")
    axes[0].set_ylabel("Mean DiT forward time (s)")
    axes[1].set_ylabel("Speedup vs 1 GPU")
    axes[0].set_title("Latency")
    axes[1].set_title("Scaling")

    methods = []
    for row in ordered:
        method = method_for_case(row["case"])
        if method not in methods:
            methods.append(method)
    legend_items = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[method],
            color="none",
            markerfacecolor=COLORS[method],
            markeredgecolor="white",
            markersize=9,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]
    legend_items.append(Line2D([0], [0], color="#a79f92", linewidth=1.2, linestyle=(0, (4, 4)), label="Ideal"))
    fig.legend(handles=legend_items, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(args.title, fontsize=15, fontweight="semibold", color="#25211d")
    fig.tight_layout(rect=(0, 0.09, 1, 0.92), w_pad=2.0)
    fig.savefig(plot_dir / "parallel_scaling.png", dpi=220)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
