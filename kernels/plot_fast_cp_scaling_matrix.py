#!/usr/bin/env python3
"""Plot Fast-CP strong scaling and the winner-strategy latency matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


SEQUENCES = (4096, 8192, 16384, 32768, 75600)
SEQUENCE_LABELS = ("4K", "8K", "16K", "32K", "75.6K")
GPU_COUNTS = (1, 2, 4, 8, 16)
SINGLE_NODE_GPU_COUNTS = (1, 2, 4, 8)

# Same-environment 1-GPU cuDNN medians measured during the 2x8 H20 run.
SINGLE_GPU_MS = {
    4096: 2.5701760053634644,
    8192: 9.881519794464111,
    16384: 39.308366775512695,
    32768: 156.02550506591797,
    75600: 828.2152099609375,
}

# Winner among Fast AGKV, Fast Ulysses, and Fast Ring for CP2/4/8.
# CP16 is the measured two-node Fast USP U8xR2 path.
WINNERS = {
    4096: {
        1: ("1 GPU", SINGLE_GPU_MS[4096]),
        2: ("Ulysses", 1.506),
        4: ("Ulysses", 0.902),
        8: ("Ulysses", 0.583),
        16: ("Fast USP\nU8xR2", 1.339296042919159),
    },
    8192: {
        1: ("1 GPU", SINGLE_GPU_MS[8192]),
        2: ("AGKV", 5.252),
        4: ("AGKV", 2.885),
        8: ("Ulysses", 1.693),
        16: ("Fast USP\nU8xR2", 2.243472099304199),
    },
    16384: {
        1: ("1 GPU", SINGLE_GPU_MS[16384]),
        2: ("AGKV", 19.806),
        4: ("AGKV", 10.272),
        8: ("AGKV", 5.574),
        16: ("Fast USP\nU8xR2", 5.543760061264038),
    },
    32768: {
        1: ("1 GPU", SINGLE_GPU_MS[32768]),
        2: ("AGKV", 78.461),
        4: ("AGKV", 39.041),
        8: ("AGKV", 20.338),
        16: ("Fast USP\nU8xR2", 17.80025577545166),
    },
    75600: {
        1: ("1 GPU", SINGLE_GPU_MS[75600]),
        2: ("AGKV", 412.09),
        4: ("AGKV", 206.94),
        8: ("AGKV", 103.74),
        16: ("Fast USP\nU8xR2", 88.51017761230469),
    },
}

SEQUENCE_COLORS = {
    4096: "#8c6d9c",
    8192: "#6f7fa8",
    16384: "#3f88a8",
    32768: "#278b82",
    75600: "#16724b",
}
STRATEGY_COLORS = {
    "1 GPU": "#e5e7eb",
    "Ulysses": "#f4c27a",
    "AGKV": "#8bb7df",
    "Fast USP\nU8xR2": "#99c7b5",
}


def _latency_text(latency_ms: float) -> str:
    if latency_ms < 10:
        return f"{latency_ms:.3f} ms"
    if latency_ms < 100:
        return f"{latency_ms:.2f} ms"
    return f"{latency_ms:.1f} ms"


def _save(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_speedup(
    output: Path,
    *,
    gpu_counts: tuple[int, ...] = GPU_COUNTS,
) -> None:
    single_node = max(gpu_counts) <= 8
    figure, axis = plt.subplots(figsize=(12.8, 7.4), constrained_layout=True)
    for sequence, label in zip(SEQUENCES, SEQUENCE_LABELS):
        speedups = [
            SINGLE_GPU_MS[sequence] / WINNERS[sequence][gpu_count][1]
            for gpu_count in gpu_counts
        ]
        axis.plot(
            gpu_counts,
            speedups,
            color=SEQUENCE_COLORS[sequence],
            marker="o",
            markersize=7,
            linewidth=2.5,
            label=f"S={label}",
        )
        axis.annotate(
            f"{speedups[-1]:.2f}x",
            (gpu_counts[-1], speedups[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=SEQUENCE_COLORS[sequence],
            fontsize=9,
            fontweight="bold",
        )

    axis.plot(
        gpu_counts,
        gpu_counts,
        color="#6b7280",
        linestyle=(0, (5, 4)),
        linewidth=1.5,
        label="Ideal linear scaling",
    )
    if not single_node:
        axis.axvspan(8.7, 18.2, color="#f3f4f6", zorder=-2)
        axis.text(
            16,
            15.25,
            "2 nodes",
            ha="center",
            color="#6b7280",
            fontsize=10,
        )
    axis.set_xticks(gpu_counts, [str(count) for count in gpu_counts])
    axis.set_xlim(0.5, max(gpu_counts) + (0.9 if single_node else 1.5))
    axis.set_ylim(0.5, max(gpu_counts) + 0.8)
    axis.set_xlabel("GPU count / context-parallel degree", fontsize=11)
    axis.set_ylabel("Strong-scaling speedup vs 1 GPU (x)", fontsize=11)
    axis.set_title(
        (
            "Fast-CP single-node strong scaling on H20"
            if single_node
            else "Fast-CP strong scaling across sequence lengths"
        ),
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    axis.grid(True, alpha=0.24, linewidth=0.8)
    axis.legend(ncol=3, loc="upper left", frameon=False)
    figure.text(
        0.5,
        -0.015,
        (
            "CP2/4/8: best of Fast AGKV, Fast Ulysses, and Fast Ring on one H20 node · "
            "speedup denominator: 1-GPU cuDNN median"
            if single_node
            else "CP2/4/8: best of Fast AGKV, Fast Ulysses, Fast Ring · "
            "CP16: measured 2x8 H20 Fast USP U8xR2 · "
            "speedup denominator: 1-GPU cuDNN median from the CP16 benchmark run"
        ),
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    _save(figure, output)


def plot_winner_matrix(
    output: Path,
    *,
    gpu_counts: tuple[int, ...] = GPU_COUNTS,
) -> None:
    single_node = max(gpu_counts) <= 8
    figure, axis = plt.subplots(figsize=(13.8, 8.0), constrained_layout=True)
    rows = len(SEQUENCES)
    columns = len(gpu_counts)
    for row, sequence in enumerate(SEQUENCES):
        display_row = rows - row - 1
        for column, gpu_count in enumerate(gpu_counts):
            strategy, latency_ms = WINNERS[sequence][gpu_count]
            rectangle = Rectangle(
                (column, display_row),
                1,
                1,
                facecolor=STRATEGY_COLORS[strategy],
                edgecolor="white",
                linewidth=3,
            )
            axis.add_patch(rectangle)
            strategy_label = strategy.replace("\n", " ")
            axis.text(
                column + 0.5,
                display_row + 0.66,
                strategy_label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#1f2937",
            )
            axis.text(
                column + 0.5,
                display_row + 0.35,
                _latency_text(latency_ms),
                ha="center",
                va="center",
                fontsize=11,
                color="#111827",
            )

    if not single_node:
        axis.axvline(4, color="#4b5563", linestyle=(0, (5, 4)), linewidth=1.4)
        axis.text(
            4.5,
            rows + 0.28,
            "2x8 H20",
            ha="center",
            fontsize=10,
            color="#4b5563",
        )
    axis.set_xlim(0, columns)
    axis.set_ylim(0, rows)
    axis.set_xticks(
        [index + 0.5 for index in range(columns)],
        [str(count) for count in gpu_counts],
    )
    axis.set_yticks(
        [index + 0.5 for index in range(rows)],
        list(reversed(SEQUENCE_LABELS)),
    )
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.set_xlabel("GPU count / context-parallel degree", fontsize=11, labelpad=12)
    axis.set_ylabel("Global sequence length", fontsize=11)
    axis.set_title(
        (
            "Fast-CP single-node winner and latency on H20"
            if single_node
            else "Fast-CP winner strategy and latency"
        ),
        fontsize=17,
        fontweight="bold",
        pad=52,
    )
    axis.tick_params(length=0, labelsize=11)
    for spine in axis.spines.values():
        spine.set_visible(False)

    strategy_names = ["1 GPU", "Ulysses", "AGKV"]
    if not single_node:
        strategy_names.append("Fast USP\nU8xR2")
    legend = [
        Patch(facecolor=STRATEGY_COLORS[name], label=name.replace("\n", " "))
        for name in strategy_names
    ]
    axis.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=len(strategy_names),
        frameon=False,
    )
    figure.text(
        0.5,
        -0.015,
        "Cell value: all-rank critical-path median · H20 · BF16 · "
        "B=1 · H=40 · D=128 · dense non-causal attention",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    _save(figure, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kernels/figures"),
    )
    args = parser.parse_args()
    speedup = args.output_dir / "fast-cp-speedup-vs-gpus.png"
    matrix = args.output_dir / "fast-cp-winner-matrix.png"
    single_node_speedup = args.output_dir / "fast-cp-h20-single-node-scaling.png"
    single_node_matrix = args.output_dir / "fast-cp-h20-single-node-winner-matrix.png"
    plot_speedup(speedup)
    plot_winner_matrix(matrix)
    plot_speedup(single_node_speedup, gpu_counts=SINGLE_NODE_GPU_COUNTS)
    plot_winner_matrix(single_node_matrix, gpu_counts=SINGLE_NODE_GPU_COUNTS)
    print(speedup)
    print(matrix)
    print(single_node_speedup)
    print(single_node_matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
