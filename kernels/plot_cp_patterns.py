"""Create article-ready diagrams for common context-parallel patterns."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


RANK_COLORS = ("#2878B5", "#42A5A5", "#F2A541", "#8E6BBE")
INK = "#17202A"
MUTED = "#5F6B76"
LIGHT = "#EEF2F6"
COMM = "#D75A4A"
COMPUTE = "#263746"
BUFFER = "#DCE7F2"


def _setup_axis(axis: Axes) -> None:
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 62)
    axis.set_aspect("equal")
    axis.axis("off")


def _card(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = "#B8C1CA",
    linewidth: float = 1.2,
    radius: float = 1.5,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.35,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def _arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COMM,
    width: float = 1.5,
    style: str = "-|>",
    connection: str = "arc3",
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            color=color,
            linewidth=width,
            connectionstyle=connection,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )
    )


def _rank_label(axis: Axes, x: float, y: float, rank: int) -> None:
    _card(
        axis,
        x,
        y,
        7.5,
        4.3,
        facecolor=RANK_COLORS[rank],
        edgecolor=RANK_COLORS[rank],
        radius=1.2,
        zorder=4,
    )
    axis.text(
        x + 3.75,
        y + 2.15,
        f"R{rank}",
        color="white",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        zorder=5,
    )


def _striped_tensor(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    colors: tuple[str, ...],
    *,
    direction: str = "vertical",
    label: str | None = None,
    zorder: int = 3,
) -> None:
    _card(
        axis,
        x,
        y,
        width,
        height,
        facecolor="white",
        edgecolor="#9BA7B2",
        radius=0.8,
        zorder=zorder,
    )
    if direction == "vertical":
        step = width / len(colors)
        for index, color in enumerate(colors):
            axis.add_patch(
                Rectangle(
                    (x + index * step + 0.3, y + 0.3),
                    step - 0.6,
                    height - 0.6,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.88,
                    zorder=zorder + 1,
                )
            )
    else:
        step = height / len(colors)
        for index, color in enumerate(colors):
            axis.add_patch(
                Rectangle(
                    (x + 0.3, y + index * step + 0.3),
                    width - 0.6,
                    step - 0.6,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.88,
                    zorder=zorder + 1,
                )
            )
    if label:
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            color=INK,
            fontsize=7.5,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=zorder + 2,
        )


def _title(axis: Axes, title: str, subtitle: str) -> None:
    axis.text(
        2,
        59,
        title,
        fontsize=15,
        fontweight="bold",
        color=INK,
        va="top",
    )
    axis.text(2, 54.8, subtitle, fontsize=9, color=MUTED, va="top")


def draw_ulysses(axis: Axes) -> None:
    _setup_axis(axis)
    _title(axis, "Ulysses", "All-to-all transposes sequence shards into head shards")
    y_positions = (42, 32, 22, 12)
    axis.text(14, 49, "Sequence-sharded Q / K / V", ha="center", fontsize=8.5, color=MUTED)
    axis.text(50, 49, "Full sequence, H / CP heads", ha="center", fontsize=8.5, color=MUTED)
    axis.text(86, 49, "Sequence-sharded output", ha="center", fontsize=8.5, color=MUTED)

    for rank, y in enumerate(y_positions):
        _rank_label(axis, 1.5, y + 1.3, rank)
        _striped_tensor(
            axis,
            10.5,
            y,
            11,
            7,
            (RANK_COLORS[rank],) * 4,
            direction="horizontal",
            label="S/4 × H",
        )
        _striped_tensor(
            axis,
            43.5,
            y,
            13,
            7,
            RANK_COLORS,
            direction="horizontal",
            label="S × H/4",
        )
        _card(axis, 78.5, y, 15, 7, facecolor=LIGHT, radius=0.8)
        axis.text(
            86,
            y + 3.5,
            f"O{rank}: S/4 × H",
            ha="center",
            va="center",
            fontsize=7.5,
            color=INK,
            fontweight="bold",
        )
        _arrow(axis, (22.5, y + 3.5), (42.5, y + 3.5), width=1.2)
        _arrow(axis, (57.5, y + 3.5), (77.5, y + 3.5), width=1.2)

    _card(axis, 29, 12, 8, 37, facecolor="#FFF1EE", edgecolor=COMM)
    axis.text(33, 30, "A2A\nQ/K/V", ha="center", va="center", fontsize=8, color=COMM, fontweight="bold")
    _card(axis, 63, 12, 9, 37, facecolor="#FFF1EE", edgecolor=COMM)
    axis.text(67.5, 30, "inverse\nA2A", ha="center", va="center", fontsize=8, color=COMM, fontweight="bold")
    _card(axis, 46, 3.5, 8, 5.5, facecolor=COMPUTE, edgecolor=COMPUTE)
    axis.text(50, 6.25, "local attention", color="white", fontsize=7.5, ha="center", va="center", fontweight="bold")
    axis.text(
        50,
        1.2,
        "4 collectives: Q, K, V, output · low logical bytes · requires H divisible by CP",
        ha="center",
        fontsize=8,
        color=MUTED,
    )


def draw_ring(axis: Axes) -> None:
    _setup_axis(axis)
    _title(axis, "Ring Attention", "Keep Q local; rotate K/V blocks and merge partial attention online")
    centers = ((28, 40), (72, 40), (72, 15), (28, 15))
    for rank, (cx, cy) in enumerate(centers):
        _card(axis, cx - 10, cy - 6, 20, 12, facecolor="white", edgecolor=RANK_COLORS[rank], linewidth=2)
        _rank_label(axis, cx - 9, cy + 0.5, rank)
        axis.text(cx + 2.5, cy + 2.2, f"Q{rank} stays", fontsize=8, color=INK, ha="center")
        _striped_tensor(
            axis,
            cx - 2,
            cy - 4.5,
            10,
            4.5,
            (RANK_COLORS[rank],),
            label=f"KV{rank}",
        )

    clockwise = ((38, 40, 62, 40), (72, 34, 72, 22), (62, 15, 38, 15), (28, 21, 28, 34))
    for x1, y1, x2, y2 in clockwise:
        _arrow(axis, (x1, y1), (x2, y2), color=COMM, width=2.4)

    _card(axis, 42, 23, 16, 9, facecolor=COMPUTE, edgecolor=COMPUTE)
    axis.text(50, 28.5, "P partial\nattentions", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
    axis.text(50, 24.5, "online LSE merge", color="white", ha="center", va="center", fontsize=7)
    for cx, cy in centers:
        _arrow(
            axis,
            (50, 27.5),
            (cx, cy),
            color="#7F8C8D",
            width=1,
            style="-",
            linestyle="--",
            alpha=0.8,
        )
    axis.text(
        50,
        4,
        "P2P neighbor traffic · CP steps · O(T/CP) KV memory · chunking and online-softmax overhead",
        ha="center",
        fontsize=8,
        color=MUTED,
    )


def draw_agkv(axis: Axes) -> None:
    _setup_axis(axis)
    _title(axis, "All-Gather KV", "Keep Q sequence-sharded; replicate the complete K/V on every rank")
    y_positions = (42, 32, 22, 12)
    axis.text(14, 49, "Local shards", ha="center", fontsize=8.5, color=MUTED)
    axis.text(50, 49, "K/V all-gather", ha="center", fontsize=8.5, color=MUTED)
    axis.text(84, 49, "Local Q × global K/V", ha="center", fontsize=8.5, color=MUTED)

    for rank, y in enumerate(y_positions):
        _rank_label(axis, 1.5, y + 1.3, rank)
        _striped_tensor(axis, 10.5, y, 12, 7, (RANK_COLORS[rank],), label=f"Q{rank} KV{rank}")
        _arrow(axis, (23.5, y + 3.5), (40, 30), width=1.2, alpha=0.85)
        _striped_tensor(
            axis,
            73.5,
            y,
            20,
            7,
            RANK_COLORS,
            direction="vertical",
            label=f"Q{rank} + full K/V",
        )
        _arrow(axis, (60, 30), (72.5, y + 3.5), width=1.2, alpha=0.85)

    _card(axis, 40, 22, 20, 16, facecolor="#FFF1EE", edgecolor=COMM, linewidth=1.8)
    axis.text(50, 32.5, "ALL-GATHER", color=COMM, fontsize=10, ha="center", fontweight="bold")
    _striped_tensor(axis, 43, 24, 14, 6, RANK_COLORS, direction="vertical", label="global K/V")
    axis.text(
        50,
        4,
        "2 collectives: K and V · full-fabric bandwidth · O(T) K/V materialized per rank",
        ha="center",
        fontsize=8,
        color=MUTED,
    )


def draw_full_mesh(axis: Axes) -> None:
    _setup_axis(axis)
    _title(axis, "Full-Mesh Fused Attention", "Every source writes directly to every peer while attention consumes ready tiles")
    centers = ((25, 40), (75, 40), (75, 14), (25, 14))

    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for first, second in pairs:
        x1, y1 = centers[first]
        x2, y2 = centers[second]
        _arrow(
            axis,
            (x1, y1),
            (x2, y2),
            color=COMM,
            width=1.35,
            style="<|-|>",
            alpha=0.72,
            zorder=1,
        )

    for rank, (cx, cy) in enumerate(centers):
        _card(
            axis,
            cx - 11,
            cy - 6,
            22,
            12,
            facecolor="white",
            edgecolor=RANK_COLORS[rank],
            linewidth=2,
            zorder=3,
        )
        _rank_label(axis, cx - 10, cy + 0.5, rank)
        _striped_tensor(
            axis,
            cx - 1,
            cy - 4.5,
            10,
            4.5,
            RANK_COLORS,
            direction="vertical",
            label="sym KV",
            zorder=4,
        )

    _card(axis, 39, 21.5, 22, 12, facecolor=COMPUTE, edgecolor=COMPUTE, linewidth=2, zorder=5)
    axis.text(50, 29, "FUSED ATTENTION", color="white", ha="center", fontsize=9, fontweight="bold", zorder=6)
    axis.text(50, 25, "poll ready tile → compute", color="white", ha="center", fontsize=7.5, zorder=6)
    for cx, cy in centers:
        _arrow(
            axis,
            (50, 27.5),
            (cx, cy),
            color="#7F8C8D",
            width=1,
            style="-",
            linestyle="--",
            alpha=0.9,
            zorder=2,
        )
    axis.text(
        50,
        4,
        "All-peer direct writes · communication/compute overlap inside the kernel · O(T) symmetric K/V buffer",
        ha="center",
        fontsize=8,
        color=MUTED,
    )


DRAWERS = {
    "ulysses": draw_ulysses,
    "ring": draw_ring,
    "agkv": draw_agkv,
    "full-mesh": draw_full_mesh,
}


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="#FBFCFE")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="#FBFCFE")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kernels/figures/cp_patterns"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, drawer in DRAWERS.items():
        figure, axis = plt.subplots(figsize=(13.2, 7.2), facecolor="#FBFCFE")
        drawer(axis)
        figure.tight_layout(pad=0.6)
        _save_figure(figure, args.output_dir / name)

    figure, axes = plt.subplots(2, 2, figsize=(16.5, 11), facecolor="#FBFCFE")
    for axis, drawer in zip(axes.flat, DRAWERS.values()):
        drawer(axis)
    figure.suptitle(
        "Context-Parallel Attention Patterns (CP = 4)",
        fontsize=20,
        fontweight="bold",
        color=INK,
        y=0.985,
    )
    figure.text(
        0.5,
        0.015,
        "Rank colors identify the original sequence shard · red arrows show inter-GPU data movement",
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.01, 0.035, 0.99, 0.965), h_pad=1.2, w_pad=1.0)
    _save_figure(figure, args.output_dir / "cp-patterns-overview")


if __name__ == "__main__":
    main()
