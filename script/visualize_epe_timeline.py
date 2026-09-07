#!/usr/bin/env python3
"""Plot static CP-max and EPE CP-rank execution timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--load", type=float, default=0.9)
    parser.add_argument("--zoom-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    return parser


def _scenario(
    payload: dict[str, Any], *, alpha: float, load: float
) -> dict[str, Any]:
    for scenario in payload["slo_baseline"]["scenarios"]:
        if (
            abs(float(scenario["alpha"]) - alpha) < 1e-9
            and abs(float(scenario["normalized_load"]) - load) < 1e-9
        ):
            return scenario
    raise ValueError(f"result has no alpha={alpha:g}, load={load:g} scenario")


def _draw(
    axis,
    result: dict[str, Any],
    *,
    cp_world_size: int,
    colors: dict[str, str],
    limit_seconds: float | None,
) -> None:
    profiles = {
        request["request_id"]: request["profile_key"]
        for request in result["requests"]
    }
    for phase in result["timeline"]:
        start = float(phase["compute_start_ms"]) / 1000.0
        end = float(phase["end_ms"]) / 1000.0
        if limit_seconds is not None and start >= limit_seconds:
            continue
        end = min(end, limit_seconds) if limit_seconds is not None else end
        if end <= start:
            continue
        cp_ranks = sorted(
            {int(rank) % cp_world_size for rank in phase["physical_ranks"]}
        )
        profile = profiles[phase["request_id"]]
        for cp_rank in cp_ranks:
            axis.broken_barh(
                [(start, end - start)],
                (cp_rank - 0.39, 0.78),
                facecolors=colors[profile],
                edgecolors=("black" if phase["switched"] else "none"),
                linewidth=0.8,
                alpha=0.9,
            )
    axis.set_yticks(range(cp_world_size))
    axis.set_yticklabels([f"CP{rank}" for rank in range(cp_world_size)])
    axis.set_ylim(-0.55, cp_world_size - 0.45)
    axis.set_xlabel("Simulated wall time (s)")
    axis.set_ylabel("CP rank")
    axis.grid(axis="x", alpha=0.25)


def main() -> None:
    args = _parser().parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    scenario = _scenario(payload, alpha=args.alpha, load=args.load)
    static = scenario["results"]["static-cp-max"]
    elastic = scenario["results"]["elastic-resize"]
    cp_world_size = int(static["planner"]["cp_world_size"])
    profile_names = sorted(
        {request["profile_key"] for request in static["requests"]}
    )
    palette = plt.get_cmap("tab10")
    colors = {
        profile: palette(index) for index, profile in enumerate(profile_names)
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 7.8), sharey=True)
    rows = (
        ("Static CP-max", static),
        ("EPE elastic resize", elastic),
    )
    for row_index, (name, result) in enumerate(rows):
        metrics = result["metrics"]
        title = (
            f"{name} · attainment {metrics['deadline_attainment']:.1%} · "
            f"P95 {metrics['latency_p95_ms'] / 1000:.1f}s · "
            f"switches {metrics['switch_count']}"
        )
        _draw(
            axes[row_index, 0],
            result,
            cp_world_size=cp_world_size,
            colors=colors,
            limit_seconds=None,
        )
        axes[row_index, 0].set_title(f"{title}\nFull trace")
        _draw(
            axes[row_index, 1],
            result,
            cp_world_size=cp_world_size,
            colors=colors,
            limit_seconds=args.zoom_seconds,
        )
        axes[row_index, 1].set_title(
            f"{title}\nFirst {args.zoom_seconds:g}s"
        )
    legend = [
        Patch(facecolor=colors[profile], label=profile)
        for profile in profile_names
    ]
    legend.append(
        Patch(facecolor="white", edgecolor="black", label="Resize boundary")
    )
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(legend),
    )
    figure.suptitle(
        f"Static CP vs EPE timeline · α={args.alpha:g}, "
        f"normalized load={args.load:g}",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    output = args.output or args.result.with_name("static_cp_vs_epe_timeline.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
