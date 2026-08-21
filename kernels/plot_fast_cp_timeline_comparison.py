#!/usr/bin/env python3
"""Compare NVTX-scoped CUDA timelines from multiple Nsight SQLite exports."""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_nsys_timeline import (
    COLORS,
    LABELS,
    Event,
    _load_events,
    _nvtx_window,
    _short_name,
)


@dataclass(frozen=True)
class Trace:
    label: str
    path: Path
    nvtx_label: str
    start_ns: int
    events: list[Event]

    @property
    def span_us(self) -> float:
        return (max(event.end_ns for event in self.events) - self.start_ns) / 1000


def _parse_trace(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("trace must be LABEL=PATH") from error
    return label, Path(path)


def _load(label: str, path: Path, nvtx: str) -> Trace:
    with sqlite3.connect(path) as connection:
        start_ns, _, nvtx_label = _nvtx_window(connection, nvtx)
        # Use the host NVTX end only to select correlated CUDA API calls.
        _, host_end_ns, _ = _nvtx_window(connection, nvtx)
        events = _load_events(connection, start_ns, host_end_ns)
    if not events:
        raise RuntimeError(f"{label}: no CUDA activities in {nvtx!r}")
    return Trace(label, path, nvtx_label, start_ns, events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        type=_parse_trace,
        required=True,
        help="repeat as LABEL=PATH; PATH is an Nsight SQLite export",
    )
    parser.add_argument("--nvtx", default="iteration_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title", default="Fast context-parallel attention · CUDA timeline"
    )
    args = parser.parse_args()

    traces = [_load(label, path, args.nvtx) for label, path in args.trace]
    max_us = max(trace.span_us for trace in traces)
    categories = [
        category
        for category in COLORS
        if any(
            event.category == category
            for trace in traces
            for event in trace.events
        )
    ]
    figure, axes = plt.subplots(
        len(traces),
        1,
        figsize=(15, 2.0 + 2.1 * len(traces)),
        sharex=True,
        squeeze=False,
    )

    for axis, trace in zip(axes[:, 0], traces, strict=True):
        streams = sorted({event.stream for event in trace.events})
        lane_for_stream = {stream: index for index, stream in enumerate(streams)}
        for event in trace.events:
            lane = lane_for_stream[event.stream]
            start_us = (event.start_ns - trace.start_ns) / 1000
            duration_us = (event.end_ns - event.start_ns) / 1000
            axis.broken_barh(
                [(start_us, duration_us)],
                (lane - 0.34, 0.68),
                facecolors=COLORS[event.category],
                edgecolors="white",
                linewidth=0.45,
            )
            if duration_us >= 22:
                axis.text(
                    start_us + duration_us / 2,
                    lane,
                    _short_name(event.name, event.category),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    clip_on=True,
                )
        axis.set_yticks(
            range(len(streams)), [f"stream {stream}" for stream in streams]
        )
        axis.set_ylim(-0.7, len(streams) - 0.3)
        axis.set_ylabel(trace.label, fontweight="bold")
        axis.grid(axis="x", color="#D0D0D0", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        axis.text(
            0.995,
            0.94,
            f"GPU drain: {trace.span_us:.1f} µs",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555",
        )

    axes[0, 0].set_title(args.title, loc="left", fontweight="bold")
    axes[-1, 0].set_xlabel("Time from NVTX iteration start (µs)")
    axes[-1, 0].set_xlim(0, max_us * 1.02)
    figure.legend(
        handles=[
            Patch(facecolor=COLORS[category], label=LABELS[category])
            for category in categories
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=min(6, len(categories)),
        frameon=False,
    )
    sources = " · ".join(
        f"{trace.label}: {trace.path.name}" for trace in traces
    )
    figure.text(
        0.01,
        0.01,
        f"Source: Nsight Systems rank 0 · NVTX selector: {args.nvtx} · {sources}",
        fontsize=7,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
