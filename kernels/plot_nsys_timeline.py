#!/usr/bin/env python3
"""Render an NVTX-scoped CUDA timeline from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


@dataclass(frozen=True)
class Event:
    start_ns: int
    end_ns: int
    stream: int
    name: str
    category: str


COLORS = {
    "attention": "#3274A1",
    "merge": "#8E5EA2",
    "p2p": "#E1812C",
    "a2a": "#5F9E6E",
    "local": "#9A9A9A",
    "sync": "#D4A72C",
    "other": "#7A7A7A",
}

LABELS = {
    "attention": "Attention",
    "merge": "Online-softmax merge",
    "p2p": "P2P K/V",
    "a2a": "All-to-all",
    "local": "Local memory operation",
    "sync": "Synchronization",
    "other": "Other CUDA kernel",
}


def _export_sqlite(report: Path, output: Path) -> Path:
    subprocess.run(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(output),
            str(report),
        ],
        check=True,
    )
    return output


def _category(name: str) -> str:
    lowered = name.lower()
    if "sdpa" in lowered or "flash_attn" in lowered or "flash_fwd" in lowered:
        return "attention"
    if "ring_merge" in lowered:
        return "merge"
    if "peer-to-peer" in lowered:
        return "p2p"
    if "a2a_" in lowered or "all_to_all" in lowered:
        return "a2a"
    if "memcpy" in lowered or "memset" in lowered or "elementwise" in lowered:
        return "local"
    if "barrier" in lowered or "publish_consumed" in lowered:
        return "sync"
    return "other"


def _short_name(name: str, category: str) -> str:
    if category == "attention":
        return "Attention"
    if category == "merge":
        return "LSE merge"
    if category == "p2p":
        return "P2P K/V"
    if category == "a2a":
        return "A2A"
    if "[CUDA memcpy" in name:
        return name.removeprefix("[CUDA memcpy ").removesuffix("]")
    if name == "[CUDA memset]":
        return "Memset"
    return name.split("(", 1)[0].replace("void ", "")[-30:]


def _string_map(connection: sqlite3.Connection) -> dict[int, str]:
    return dict(connection.execute("SELECT id, value FROM StringIds"))


def _nvtx_window(
    connection: sqlite3.Connection, selector: str
) -> tuple[int, int, str]:
    strings = _string_map(connection)
    matches: list[tuple[int, int, str]] = []
    for start, end, text, text_id in connection.execute(
        """
        SELECT start, end, text, textId
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL
        ORDER BY start
        """
    ):
        label = text if text is not None else strings.get(text_id, "")
        if selector in label:
            matches.append((start, end, label))
    if not matches:
        raise ValueError(f"no NVTX range contains {selector!r}")
    if len(matches) > 1:
        choices = ", ".join(label for _, _, label in matches)
        raise ValueError(f"NVTX selector is ambiguous: {choices}")
    return matches[0]


def _load_events(
    connection: sqlite3.Connection, start_ns: int, end_ns: int
) -> list[Event]:
    strings = _string_map(connection)
    events: list[Event] = []
    # NVTX is a host-side range: queued GPU work can start after the range has
    # closed. Select activities by the CUDA API calls issued inside the range,
    # rather than by GPU timestamps, or a serialized inverse collective (and
    # often the tail of attention) disappears from the plot.
    correlation_ids = {
        row[0]
        for row in connection.execute(
            """
            SELECT correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE start >= ? AND start <= ?
            """,
            (start_ns, end_ns),
        )
    }
    if not correlation_ids:
        raise RuntimeError("selected NVTX range contains no CUDA API calls")
    placeholders = ",".join("?" for _ in correlation_ids)
    correlation_args = tuple(correlation_ids)

    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_KERNEL'"
    ).fetchone():
        for start, end, stream, short_name in connection.execute(
            f"""
            SELECT start, end, streamId, shortName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE correlationId IN ({placeholders})
            """,
            correlation_args,
        ):
            name = strings.get(short_name, f"kernel:{short_name}")
            events.append(Event(start, end, stream, name, _category(name)))

    memcpy_labels = dict(
        connection.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")
    )
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_MEMCPY'"
    ).fetchone():
        for start, end, stream, copy_kind in connection.execute(
            f"""
            SELECT start, end, streamId, copyKind
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE correlationId IN ({placeholders})
            """,
            correlation_args,
        ):
            name = f"[CUDA memcpy {memcpy_labels.get(copy_kind, copy_kind)}]"
            events.append(Event(start, end, stream, name, _category(name)))

    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_MEMSET'"
    ).fetchone():
        for start, end, stream in connection.execute(
            f"""
            SELECT start, end, streamId
            FROM CUPTI_ACTIVITY_KIND_MEMSET
            WHERE correlationId IN ({placeholders})
            """,
            correlation_args,
        ):
            events.append(Event(start, end, stream, "[CUDA memset]", "local"))

    return sorted(events, key=lambda event: (event.start_ns, event.stream))


def plot_timeline(
    events: list[Event],
    *,
    start_ns: int,
    end_ns: int,
    nvtx_label: str,
    source: Path,
    output: Path,
    title: str,
) -> None:
    streams = sorted({event.stream for event in events})
    lane_for_stream = {stream: index for index, stream in enumerate(streams)}
    figure, axis = plt.subplots(
        figsize=(14, max(3.2, 1.25 + 0.9 * len(streams))),
        constrained_layout=True,
    )

    for event in events:
        lane = lane_for_stream[event.stream]
        start_us = (max(event.start_ns, start_ns) - start_ns) / 1000
        duration_us = (
            min(event.end_ns, end_ns) - max(event.start_ns, start_ns)
        ) / 1000
        axis.broken_barh(
            [(start_us, duration_us)],
            (lane - 0.34, 0.68),
            facecolors=COLORS[event.category],
            edgecolors="white",
            linewidth=0.5,
        )
        if duration_us >= 18:
            axis.text(
                start_us + duration_us / 2,
                lane,
                _short_name(event.name, event.category),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                clip_on=True,
            )

    span_us = (end_ns - start_ns) / 1000
    axis.set_xlim(0, span_us)
    axis.set_ylim(-0.7, len(streams) - 0.3)
    axis.set_yticks(range(len(streams)), [f"CUDA stream {s}" for s in streams])
    axis.set_xlabel("Time from NVTX range start (µs)")
    axis.set_ylabel("GPU execution lane")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="x", color="#D0D0D0", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)

    categories = [category for category in COLORS if any(
        event.category == category for event in events
    )]
    axis.legend(
        handles=[
            Patch(facecolor=COLORS[category], label=LABELS[category])
            for category in categories
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(4, len(categories)),
        frameon=False,
    )
    figure.text(
        0.01,
        0.01,
        f"Source: {source.name} · NVTX: {nvtx_label} · span: {span_us:.1f} µs",
        fontsize=8,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".nsys-rep or exported .sqlite")
    parser.add_argument("--nvtx", default="iteration_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Nsight Systems CUDA timeline")
    args = parser.parse_args()

    sqlite_path = args.input
    if args.input.suffix == ".nsys-rep":
        sqlite_path = args.output.with_suffix(".sqlite")
        _export_sqlite(args.input, sqlite_path)
    with sqlite3.connect(sqlite_path) as connection:
        start_ns, end_ns, nvtx_label = _nvtx_window(connection, args.nvtx)
        events = _load_events(connection, start_ns, end_ns)
    if not events:
        raise RuntimeError("the selected NVTX range contains no CUDA activities")
    # Drain all GPU work launched by the host-side NVTX range.
    end_ns = max(event.end_ns for event in events)
    plot_timeline(
        events,
        start_ns=start_ns,
        end_ns=end_ns,
        nvtx_label=nvtx_label,
        source=args.input,
        output=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
