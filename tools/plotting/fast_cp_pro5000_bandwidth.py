#!/usr/bin/env python3
"""Plot compact CP4/CP8 RTX PRO 5000 bandwidth panels as SVG."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

WIDTH, HEIGHT = 1040, 465
TOP, BOTTOM = 116, 375
PANELS = ((70, 500), (580, 1010))
Y_MIN, Y_MAX = 15.0, 55.0
STYLES = {
    "NCCL Ulysses": ("#b7791f", "2 5", "circle"),
    "Fast Ulysses SM": ("#b7791f", "", "circle"),
    "Fast Ulysses CE": ("#b7791f", "8 5", "circle"),
    "NCCL AGKV": ("#2878b5", "2 5", "square"),
    "Fast AGKV SM": ("#2878b5", "", "square"),
    "Fast AGKV CE": ("#2878b5", "8 5", "square"),
}


def _x(sequence: int, low: int, high: int, left: int, right: int) -> float:
    fraction = (math.log2(sequence) - math.log2(low)) / (
        math.log2(high) - math.log2(low)
    )
    return left + fraction * (right - left)


def _y(value: float) -> float:
    return BOTTOM - (value - Y_MIN) / (Y_MAX - Y_MIN) * (BOTTOM - TOP)


def _marker(x: float, y: float, color: str, shape: str) -> str:
    if shape == "square":
        return (
            f'<rect x="{x - 3.2:.1f}" y="{y - 3.2:.1f}" width="6.4" '
            f'height="6.4" rx="0.7" fill="{color}"/>'
        )
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{color}"/>'


def _draw_panel(
    lines: list[str],
    payload: dict,
    left: int,
    right: int,
    *,
    show_y_labels: bool,
) -> None:
    rows = payload["rows"]
    sequences = [row["global_sequence"] for row in rows]
    labels = ("4K", "8K", "16K", "32K", "75.6K")
    center = (left + right) / 2
    lines.append(
        f'<text class="panel-title" x="{center}" y="{TOP - 13}" '
        f'text-anchor="middle">CP{payload["world_size"]}</text>'
    )
    for tick in range(15, 56, 5):
        y = _y(float(tick))
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#d1d5db" stroke-width="0.7"/>'
        )
        if show_y_labels:
            lines.append(
                f'<text class="tick" x="{left - 9}" y="{y + 3.5:.1f}" '
                f'text-anchor="end">{tick}</text>'
            )
    for sequence, label in zip(sequences, labels, strict=True):
        x = _x(sequence, sequences[0], sequences[-1], left, right)
        lines.append(
            f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{BOTTOM}" '
            'stroke="#e5e7eb" stroke-width="0.6"/>'
        )
        lines.append(
            f'<text class="tick" x="{x:.1f}" y="{BOTTOM + 18}" '
            f'text-anchor="middle">{label}</text>'
        )
    lines.extend(
        (
            f'<line x1="{left}" y1="{BOTTOM}" x2="{right}" y2="{BOTTOM}" '
            'stroke="#6b7280" stroke-width="1"/>',
            f'<line x1="{left}" y1="{TOP}" x2="{left}" y2="{BOTTOM}" '
            'stroke="#6b7280" stroke-width="1"/>',
        )
    )
    for name, (color, dash, shape) in STYLES.items():
        points = [
            (
                _x(
                    row["global_sequence"],
                    sequences[0],
                    sequences[-1],
                    left,
                    right,
                ),
                _y(row["methods"][name]),
            )
            for row in rows
        ]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" '
            f'stroke-width="2"{dash_attr}/>'
        )
        lines.extend(_marker(x, y, color, shape) for x, y in points)


def plot(input_paths: tuple[Path, Path], output_path: Path) -> None:
    payloads = [
        json.loads(input_path.read_text(encoding="utf-8"))
        for input_path in input_paths
    ]
    if [payload["world_size"] for payload in payloads] != [4, 8]:
        raise ValueError("inputs must contain CP4 followed by CP8")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<style>"
        "text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;"
        "fill:#374151}.title{font-size:18px;font-weight:700;fill:#111827}"
        ".panel-title{font-size:13px;font-weight:700;fill:#1f2937}"
        ".label{font-size:11px}.tick{font-size:10px;fill:#6b7280}"
        ".legend{font-size:10px;fill:#374151}.note{font-size:9px;fill:#6b7280}"
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="{WIDTH / 2}" y="25" text-anchor="middle">'
        "RTX PRO 5000 · Fast CP transport bandwidth</text>",
    ]

    # Two legend rows: Ulysses in amber, AGKV in blue.
    legend_x = (285, 510, 735)
    for row_index, names in enumerate(
        (
            ("NCCL Ulysses", "Fast Ulysses SM", "Fast Ulysses CE"),
            ("NCCL AGKV", "Fast AGKV SM", "Fast AGKV CE"),
        )
    ):
        y = 50 + row_index * 21
        for x, name in zip(legend_x, names, strict=True):
            color, dash, shape = STYLES[name]
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            lines.append(
                f'<line x1="{x - 34}" y1="{y}" x2="{x - 7}" y2="{y}" '
                f'stroke="{color}" stroke-width="2"{dash_attr}/>'
            )
            lines.append(_marker(x - 20.5, y, color, shape))
            lines.append(f'<text class="legend" x="{x}" y="{y + 3.5}">{name}</text>')

    for index, (payload, bounds) in enumerate(zip(payloads, PANELS, strict=True)):
        _draw_panel(lines, payload, *bounds, show_y_labels=index == 0)
    lines.extend(
        (
            f'<text class="label" x="{WIDTH / 2}" y="{BOTTOM + 42}" '
            'text-anchor="middle">Global sequence length (tokens)</text>',
            f'<text class="label" transform="translate(18 {(TOP + BOTTOM) / 2}) '
            'rotate(-90)" text-anchor="middle">'
            "Effective remote bandwidth (GiB/s)</text>",
            f'<text class="note" x="{WIDTH / 2}" y="{HEIGHT - 9}" '
            'text-anchor="middle">BF16 · B=1 · H=40 · D=128 · '
            "worst-rank median · 10 warmups + 30 iterations</text>",
            "</svg>",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cp4-input",
        type=Path,
        default=Path("docs/assets/fast_cp/pro5000-cp4-bandwidth.json"),
    )
    parser.add_argument(
        "--cp8-input",
        type=Path,
        default=Path("docs/assets/fast_cp/pro5000-cp8-bandwidth.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/fast_cp/fast-cp-pro5000-bandwidth.svg"),
    )
    args = parser.parse_args()
    plot((args.cp4_input, args.cp8_input), args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
