#!/usr/bin/env python3
"""Fit and visualize MiniMax-H3 step-cost candidates from a warmup report."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("warmup_report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _sequence_length(row: dict[str, Any]) -> float:
    return float(row["sequence_length"])


def _nearest(train: list[dict[str, Any]], row: dict[str, Any]) -> float:
    candidate = min(
        train,
        key=lambda item: abs(_sequence_length(item) - _sequence_length(row)),
    )
    return float(candidate["latency_ms"])


def _piecewise(train: list[dict[str, Any]], row: dict[str, Any]) -> float:
    target = _sequence_length(row)
    ordered = sorted(train, key=_sequence_length)
    if len(ordered) == 1:
        return float(ordered[0]["latency_ms"])
    lower, upper = ordered[0], ordered[1]
    if target >= _sequence_length(ordered[-1]):
        lower, upper = ordered[-2], ordered[-1]
    else:
        for left, right in zip(ordered, ordered[1:]):
            if _sequence_length(left) <= target <= _sequence_length(right):
                lower, upper = left, right
                break
    x0, x1 = _sequence_length(lower), _sequence_length(upper)
    y0, y1 = float(lower["latency_ms"]), float(upper["latency_ms"])
    if x1 == x0:
        return statistics.mean((y0, y1))
    return max(1e-6, y0 + (target - x0) * (y1 - y0) / (x1 - x0))


def _regression(
    train: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    degree: int,
) -> float:
    design = []
    for item in train:
        normalized = _sequence_length(item) / 10_000.0
        values = [normalized**power for power in range(degree + 1)]
        design.append(values)
    target = np.asarray(
        [float(item["latency_ms"]) for item in train], dtype=np.float64
    )
    fitted, *_ = np.linalg.lstsq(
        np.asarray(design, dtype=np.float64),
        target,
        rcond=None,
    )
    normalized = _sequence_length(row) / 10_000.0
    values = [normalized**power for power in range(degree + 1)]
    return max(1e-6, float(np.dot(fitted, np.asarray(values))))


def _metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    errors = [
        abs(record["predicted_ms"] - record["actual_ms"]) / record["actual_ms"]
        for record in records
    ]
    signed = [
        (record["predicted_ms"] - record["actual_ms"]) / record["actual_ms"]
        for record in records
    ]
    return {
        "count": len(records),
        "mape": statistics.mean(errors),
        "p95_absolute_relative_error": float(np.percentile(errors, 95)),
        "max_absolute_relative_error": max(errors),
        "max_underestimate": abs(min(0.0, min(signed))),
        "bias": statistics.mean(signed),
    }


def _evaluate(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictors: dict[
        str, Callable[[list[dict[str, Any]], dict[str, Any]], float]
    ] = {
        "nearest": _nearest,
        "sequence_linear": lambda train, row: _regression(
            train, row, degree=1
        ),
        "sequence_piecewise": _piecewise,
        "sequence_quadratic": lambda train, row: _regression(
            train, row, degree=2
        ),
    }
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["width"]), []).append(row)

    records = []
    summary: dict[str, Any] = {}
    for width, group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda row: (
                int(row["sequence_length"]),
                str(row.get("profile_id")),
            ),
        )
        key = f"cp{width}"
        group_summary = {}
        for name, predictor in predictors.items():
            predictions = []
            explicit_train = [row for row in ordered if row.get("split") == "train"]
            explicit_validation = [
                row for row in ordered if row.get("split") == "validation"
            ]
            evaluations = []
            if explicit_train and explicit_validation:
                evaluations = [
                    (explicit_train, row) for row in explicit_validation
                ]
            else:
                evaluations = [
                    (ordered[:index] + ordered[index + 1 :], row)
                    for index, row in enumerate(ordered)
                ]
            for train, row in evaluations:
                minimum = 3 if name == "sequence_quadratic" else 2
                if len(train) < minimum:
                    continue
                predicted = predictor(train, row)
                record = {
                    "group": key,
                    "candidate": name,
                    "profile_id": row.get("profile_id"),
                    "cp_width": width,
                    "sequence_length": row["sequence_length"],
                    "actual_ms": float(row["latency_ms"]),
                    "predicted_ms": predicted,
                }
                predictions.append(record)
                records.append(record)
            if predictions:
                group_summary[name] = _metrics(predictions)
        if group_summary:
            group_summary["recommended"] = min(
                (
                    name
                    for name in predictors
                    if name in group_summary
                ),
                key=lambda name: (
                    group_summary[name]["p95_absolute_relative_error"],
                    group_summary[name]["max_underestimate"],
                    group_summary[name]["mape"],
                ),
            )
        summary[key] = group_summary
    return summary, records


def _write_plot(rows: list[dict[str, Any]], path_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = sorted({int(row["width"]) for row in rows})
    figure, axes = plt.subplots(
        1,
        len(widths),
        figsize=(5.4 * len(widths), 5.2),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, width in zip(axes, widths):
        selected = [row for row in rows if int(row["width"]) == width]
        training = [row for row in selected if row.get("split") != "validation"]
        validation = [row for row in selected if row.get("split") == "validation"]
        normalized = np.asarray(
            [_sequence_length(row) / 10_000.0 for row in training]
        )
        design = np.stack(
            [np.ones_like(normalized), normalized, normalized**2], axis=1
        )
        coefficients, *_ = np.linalg.lstsq(
            design,
            np.asarray([float(row["latency_ms"]) for row in training]),
            rcond=None,
        )
        curve_x = np.linspace(
            min(_sequence_length(row) for row in selected),
            max(_sequence_length(row) for row in selected),
            300,
        )
        curve_normalized = curve_x / 10_000.0
        curve_y = (
            coefficients[0]
            + coefficients[1] * curve_normalized
            + coefficients[2] * curve_normalized**2
        )
        validation_x = np.asarray(
            [_sequence_length(row) for row in validation], dtype=np.float64
        )
        validation_normalized = validation_x / 10_000.0
        validation_fit = (
            coefficients[0]
            + coefficients[1] * validation_normalized
            + coefficients[2] * validation_normalized**2
        )
        axis.plot(
            curve_x / 1000.0,
            curve_y / 1000.0,
            linewidth=2.2,
            label="Quadratic fit",
        )
        axis.scatter(
            [_sequence_length(row) / 1000.0 for row in training],
            [float(row["latency_ms"]) / 1000.0 for row in training],
            s=58,
            marker="o",
            label="Training measurements",
            zorder=3,
        )
        axis.scatter(
            validation_x / 1000.0,
            [float(row["latency_ms"]) / 1000.0 for row in validation],
            s=70,
            marker="^",
            label="Held-out measurements",
            zorder=3,
        )
        axis.scatter(
            validation_x / 1000.0,
            validation_fit / 1000.0,
            s=72,
            marker="x",
            linewidths=2,
            label="Fitted validation points",
            zorder=4,
        )
        relative_error = np.abs(
            validation_fit
            - np.asarray([float(row["latency_ms"]) for row in validation])
        ) / np.asarray([float(row["latency_ms"]) for row in validation])
        axis.set_title(f"TP2 × CP{width} · MAPE {np.mean(relative_error):.2%}")
        axis.set_xlabel("Padded sequence length (K tokens)")
        axis.set_ylabel("DiT step latency (s)")
        axis.grid(True, alpha=0.28)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
    )
    figure.suptitle(
        "MiniMax-H3 cost model: quadratic sequence-length fit",
        y=0.99,
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(path_stem.with_suffix(".png"), dpi=180)
    figure.savefig(path_stem.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    args = _parser().parse_args()
    report = json.loads(args.warmup_report.read_text(encoding="utf-8"))
    rows = [
        row
        for row in report["rows"]
        if isinstance(row.get("cost_features"), dict)
        and row["cost_features"].get("kind") == "minimax_h3"
    ]
    if not rows:
        raise ValueError("warmup report contains no H3 cost feature rows")
    output_dir = (args.output_dir or args.warmup_report.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, records = _evaluate(rows)
    (output_dir / "cost_model_analysis.json").write_text(
        json.dumps({"groups": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "cost_model_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _write_plot(rows, output_dir / "cost_model_fit")
    lines = ["# MiniMax-H3 Cost Model Profile", ""]
    for group, metrics in summary.items():
        lines.extend((f"## {group}", "", f"Recommended: `{metrics.get('recommended')}`", ""))
        for candidate, values in metrics.items():
            if candidate == "recommended":
                continue
            lines.append(
                f"- `{candidate}`: MAPE {values['mape']:.2%}, "
                f"P95 {values['p95_absolute_relative_error']:.2%}, "
                f"max underestimate {values['max_underestimate']:.2%}"
            )
        lines.append("")
    (output_dir / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
