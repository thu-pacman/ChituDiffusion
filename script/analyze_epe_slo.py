#!/usr/bin/env python3
"""Render and summarize a generic EPAC SLO acceptance sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ("static-cp-max", "elastic-no-resize", "elastic-resize")
LABELS = {
    "static-cp-max": "Static CP-max",
    "elastic-no-resize": "Elastic placement, no resize",
    "elastic-resize": "EPE elastic resize",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--primary-alpha", type=float, default=2.0)
    return parser


def _scenarios(payload: dict[str, Any], alpha: float) -> list[dict[str, Any]]:
    selected = [
        scenario
        for scenario in payload["slo_baseline"]["scenarios"]
        if abs(float(scenario["alpha"]) - alpha) < 1e-9
    ]
    return sorted(selected, key=lambda scenario: float(scenario["normalized_load"]))


def _metric(
    scenarios: list[dict[str, Any]], mode: str, metric: str
) -> list[float]:
    return [
        float(scenario["results"][mode]["metrics"][metric])
        for scenario in scenarios
    ]


def _write_plot(
    scenarios: list[dict[str, Any]],
    *,
    alpha: float,
    output: Path,
) -> None:
    loads = [float(scenario["normalized_load"]) for scenario in scenarios]
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True)
    panels = (
        ("slo_goodput_rps", "SLO goodput (requests/s)"),
        ("deadline_attainment", "Deadline attainment"),
        ("latency_p95_ms", "End-to-end P95 latency (ms)"),
        ("switch_count", "Resize count"),
    )
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        for mode in MODES:
            axis.plot(
                loads,
                _metric(scenarios, mode, metric),
                marker="o",
                linewidth=2,
                label=LABELS[mode],
            )
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.28)
    for axis in axes[-1]:
        axis.set_xlabel("Normalized offered load vs static CP-max")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
    )
    figure.suptitle(
        f"EPE SLO acceptance · isolated-P95 multiplier α={alpha:g}",
        y=0.99,
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def _acceptance(
    scenarios: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    rows = []
    latency_pass = True
    p99_pass = True
    attainment_pass = True
    for scenario in scenarios:
        load = float(scenario["normalized_load"])
        static = scenario["results"]["static-cp-max"]["metrics"]
        elastic = scenario["results"]["elastic-resize"]["metrics"]
        goodput_speedup = float(elastic["slo_goodput_rps"]) / max(
            1e-12, float(static["slo_goodput_rps"])
        )
        row = {
            "normalized_load": load,
            "static_attainment": float(static["deadline_attainment"]),
            "elastic_attainment": float(elastic["deadline_attainment"]),
            "goodput_speedup": goodput_speedup,
            "p95_ratio": float(elastic["latency_p95_ms"])
            / float(static["latency_p95_ms"]),
            "p99_ratio": float(elastic["latency_p99_ms"])
            / float(static["latency_p99_ms"]),
            "switches_per_request": float(elastic["switch_count"])
            / float(elastic["request_count"]),
        }
        rows.append(row)
        attainment_pass &= (
            row["elastic_attainment"] + 1e-12 >= row["static_attainment"]
        )
        if load <= 0.9:
            latency_pass &= row["p95_ratio"] <= 1.001
            p99_pass &= row["p99_ratio"] <= 1.02
    switch_pass = all(row["switches_per_request"] < 0.5 for row in rows)
    checks = {
        "deadline_attainment_no_regression": attainment_pass,
        "p95_no_regression_at_load_le_0_9": latency_pass,
        "p99_regression_le_2pct_at_load_le_0_9": p99_pass,
        "switches_below_0_5_per_request": switch_pass,
    }
    lines = [
        "# EPE SLO Acceptance",
        "",
        "Primary workload: shape-aware deadline at 2× isolated CP-max P95.",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(("", "## Load points", ""))
    for row in rows:
        lines.append(
            f"- load {row['normalized_load']:.1f}: attainment "
            f"{row['static_attainment']:.1%} → {row['elastic_attainment']:.1%}, "
            f"goodput {row['goodput_speedup']:.3f}×, "
            f"P95 {row['p95_ratio']:.3f}×, P99 {row['p99_ratio']:.3f}×, "
            f"switches/request {row['switches_per_request']:.3f}"
        )
    return {"checks": checks, "rows": rows}, lines


def main() -> None:
    args = _parser().parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    scenarios = _scenarios(payload, args.primary_alpha)
    if not scenarios:
        raise ValueError(f"result contains no alpha={args.primary_alpha:g} scenarios")
    output_dir = (args.output_dir or args.result.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_plot(
        scenarios,
        alpha=args.primary_alpha,
        output=output_dir / "epe_slo_acceptance",
    )
    acceptance, lines = _acceptance(scenarios)
    (output_dir / "epe_slo_acceptance.json").write_text(
        json.dumps(acceptance, indent=2),
        encoding="utf-8",
    )
    (output_dir / "EPE_SLO_ACCEPTANCE.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
