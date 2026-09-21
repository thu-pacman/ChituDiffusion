"""Summarize native model-shaped CP attention reports across frameworks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BASELINE_METHODS = {
    "chitu": "nccl_agkv",
    "xdit": "ulysses",
    "sglang": "ulysses",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    reports: list[dict[str, Any]] = []
    for path in files:
        if "smoke" in path.parts:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if report.get("status") in {"complete", "unsupported"} and report.get(
            "model"
        ) and report.get("implementation", {}).get("schedule", "native") == "native":
            reports.append(report)
    if not reports:
        raise SystemExit("no complete or unsupported model reports found")
    return reports


def main() -> None:
    args = parse_args()
    reports = load_reports(args.inputs)
    identities: set[tuple[str, str, str, int]] = set()
    for report in reports:
        identity = (
            report["framework"],
            report["model"],
            report["method"],
            int(report.get("shape", {}).get("cp_degree", 0)),
        )
        if identity in identities:
            raise SystemExit(f"duplicate model benchmark cell: {identity}")
        identities.add(identity)
    baselines: dict[tuple[str, str], float] = {}
    for report in reports:
        shape = report.get("shape", {})
        if (
            report.get("status") == "complete"
            and int(shape.get("cp_degree", 0)) == 1
            and report.get("method") == BASELINE_METHODS.get(report.get("framework"))
        ):
            baselines[(report["framework"], report["model"])] = float(
                report["timing"]["median_ms"]
            )

    lines = [
        "# Native model-shaped CP attention",
        "",
        "| Framework | Model | Method | CP | Status | Median ms | Speedup vs CP1 | Backend | Transport | Node |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    reports.sort(
        key=lambda report: (
            report["model"],
            int(report.get("shape", {}).get("cp_degree", 0)),
            report["framework"],
            report["method"],
        )
    )
    for report in reports:
        shape = report.get("shape", {})
        status = report["status"]
        cp_degree = int(shape.get("cp_degree", 0))
        median = "-"
        speedup = "-"
        backend = "-"
        transport = "-"
        nodes = "-"
        if status == "complete":
            median_ms = float(report["timing"]["median_ms"])
            median = f"{median_ms:.3f}"
            baseline = baselines.get((report["framework"], report["model"]))
            if baseline is not None:
                speedup = f"{baseline / median_ms:.2f}x"
            implementation = report.get("implementation", {})
            backend = str(implementation.get("attention_backend", "-"))
            transport = str(implementation.get("transport", "-"))
            nodes = ",".join(report.get("hardware", {}).get("nodes", ())) or "-"
        else:
            status = f"unsupported: {report.get('reason', 'native constraint')}"
        lines.append(
            "| "
            + " | ".join(
                (
                    str(report["framework"]),
                    str(report["model"]),
                    str(report["method"]),
                    str(cp_degree or "-"),
                    status,
                    median,
                    speedup,
                    backend,
                    transport,
                    nodes,
                )
            )
            + " |"
        )

    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
