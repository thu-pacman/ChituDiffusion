"""Summarize native end-to-end, communication, and attention measurements."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

COMPONENTS = ("e2e", "communication", "attention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    return [report for report in reports if report.get("status") == "complete"]


def software_key(report: dict[str, Any]) -> tuple[Any, ...]:
    software = report["software"]
    return (
        software["torch"],
        software["torch_cuda"],
        software["cudnn"],
        software["framework_revision"],
    )


def cell_key(report: dict[str, Any]) -> tuple[Any, ...]:
    shape = report["shape"]
    return (
        report["framework"],
        report["method"],
        shape["global_sequence"],
        shape["heads"],
        shape["head_dim"],
        shape["cp_degree"],
        software_key(report),
    )


def main() -> None:
    args = parse_args()
    reports = load_reports(args.inputs)
    if not reports:
        raise SystemExit("no complete benchmark reports found")

    cells: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for report in reports:
        component = report.get("arguments", {}).get("component", "e2e")
        if component not in COMPONENTS:
            continue
        key = cell_key(report)
        if component in cells[key]:
            raise SystemExit(f"duplicate {component} report for {key}")
        cells[key][component] = report

    incomplete = {key: sorted(set(COMPONENTS) - set(parts)) for key, parts in cells.items() if set(parts) != set(COMPONENTS)}
    if incomplete:
        details = "\n".join(f"{key}: missing {missing}" for key, missing in incomplete.items())
        raise SystemExit(f"incomplete breakdown cells:\n{details}")

    nodes = {
        tuple(parts["e2e"]["hardware"].get("nodes", ()))
        for parts in cells.values()
    }
    gpus = {parts["e2e"]["hardware"]["gpu"] for parts in cells.values()}
    if len(nodes) != 1 or len(gpus) != 1:
        raise SystemExit(f"mixed hardware: nodes={nodes}, gpus={gpus}")

    lines = [
        "# Fast CP native QKV-to-O breakdown",
        "",
        f"- GPU: {next(iter(gpus))}",
        f"- Node: {','.join(next(iter(nodes))) or 'unknown'}",
        "- Statistic: median rank-max CUDA-event latency",
        "- Delta is E2E - communication - attention from independent runs; it is not an overlap measurement.",
        "",
        "| S | H | CP | Framework | Method | E2E ms | Comm ms | Attention ms | Delta ms |",
        "| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(cells.items(), key=lambda item: (item[0][2], item[0][5], item[0][0], item[0][1]))
    for key, parts in ordered:
        framework, method, sequence, heads, _, cp_degree, _ = key
        e2e = parts["e2e"]["timing"]["median_ms"]
        communication = parts["communication"]["timing"]["median_ms"]
        attention = parts["attention"]["timing"]["median_ms"]
        delta = e2e - communication - attention
        lines.append(
            f"| {sequence} | {heads} | {cp_degree} | {framework} | {method} | "
            f"{e2e:.3f} | {communication:.3f} | {attention:.3f} | {delta:+.3f} |"
        )

    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
