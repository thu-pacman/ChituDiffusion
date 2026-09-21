"""Summarize weight-free Wan native and forced-serial attention runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def main() -> None:
    args = parse_args()
    reports = load_reports(args.inputs)
    if not reports:
        raise SystemExit("no complete Wan reports found")

    cells: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for report in reports:
        shape = report["shape"]
        key = (
            report["model"],
            report["method"],
            shape["global_sequence"],
            shape["cp_degree"],
        )
        implementation = report["implementation"]
        schedule = implementation.get("schedule")
        if schedule is None:
            schedule = "sync" if implementation["forced_serial"] else "native"
        cells.setdefault(key, {})[schedule] = report

    lines = [
        "# Chitu native Wan attention",
        "",
        "- Boundary: projections, normalization, CP attention, and output projection",
        "- Inputs and parameters are deterministic synthetic BF16 tensors; checkpoint weights are not loaded.",
        "- Overlap gain is serialized-CE latency minus native-CE latency; both use the same communication kernels.",
        "",
        "| Model | S | CP | Method | Native ms | Speedup vs CP1 | Serialized CE ms | Sync ms | CE overlap gain ms | Native vs NCCL |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(cells, key=lambda value: (value[0], value[2], value[3], value[1])):
        model, method, sequence, cp_degree = key
        native = cells[key].get("native")
        if native is None:
            continue
        native_ms = native["timing"]["median_ms"]
        cp1 = cells.get((model, "nccl_agkv", sequence, 1), {}).get("native")
        scaling = (
            f"{cp1['timing']['median_ms'] / native_ms:.3f}x" if cp1 else "-"
        )
        serialized_ce = cells[key].get("serialized_ce")
        serialized_ce_ms = (
            serialized_ce["timing"]["median_ms"] if serialized_ce else None
        )
        sync = cells[key].get("sync")
        sync_ms = sync["timing"]["median_ms"] if sync else None
        async_transport = bool(native["implementation"].get("async_transport"))
        gain = (
            serialized_ce_ms - native_ms
            if serialized_ce_ms is not None and async_transport
            else None
        )
        mode = "agkv" if method.endswith("agkv") else "ulysses"
        baseline = cells.get((model, f"nccl_{mode}", sequence, cp_degree), {}).get(
            "native"
        )
        versus_nccl = "-"
        if baseline is not None:
            ratio = baseline["timing"]["median_ms"] / native_ms
            versus_nccl = f"{ratio:.3f}x"
        serialized_text = (
            f"{serialized_ce_ms:.3f}" if serialized_ce_ms is not None else "-"
        )
        sync_text = f"{sync_ms:.3f}" if sync_ms is not None else "-"
        gain_text = f"{gain:+.3f}" if gain is not None else "-"
        lines.append(
            f"| {model} | {sequence} | {cp_degree} | {method} | {native_ms:.3f} | {scaling} | "
            f"{serialized_text} | {sync_text} | {gain_text} | {versus_nccl} |"
        )

    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
