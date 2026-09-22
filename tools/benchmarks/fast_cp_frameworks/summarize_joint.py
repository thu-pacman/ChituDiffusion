"""Summarize weight-free Qwen-Image and Z-Image joint-attention runs."""

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
        raise SystemExit("no complete joint-attention reports found")

    cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    for report in reports:
        shape = report["shape"]
        key = (
            report["model"],
            shape["global_image_sequence"],
            shape["text_sequence"],
            shape["cp_degree"],
            report["method"],
        )
        cells[key] = report

    lines = [
        "# Chitu native joint attention",
        "",
        "- Boundary: native projections, normalization, RoPE, CP attention, and output projections.",
        "- Token partition is read from each report because Qwen-Image and optimized Z-Image shard text while the legacy Z-Image path replicates it.",
        "- Inputs and parameters are deterministic synthetic BF16 tensors; checkpoint weights are not loaded.",
        "",
        "| Model | Image S | Text S | CP | Method | Partition | Latency ms | Speedup vs CP1 | Native vs NCCL | Max abs error |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(cells, key=lambda item: (item[0], item[1], item[2], item[3], item[4])):
        model, sequence, text_sequence, cp_degree, method = key
        report = cells[key]
        latency = report["timing"]["median_ms"]
        cp1 = cells.get((model, sequence, text_sequence, 1, "nccl_agkv"))
        scaling = f"{cp1['timing']['median_ms'] / latency:.3f}x" if cp1 else "-"
        mode = "agkv" if method.endswith("agkv") else "ulysses"
        baseline = cells.get(
            (model, sequence, text_sequence, cp_degree, f"nccl_{mode}")
        )
        versus_nccl = (
            f"{baseline['timing']['median_ms'] / latency:.3f}x"
            if baseline
            else "-"
        )
        partition = report.get("implementation", {}).get("partition", "unknown")
        lines.append(
            f"| {model} | {sequence} | {text_sequence} | {cp_degree} | {method} | {partition} | "
            f"{latency:.3f} | {scaling} | {versus_nccl} | "
            f"{report['accuracy']['max_abs']:.3e} |"
        )

    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
