"""Build a comparable Markdown table from Fast CP benchmark JSON reports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, default=[Path("outputs/fast-cp/qkvo")])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline",
        action="append",
        default=["chitu=nccl_agkv", "sglang=ulysses"],
        metavar="FRAMEWORK=METHOD",
        help="CP1 method used as that framework's scaling baseline",
    )
    return parser.parse_args()


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    reports = [report for report in reports if report.get("status") == "complete"]
    if not reports:
        raise SystemExit("no complete benchmark reports found")
    return reports


def uniform_value(reports: list[dict[str, Any]], field: str) -> Any:
    keys = field.split(".")
    values = []
    for report in reports:
        value: Any = report
        for key in keys:
            value = value[key]
        values.append(value)
    encoded = {json.dumps(value, sort_keys=True) for value in values}
    if len(encoded) != 1:
        raise SystemExit(f"incompatible reports: {field} differs")
    return values[0]


def main() -> None:
    args = parse_args()
    reports = load_reports(args.inputs)
    for field in (
        "schema_version",
        "boundary",
        "semantics",
        "shape.batch",
        "shape.heads",
        "shape.head_dim",
        "shape.dtype",
        "hardware.gpu",
        "hardware.compute_capability",
    ):
        uniform_value(reports, field)

    baseline_methods = dict(item.split("=", 1) for item in args.baseline)
    baselines: dict[tuple[Any, ...], dict[str, Any]] = {}
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    software_by_framework: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for report in reports:
        shape = report["shape"]
        software = report["software"]
        software_key = (
            software["torch"],
            software["torch_cuda"],
            software["cudnn"],
            software["framework_revision"],
        )
        software_by_framework[report["framework"]].add(software_key)
        identity = (
            report["framework"],
            report["method"],
            shape["global_sequence"],
            shape["cp_degree"],
            software_key,
        )
        if identity in rows:
            raise SystemExit(f"duplicate benchmark cell: {identity}")
        rows[identity] = report
        baseline_method = baseline_methods.get(report["framework"], report["method"])
        if shape["cp_degree"] == 1 and report["method"] == baseline_method:
            baselines[(report["framework"], shape["global_sequence"], software_key)] = report

    lines = [
        "# Fast CP framework benchmark",
        "",
        f"- Boundary: {reports[0]['boundary']}",
        f"- Semantics: {reports[0]['semantics']}",
        (
            "- Shape: "
            f"B{reports[0]['shape']['batch']} H{reports[0]['shape']['heads']} "
            f"D{reports[0]['shape']['head_dim']} {reports[0]['shape']['dtype']}"
        ),
        f"- GPU: {reports[0]['hardware']['gpu']}",
        "",
        "| Framework | Method | S | CP | Median ms | P95 ms | Speedup vs CP1 | Transport | Torch | Nodes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    order = sorted(
        rows.values(),
        key=lambda report: (
            report["shape"]["global_sequence"],
            report["shape"]["cp_degree"],
            report["framework"],
            report["method"],
        ),
    )
    for report in order:
        shape = report["shape"]
        software = report["software"]
        software_key = (
            software["torch"],
            software["torch_cuda"],
            software["cudnn"],
            software["framework_revision"],
        )
        baseline = baselines.get(
            (report["framework"], shape["global_sequence"], software_key)
        )
        speedup = "-"
        if baseline is not None:
            ratio = baseline["timing"]["median_ms"] / report["timing"]["median_ms"]
            speedup = f"{ratio:.2f}x"
        implementation = report.get("implementation", {})
        lines.append(
            "| "
            + " | ".join(
                (
                    report["framework"],
                    report["method"],
                    str(shape["global_sequence"]),
                    str(shape["cp_degree"]),
                    f"{report['timing']['median_ms']:.3f}",
                    f"{report['timing']['p95_ms']:.3f}",
                    speedup,
                    str(implementation.get("transport", "-")),
                    str(software["torch"]),
                    ",".join(report["hardware"].get("nodes", ())) or "unknown",
                )
            )
            + " |"
        )

    lines.extend(("", "## Software", ""))
    for framework in sorted(software_by_framework):
        for torch, cuda, cudnn, revision in sorted(software_by_framework[framework]):
            lines.append(
                f"- {framework}: torch {torch}, CUDA {cuda}, cuDNN {cudnn}, revision {revision}"
            )
    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
