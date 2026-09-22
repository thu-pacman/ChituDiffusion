"""Validate and summarize the six-method native CP-attention matrix."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

SEQUENCES = (4096, 8192, 16384, 32768)
CP_DEGREES = (2, 4, 8)
METHODS = (
    ("chitu", "fast_agkv"),
    ("chitu", "fast_ulysses"),
    ("xdit", "ulysses"),
    ("xdit", "usp"),
    ("sglang", "kv_gather"),
    ("sglang", "ulysses"),
)
BASELINES = {"chitu": "nccl_agkv", "xdit": "ulysses", "sglang": "ulysses"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_reports(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(item.read_text(encoding="utf-8"))
        for item in sorted(path.glob("*.json"))
    ]


def main() -> None:
    args = parse_args()
    reports = load_reports(args.input)
    cells = {
        (
            item["framework"],
            item["method"],
            item["shape"]["global_sequence"],
            item["shape"]["cp_degree"],
        ): item
        for item in reports
        if item.get("status") == "complete"
    }
    expected = {
        (framework, method, sequence, cp)
        for framework, method in METHODS
        for sequence in SEQUENCES
        for cp in CP_DEGREES
    } | {
        (framework, method, sequence, 1)
        for framework, method in BASELINES.items()
        for sequence in SEQUENCES
    }
    missing = sorted(expected - cells.keys())
    if missing:
        raise SystemExit(f"missing {len(missing)} expected cells: {missing[:5]}")
    selected = [cells[key] for key in sorted(expected, key=lambda k: (k[2], k[3], k[0], k[1]))]
    nodes = {tuple(item["hardware"]["nodes"]) for item in selected}
    if len(nodes) != 1:
        raise SystemExit(f"matrix spans multiple node sets: {sorted(nodes)}")
    failed = [item for item in selected if not item["accuracy"].get("checked")]
    if failed:
        raise SystemExit(f"correctness not checked for {len(failed)} cells")

    baselines = {
        (item["framework"], item["shape"]["global_sequence"]): item
        for item in selected
        if item["shape"]["cp_degree"] == 1
    }
    fields = [
        "framework", "method", "sequence", "cp", "median_ms", "p95_ms",
        "speedup_vs_framework_cp1", "max_abs", "relative_l2", "attention_backend",
        "transport", "ulysses_degree", "ring_degree", "node", "torch", "cuda",
        "cudnn", "revision", "worktree_dirty", "slurm_job_id", "transfer_paths",
    ]
    rows: list[dict[str, Any]] = []
    for item in selected:
        shape, impl = item["shape"], item["implementation"]
        baseline = baselines[(item["framework"], shape["global_sequence"])]
        plan = impl.get("attention_plan")
        backend = impl.get("attention_backend", "unknown")
        if plan:
            kernel = "cuDNN" if plan["strategy"] == "cudnn" else "CuTe"
            backend = (
                f"Chitu planner: {kernel}; strategy={plan['strategy']} "
                f"tile={plan['main_tile']}"
            )
        rows.append({
            "framework": item["framework"], "method": item["method"],
            "sequence": shape["global_sequence"], "cp": shape["cp_degree"],
            "median_ms": item["timing"]["median_ms"], "p95_ms": item["timing"]["p95_ms"],
            "speedup_vs_framework_cp1": baseline["timing"]["median_ms"] / item["timing"]["median_ms"],
            "max_abs": item["accuracy"]["max_abs"], "relative_l2": item["accuracy"]["relative_l2"],
            "attention_backend": backend, "transport": impl.get("transport"),
            "ulysses_degree": impl.get("ulysses_degree", ""), "ring_degree": impl.get("ring_degree", ""),
            "node": ",".join(item["hardware"]["nodes"]), "torch": item["software"]["torch"],
            "cuda": item["software"]["torch_cuda"], "cudnn": item["software"]["cudnn"],
            "revision": item["software"]["framework_revision"],
            "worktree_dirty": item["software"].get("framework_worktree_dirty"),
            "slurm_job_id": item["hardware"].get("slurm_job_id"),
            "transfer_paths": json.dumps(impl.get("transfer_paths", {}), sort_keys=True),
        })

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    args.output_csv.write_text(buffer.getvalue(), encoding="utf-8")

    job_ids = sorted({str(item["hardware"].get("slurm_job_id")) for item in selected})
    lines = [
        "# Native CP attention matrix", "",
        "- Boundary: pre-projected local BF16 Q/K/V through native communication and attention to local O.",
        "- Shape: B1, H40, D128, dense non-causal; each rank starts with S/CP query, key, and value tokens.",
        "- Timing: median of 75 rank-maximum CUDA-event samples after 5 warmups.",
        "- Correctness: every cell is checked against dense full-K/V attention before timing.",
        f"- Node: {next(iter(nodes))[0]}",
        f"- Slurm jobs: {', '.join(job_ids)}. Job 244918 produced the CP1 references; its multi-GPU phase found a stale installed Fast CP extension, and job 244920 resumed with the current dual-architecture build.",
        "- xDiT USP: U=2 and R=CP/2 following xDiT's native hybrid test convention; CP2 has R=1 and therefore degenerates to Ulysses.",
        "", "## Latency summary", "",
        "| S | CP | Fast AGKV | Fast Ulysses | xDiT Ulysses | xDiT USP | SGLang AGKV | SGLang Ulysses |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    row_index = {
        (row["sequence"], row["cp"], row["framework"], row["method"]): row
        for row in rows
    }
    for sequence in SEQUENCES:
        for cp in CP_DEGREES:
            values = [
                f"{row_index[(sequence, cp, framework, method)]['median_ms']:.3f}"
                for framework, method in METHODS
            ]
            lines.append(f"| {sequence} | {cp} | " + " | ".join(values) + " |")
    lines.extend((
        "", "## Detailed results", "",
        "| Framework | Method | S | CP | Median ms | P95 ms | Speedup vs framework CP1 | Backend | Transport | Max abs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ))
    for row in rows:
        lines.append(
            f"| {row['framework']} | {row['method']} | {row['sequence']} | {row['cp']} | "
            f"{row['median_ms']:.3f} | {row['p95_ms']:.3f} | {row['speedup_vs_framework_cp1']:.2f}x | "
            f"{row['attention_backend']} | {row['transport']} | {row['max_abs']:.3g} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
