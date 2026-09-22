"""Create reviewable CSV snapshots from distributed wrapper JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

MODELS = ("wan1.3b", "wan14b", "qwen", "zimage")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument(
        "--node",
        action="append",
        default=[],
        metavar="JOB_ID=HOSTNAME",
        help="node metadata for older reports without a hostnames field",
    )
    args = parser.parse_args()
    node_overrides = dict(item.split("=", 1) for item in args.node)

    selected = set(args.models)
    cases = {}
    for path in args.reports:
        report = json.loads(path.read_text())
        if report.get("status") != "complete":
            raise ValueError(f"incomplete report: {path}")
        for case in report["cases"]:
            if case["model"] in selected:
                if not case["correct"]:
                    raise ValueError(f"incorrect case in {path}: {case}")
                key = (report["cp"], case["model"], case["algorithm"])
                if key in cases:
                    raise ValueError(f"duplicate case {key}: {path}")
                cases[key] = (report, case)

    expected = {
        (cp, model, algorithm)
        for cp in (2, 4, 8)
        for model in selected
        for algorithm in ("agkv", "ulysses")
    }
    if set(cases) != expected:
        missing, extra = sorted(expected - set(cases)), sorted(set(cases) - expected)
        raise ValueError(f"case matrix mismatch; missing={missing}, extra={extra}")

    summary, samples = [], []
    for key in sorted(cases, key=lambda x: (x[0], MODELS.index(x[1]), x[2])):
        report, case = cases[key]
        nodes = sorted(set(report.get("hostnames", [])))
        if not nodes and report.get("job_id") in node_overrides:
            nodes = [node_overrides[report["job_id"]]]
        flattened = {
            side: [value for repeat in repeats for value in repeat["samples_ms"]]
            for side, repeats in case["timings"].items()
        }
        direct = statistics.median(flattened["direct"])
        bound = statistics.median(flattened["bound"])
        metadata = case["metadata"]
        summary.append(
            {
                "job_id": report.get("job_id", ""),
                "nodes": ",".join(nodes),
                "gpu": "NVIDIA H20",
                "dtype": "bfloat16",
                "batch": 1,
                "cp": report["cp"],
                "model": case["model"],
                "algorithm": case["algorithm"],
                "image_tokens": case["sequence"],
                "text_tokens": 0 if case["model"].startswith("wan") else 512,
                "heads": case["heads"],
                "head_dim": case["head_dim"],
                "schedule": metadata["schedule"],
                "direct_ms": direct,
                "bound_ms": bound,
                "overhead_us": (bound - direct) * 1000,
                "overhead_pct": (bound / direct - 1) * 100,
                "correct": case["correct"],
                "attention_plan": json.dumps(
                    metadata.get("attention_plan"), sort_keys=True
                ),
                "selected_transfers": json.dumps(
                    metadata.get("selected_transfers"), sort_keys=True
                ),
            }
        )
        for side, repeats in case["timings"].items():
            for repeat_index, repeat in enumerate(repeats, 1):
                for sample_index, value in enumerate(repeat["samples_ms"], 1):
                    samples.append(
                        {
                            "job_id": report.get("job_id", ""),
                            "cp": report["cp"],
                            "model": case["model"],
                            "algorithm": case["algorithm"],
                            "round": repeat_index,
                            "side": side,
                            "sample": sample_index,
                            "rank_max_ms": value,
                        }
                    )

    write_csv(args.summary, summary)
    write_csv(args.samples, samples)


if __name__ == "__main__":
    main()
