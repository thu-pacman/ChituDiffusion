#!/usr/bin/env python3
"""Profile H3 latency as a sparse function of sequence length and CP width."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_TOKENS = (8192, 16384, 24576, 32768, 49152)
VALIDATION_TOKENS = (12288, 20480, 40960)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/dockerdata/MiniMax-H3/FL2VA/transformer",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/chitu-api/h3-cost-profile-sparse"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations positive")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    token_grid = (*TRAIN_TOKENS, *VALIDATION_TOKENS)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    combined_rows = []
    width_reports = {}
    for width in (1, 2, 4):
        report_path = output_root / f"cp{width}.json"
        log_path = output_root / f"cp{width}.log"
        if not report_path.exists():
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=8",
                str(ROOT / "script/h3_tp_smoke.py"),
                "--model-path",
                args.model_path,
                "--tp",
                "2",
                "--up",
                str(width),
                "--backend",
                args.backend,
                "--token-grid",
                ",".join(str(value) for value in token_grid),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
                "--report",
                str(report_path),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_s,
                    check=True,
                )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        width_reports[str(width)] = report
        for row in report["rows"]:
            samples_ms = [float(value) * 1000.0 for value in row["forward_seconds"]]
            tokens = int(row["tokens"])
            combined_rows.append(
                {
                    "profile_id": f"tokens-{tokens}-cp{width}",
                    "split": (
                        "train" if tokens in TRAIN_TOKENS else "validation"
                    ),
                    "sequence_length": tokens,
                    "image_tokens": tokens,
                    "width": width,
                    "batch_size": 1,
                    "conditions": 1,
                    "latency_ms": statistics.median(samples_ms),
                    "p95_latency_ms": float(np.percentile(samples_ms, 95)),
                    "samples_ms": samples_ms,
                    "cost_features": {
                        "kind": "minimax_h3",
                        "text_tokens": 0,
                        "condition_tokens": 0,
                        "audio_tokens": 0,
                        "video_tokens": tokens,
                        "used_tokens": tokens,
                        "padded_tokens": tokens,
                        "pad_tokens": 0,
                        "attention_work": tokens**2,
                        "output_type": "latent",
                        "synthetic": True,
                    },
                }
            )
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip().splitlines()
    combined = {
        "source": "h3_sparse_gpu_profile",
        "independent_variables": ["sequence_length", "cp_width"],
        "tp_degree": 2,
        "cp_widths": [1, 2, 4],
        "train_tokens": list(TRAIN_TOKENS),
        "validation_tokens": list(VALIDATION_TOKENS),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "hardware": hardware,
        "rows": combined_rows,
        "raw_reports": width_reports,
    }
    target = output_root / "h3_sparse_profile.json"
    target.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
