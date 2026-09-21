"""Collect and fit FreeCache profiles in one Slurm debug job."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from .collect import add_arguments, check_gpu_job, injection_steps, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--coherence", type=float, default=0.0)
    parser.add_argument("--relative-epsilon", type=float, required=True)
    parser.add_argument("--inject-steps", type=int, nargs="+")
    args = parser.parse_args()
    if not 2 <= args.warmup <= args.steps or args.steps < 4:
        parser.error("require steps >= 4 and 2 <= warmup <= steps")
    if any(b < args.warmup or b > args.steps for b in args.budgets):
        parser.error("budgets must lie in [warmup,steps]")
    if len(set(args.budgets)) != len(args.budgets):
        parser.error("budgets must be unique")
    if not 0 <= args.coherence <= 1 or not 0 < args.relative_epsilon <= 1:
        parser.error("coherence must be in [0,1]; relative-epsilon in (0,1]")
    probes = (
        args.inject_steps
        if args.inject_steps is not None
        else injection_steps(args.steps)
    )
    if len(set(probes)) < 2 or any(i < 2 or i >= args.steps for i in probes):
        parser.error("provide at least two distinct injection steps in [2,steps-1]")
    check_gpu_job()
    args.output.mkdir(parents=True, exist_ok=False)
    common = [
        "--model",
        args.model,
        "--model-path",
        args.model_path,
        "--prompts",
        str(args.prompts),
        "--seeds",
        *map(str, args.seeds),
        "--steps",
        str(args.steps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--guidance",
        str(args.guidance),
    ]
    record = {"complete": False, "steps": args.steps, "inject_steps": probes}
    started = time.perf_counter()
    try:
        for mode in ("trace", "propagation"):
            command = [
                sys.executable,
                "-m",
                "tools.freecache.collect",
                mode,
                *common,
                "--output",
                str(args.output / mode),
            ]
            if mode == "propagation":
                command += [
                    "--relative-epsilon",
                    str(args.relative_epsilon),
                    "--inject-steps",
                    *map(str, probes),
                ]
            subprocess.run(command, check=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.freecache.preprocess",
                "fit",
                "--traces",
                str(args.output / "trace" / "traces.json"),
                "--propagation",
                str(args.output / "propagation"),
                "--warmup",
                str(args.warmup),
                "--coherence",
                str(args.coherence),
                "--budgets",
                *map(str, args.budgets),
                "--output",
                str(args.output / "candidates.json"),
            ],
            check=True,
        )
        record["complete"] = True
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        save(args.output / "run.json", record)


if __name__ == "__main__":
    main()
