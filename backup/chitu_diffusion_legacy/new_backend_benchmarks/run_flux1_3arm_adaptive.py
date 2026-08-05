from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .flux1_adaptive_trace import build_adaptive_trace


STRATEGIES = ("static_dp", "static_cp", "elastic")


def _parse_resolutions(value: str) -> tuple[int, ...]:
    resolutions = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not resolutions or any(value < 16 or value % 16 for value in resolutions):
        raise argparse.ArgumentTypeError(
            "resolutions must be positive multiples of 16"
        )
    return resolutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and run Flux.1 static-DP/static-CP/EPE on one trace"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolutions", type=_parse_resolutions, default=(512, 1024, 2048))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--pulse-steps", type=int, default=5)
    parser.add_argument("--requests-per-phase", type=int, default=6)
    parser.add_argument("--slo-factor", type=float, default=1.2)
    parser.add_argument("--sparse-factor", type=float, default=0.5)
    parser.add_argument("--overload-factor", type=float, default=1.2)
    parser.add_argument("--phase-gap-factor", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--no-parallel-vae", action="store_true")
    parser.add_argument("--vae-parallel-halo", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.pulse_steps < 1:
        parser.error("step counts must be positive")
    if args.warmup_steps < 3:
        parser.error("--warmup-steps must be >= 3")
    if args.requests_per_phase < 1:
        parser.error("--requests-per-phase must be positive")
    return args


def _wait_for_paths(paths: list[Path], timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    missing = paths
    while missing and time.monotonic() < deadline:
        missing = [path for path in paths if not path.exists()]
        if missing:
            time.sleep(0.1)
    if missing:
        raise TimeoutError(f"timed out waiting for: {missing}")


def _run_rank_command(
    command: list[str],
    *,
    log_path: Path,
    master_port: int,
) -> None:
    env = os.environ.copy()
    env["MASTER_PORT"] = str(master_port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 4:
        raise ValueError("the Flux.1 three-arm benchmark requires exactly four ranks")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "adaptive_trace.json"
    benchmark_module = "chitu_diffusers.benchmarks.flux1_embedded_benchmark"
    base_port = int(os.environ.get("MASTER_PORT", "29500"))
    common = [
        sys.executable,
        "-m",
        benchmark_module,
        "--model-path",
        args.model_path,
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--pulse-steps",
        str(args.pulse_steps),
        "--timeout-s",
        str(args.timeout_s),
        "--vae-parallel-halo",
        str(args.vae_parallel_halo),
    ]
    if args.no_parallel_vae:
        common.append("--no-parallel-vae")

    calibration_dir = output_dir / "calibration"
    _run_rank_command(
        [
            *common,
            "--strategy",
            "elastic",
            "--output-dir",
            str(calibration_dir),
            "--resolutions",
            ",".join(str(value) for value in args.resolutions),
            "--num-requests",
            str(len(args.resolutions)),
            "--warmup-only",
        ],
        log_path=calibration_dir / f"service-rank{rank}.log",
        master_port=base_port,
    )
    warmup_path = calibration_dir / "epe_warmup.json"
    if rank == 0:
        warmup = json.loads(warmup_path.read_text(encoding="utf-8"))
        trace = build_adaptive_trace(
            warmup,
            steps=args.steps,
            requests_per_phase=args.requests_per_phase,
            world_size=world_size,
            slo_factor=args.slo_factor,
            sparse_factor=args.sparse_factor,
            overload_factor=args.overload_factor,
            phase_gap_factor=args.phase_gap_factor,
            seed=args.seed,
        )
        trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    _wait_for_paths([trace_path])

    for index, strategy in enumerate(STRATEGIES, start=1):
        run_dir = output_dir / strategy
        _run_rank_command(
            [
                *common,
                "--strategy",
                strategy,
                "--trace",
                str(trace_path),
                "--output-dir",
                str(run_dir),
            ],
            log_path=run_dir / f"service-rank{rank}.log",
            master_port=base_port + index,
        )

    if rank != 0:
        return
    expected = [
        output_dir / strategy / filename
        for strategy in STRATEGIES
        for filename in (
            "metrics.json",
            *(f"timeline-rank{index}.jsonl" for index in range(world_size)),
        )
    ]
    _wait_for_paths(expected)
    for strategy in STRATEGIES:
        (output_dir / strategy / "service.log").touch()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "chitu_diffusers.benchmarks.flux1_3arm_summary",
            "--trace",
            str(trace_path),
            "--run-root",
            str(output_dir),
            "--output",
            str(output_dir / "comparison.json"),
        ],
        check=True,
    )
    plot_command = [
        sys.executable,
        "-m",
        "chitu_diffusers.benchmarks.plot_strategy_timeline",
        "--trace",
        str(trace_path),
    ]
    for strategy, label in (
        ("static_dp", "Static DP"),
        ("static_cp", "Static CP"),
        ("elastic", "Elastic"),
    ):
        plot_command.extend(["--run", f"{label}={output_dir / strategy}"])
    plot_command.extend(["--output", str(output_dir / "strategy_timeline.png")])
    subprocess.run(plot_command, check=True)
    print(f"FLUX1_3ARM_OUTPUT={output_dir}")


if __name__ == "__main__":
    main()
