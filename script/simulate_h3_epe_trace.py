#!/usr/bin/env python3
"""Replay mixed-resolution MiniMax-H3 arrivals through the production EPE planner.

Example:
  python script/simulate_h3_epe_trace.py \
    --stage-config examples/stage-minimax-h3.yaml \
    --requests 200 --arrival-rate 0.8 \
    --shapes 512x768@2,768x768@5,768x1024@8 \
    --trace-out outputs/h3_trace.json \
    --result-out outputs/h3_epe_vs_tp2cp4.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chitu_diffusion.epe.trace_simulator import (  # noqa: E402
    compare_epe_to_static_cp,
    generate_profile_trace,
    load_trace,
    load_warmup_models,
    run_slo_baseline_sweep,
    save_trace,
)
from chitu_diffusion.models.minimax_h3.fl2va_geometry import (  # noqa: E402
    FL2VATimeRequest,
    resolve_fl2va_plan,
)


def _csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value)
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _csv_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split(",") if value)
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return values


def _shapes(raw: str) -> tuple[tuple[int, int, float, int], ...]:
    values = []
    try:
        for item in raw.split(","):
            resolution, duration = item.split("@", 1)
            height, width = resolution.lower().split("x", 1)
            values.append((int(height), int(width), float(duration), 24))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "shapes must use HxW@seconds,HxW@seconds syntax"
        ) from error
    if not values or any(min(h, w, duration) <= 0 for h, w, duration, _ in values):
        raise argparse.ArgumentTypeError("shape dimensions and durations must be positive")
    return tuple(values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--warmup-report", type=Path)
    source.add_argument(
        "--stage-config",
        type=Path,
        help="launch this deployment, collect its startup warmup, then replay",
    )
    parser.add_argument("--service-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--warmup-timeout-s", type=float, default=3600.0)
    parser.add_argument("--warmup-log", type=Path)
    parser.add_argument("--trace-in", type=Path)
    parser.add_argument("--trace-out", type=Path)
    parser.add_argument("--result-out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--arrival-rate", type=float, default=0.8)
    parser.add_argument(
        "--shapes",
        type=_shapes,
        default=_shapes("512x768@2,768x768@5,768x1024@8"),
        help="mixed requested H3 shapes as HxW@seconds,...",
    )
    parser.add_argument("--steps", type=_csv_ints, default=(20, 30, 50))
    parser.add_argument(
        "--cycle-shapes",
        action="store_true",
        help="assign shapes round-robin so each listed shape appears in the trace",
    )
    parser.add_argument("--deadline-ms", type=float)
    parser.add_argument(
        "--alphas",
        type=_csv_floats,
        default=(1.5, 2.0, 3.0, 5.0),
        help="isolated-P95 deadline multipliers",
    )
    parser.add_argument(
        "--normalized-loads",
        type=_csv_floats,
        default=(0.5, 0.7, 0.9, 1.1, 1.3),
        help="offered loads relative to static CP-max capacity",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-length", type=int, default=64)
    parser.add_argument("--keyframes", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--tp-degree", type=int, default=2)
    parser.add_argument("--cp-world-size", type=int, default=4)
    parser.add_argument("--lane-widths", type=_csv_ints, default=(1, 2, 4))
    parser.add_argument("--pulse-steps", type=int, default=2)
    parser.add_argument(
        "--switch-until-step",
        type=int,
        default=0,
        help="deprecated compatibility option; resize uses dynamic migration payback",
    )
    parser.add_argument("--starvation-ms", type=float, default=30_000.0)
    parser.add_argument("--deadline-guard-ms", type=float, default=0.0)
    parser.add_argument("--min-resize-gain-ms", type=float, default=0.0)
    parser.add_argument("--resize-hysteresis-ms", type=float, default=0.0)
    parser.add_argument("--resize-control-cost-ms", type=float, default=0.0)
    parser.add_argument("--max-active-requests", type=int)
    parser.add_argument(
        "--planner-overhead-ms",
        type=float,
        default=0.0,
        help="simulated control-plane cost charged once per pulse",
    )
    return parser


def _record_startup_warmup(
    stage_config: Path,
    *,
    service_python: Path,
    timeout_s: float,
    log_path: Path,
) -> Path:
    if timeout_s <= 0:
        raise ValueError("warmup timeout must be positive")
    raw = yaml.safe_load(stage_config.read_text(encoding="utf-8"))
    gpu_ids = tuple(int(value) for value in raw["gpu"])
    if not gpu_ids:
        raise ValueError("stage config must contain at least one GPU")
    output_root = Path(raw["output_root"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    report_path = output_root / "epe_warmup.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(service_python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={len(gpu_ids)}",
        "--module",
        "chitu_diffusion.commands.serve",
        "--stage-config",
        str(stage_config.resolve()),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpu_ids)
    environment.setdefault("CHITU_DIST_TIMEOUT_SECONDS", "3600")
    previous_stamp = (
        None
        if not report_path.exists()
        else (report_path.stat().st_mtime_ns, report_path.stat().st_size)
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if (
                    report_path.exists()
                    and (
                        previous_stamp is None
                        or (
                            report_path.stat().st_mtime_ns,
                            report_path.stat().st_size,
                        )
                        != previous_stamp
                    )
                ):
                    try:
                        json.loads(report_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
                    else:
                        return report_path
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"warmup service exited with code {exit_code}; see {log_path}"
                    )
                time.sleep(1.0)
            raise TimeoutError(
                f"startup warmup did not finish within {timeout_s:g}s; see {log_path}"
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)


def _geometry_callbacks(*, text_length: int, keyframes: int):
    def plan_for(height: int, width: int, duration_s: float, fps: int):
        return resolve_fl2va_plan(
            FL2VATimeRequest(
                duration_seconds=duration_s,
                width=width,
                height=height,
                fps=fps,
            )
        )

    def sequence_length(height: int, width: int, duration_s: float, fps: int) -> int:
        plan = plan_for(height, width, duration_s, fps)
        frame_rows = (plan.latent_height // 2) * (plan.latent_width // 2)
        used = (
            text_length
            + keyframes * frame_rows
            + plan.audio_latent_t * 2
            + plan.video_latent_t * frame_rows
        )
        return ((used + 63) // 64) * 64

    def state_bytes(height: int, width: int, duration_s: float, fps: int) -> int:
        plan = plan_for(height, width, duration_s, fps)
        latent_bytes = 4 * (
            24
            * plan.video_latent_t
            * plan.latent_height
            * plan.latent_width
            + 2 * plan.audio_latent_t * 32
        )
        static_bytes = 2 * text_length * 5376
        return latent_bytes + static_bytes

    return sequence_length, state_bytes


def _validate_profile_widths(cost_model, widths: tuple[int, ...]) -> None:
    measured = {
        int(row["width"]) for row in cost_model.snapshot()["rows"]
    }
    missing = sorted(set(widths) - measured)
    if missing:
        raise ValueError(
            "warmup report is missing CP widths "
            f"{missing}; run a TP2/CP4 warmup with allowed widths [1, 2, 4]"
        )


def _trace_profiles(args, sequence_length, state_bytes):
    profiles = []
    for height, width, duration_s, fps in args.shapes:
        for num_steps in args.steps:
            profiles.append(
                {
                    "profile_key": (
                        f"{height}x{width}@{duration_s:g}s/{num_steps}steps"
                    ),
                    "sequence_length": sequence_length(
                        height, width, duration_s, fps
                    ),
                    "num_steps": num_steps,
                    "state_bytes": state_bytes(height, width, duration_s, fps),
                    "metadata": {
                        "height": height,
                        "width": width,
                        "duration_s": duration_s,
                        "fps": fps,
                    },
                }
            )
    return tuple(profiles)


def main() -> int:
    args = _parser().parse_args()
    warmup_report = args.warmup_report
    if args.stage_config is not None:
        warmup_log = args.warmup_log or args.result_out.with_suffix(".warmup.log")
        warmup_report = _record_startup_warmup(
            args.stage_config,
            service_python=args.service_python,
            timeout_s=args.warmup_timeout_s,
            log_path=warmup_log,
        )
    assert warmup_report is not None
    cost_model, transfer_model = load_warmup_models(warmup_report)
    _validate_profile_widths(cost_model, args.lane_widths)
    sequence_length, state_bytes = _geometry_callbacks(
        text_length=args.text_length,
        keyframes=args.keyframes,
    )
    if args.trace_in is not None:
        trace = load_trace(args.trace_in)
    else:
        trace = generate_profile_trace(
            request_count=args.requests,
            arrival_rate_rps=args.arrival_rate,
            profiles=_trace_profiles(args, sequence_length, state_bytes),
            deadline_ms=args.deadline_ms,
            seed=args.seed,
            cycle_profiles=args.cycle_shapes,
        )
    if args.trace_out is not None:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        save_trace(args.trace_out, trace)

    result = compare_epe_to_static_cp(
        trace,
        cost_model=cost_model,
        transfer_cost_model=transfer_model,
        tp_degree=args.tp_degree,
        cp_world_size=args.cp_world_size,
        allowed_lane_widths=args.lane_widths,
        pulse_steps=args.pulse_steps,
        switch_allowed_until_step=args.switch_until_step,
        starvation_ms=args.starvation_ms,
        deadline_guard_ms=args.deadline_guard_ms,
        min_resize_gain_ms=args.min_resize_gain_ms,
        resize_hysteresis_ms=args.resize_hysteresis_ms,
        resize_control_cost_ms=args.resize_control_cost_ms,
        max_active_requests=args.max_active_requests,
        planner_overhead_ms=args.planner_overhead_ms,
    )
    result["slo_baseline"] = run_slo_baseline_sweep(
        trace,
        cost_model=cost_model,
        transfer_cost_model=transfer_model,
        alphas=args.alphas,
        normalized_loads=args.normalized_loads,
        tp_degree=args.tp_degree,
        cp_world_size=args.cp_world_size,
        allowed_lane_widths=args.lane_widths,
        pulse_steps=args.pulse_steps,
        switch_allowed_until_step=args.switch_until_step,
        starvation_ms=args.starvation_ms,
        deadline_guard_ms=args.deadline_guard_ms,
        min_resize_gain_ms=args.min_resize_gain_ms,
        resize_hysteresis_ms=args.resize_hysteresis_ms,
        resize_control_cost_ms=args.resize_control_cost_ms,
        max_active_requests=args.max_active_requests,
        planner_overhead_ms=args.planner_overhead_ms,
    )
    result["service_warmup"] = json.loads(
        warmup_report.read_text(encoding="utf-8")
    )
    result["experiment"] = {
        "warmup_report": str(warmup_report),
        "stage_config": (
            None if args.stage_config is None else str(args.stage_config)
        ),
        "trace_request_count": len(trace),
        "arrival_rate_rps": args.arrival_rate,
        "shapes": [
            {
                "profile_key": request.profile_key,
                "metadata": dict(request.metadata),
                "sequence_length": request.sequence_length,
                "num_steps": request.num_steps,
            }
            for request in trace
        ],
    }
    args.result_out.parent.mkdir(parents=True, exist_ok=True)
    args.result_out.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "trace_requests": len(trace),
                "warmup_report": str(warmup_report),
                **result["comparison"],
                "elastic": result["elastic"]["metrics"],
                "static_tp_cp": result["static_tp_cp"]["metrics"],
                "result": str(args.result_out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
