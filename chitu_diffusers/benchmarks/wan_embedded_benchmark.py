from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.utils import export_to_video

from chitu_diffusers import (
    EmbeddedDiffusionRuntime,
    EmbeddedRuntimeConfig,
    HotSwitchPoolConfig,
    StageWorldSpec,
    WanExecutorFactory,
    WanRequest,
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _export_video_async(
    video: np.ndarray,
    output_path: Path,
    *,
    submitted_at: float,
    fps: int,
) -> dict[str, Any]:
    started_at = time.time()
    export_to_video(video, str(output_path), fps=fps)
    completed_at = time.time()
    with output_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "output": str(output_path),
        "mp4_sha256": digest,
        "bytes": output_path.stat().st_size,
        "encode_queue_ms": (started_at - submitted_at) * 1000.0,
        "encode_ms": (completed_at - started_at) * 1000.0,
    }


def _safe_request_id(request_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", request_id).strip("._")
    return name or "request"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Wan2.1 T2V through the embedded EPAC runtime"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--strategy", choices=("static_dp", "static_cp", "elastic"), required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trace")
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--pulse-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--warmup-only", action="store_true")
    parser.add_argument("--no-balanced-k", action="store_true")
    parser.add_argument("--no-parallel-vae", action="store_true")
    parser.add_argument("--vae-parallel-halo", type=int, default=8)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-workers", type=int, default=4)
    parser.add_argument("--video-fps", type=int, default=16)
    args = parser.parse_args()
    if args.num_requests < 1:
        parser.error("--num-requests must be positive")
    if min(args.steps, args.pulse_steps) < 1:
        parser.error("step counts must be positive")
    if args.warmup_steps < 3:
        parser.error("--warmup-steps must be >= 3")
    if args.frames < 1 or (args.frames - 1) % 4:
        parser.error("--frames must equal 4n+1")
    if args.height % 16 or args.width % 16:
        parser.error("height and width must be multiples of 16")
    if args.vae_parallel_halo < 0:
        parser.error("--vae-parallel-halo must be non-negative")
    if args.video_workers < 1:
        parser.error("--video-workers must be positive")
    if args.video_fps < 1:
        parser.error("--video-fps must be positive")
    return args


def _request_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.trace:
        trace_path = Path(args.trace).expanduser().resolve()
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        requests = sorted(
            trace.get("requests", ()), key=lambda item: float(item["arrival_ms"])
        )
        if not requests:
            raise ValueError(f"trace has no requests: {trace_path}")
        return [
            {
                "request_id": str(entry["request_id"]),
                "arrival_ms": float(entry["arrival_ms"]),
                "prompt": str(entry["prompt"]),
                "negative_prompt": str(entry.get("negative_prompt", "")),
                "width": int(entry["width"]),
                "height": int(entry["height"]),
                "num_frames": int(entry.get("num_frames", args.frames)),
                "num_steps": int(entry.get("num_steps", args.steps)),
                "seed": int(entry.get("seed", 0)),
                "deadline_ms": (
                    None
                    if entry.get("deadline_ms") is None
                    else float(entry["deadline_ms"])
                ),
                "priority": int(entry.get("priority", 0)),
            }
            for entry in requests
        ]
    prompts = (
        "A small sailboat crosses a calm alpine lake at sunrise, cinematic realism.",
        "A golden retriever runs through a field of wildflowers, slow motion.",
    )
    return [
        {
            "request_id": f"wan-bench-{index:03d}",
            "arrival_ms": 0.0,
            "prompt": prompts[index % len(prompts)],
            "negative_prompt": "",
            "width": args.width,
            "height": args.height,
            "num_frames": args.frames,
            "num_steps": args.steps,
            "seed": args.seed + index,
            "deadline_ms": None,
            "priority": 0,
        }
        for index in range(args.num_requests)
    ]


def _summary(
    rows: dict[str, dict[str, float | bool | None]],
) -> dict[str, float | int | None]:
    values = list(rows.values())
    latencies = [float(row["latency_ms"]) for row in values]
    queues = [float(row["queue_delay_ms"]) for row in values]
    services = [float(row["service_ms"]) for row in values]
    admissions = [float(row["admission_delay_ms"]) for row in values]
    slo_rows = [row for row in values if row["deadline_ms"] is not None]
    makespan_ms = max(float(row["finish_ms"]) for row in values)
    return {
        "num_requests": len(values),
        "makespan_ms": makespan_ms,
        "throughput_req_per_s": len(values) / (makespan_ms / 1000.0),
        "mean_latency_ms": statistics.fmean(latencies),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "max_latency_ms": max(latencies),
        "mean_queue_delay_ms": statistics.fmean(queues),
        "p95_queue_delay_ms": _percentile(queues, 0.95),
        "mean_admission_delay_ms": statistics.fmean(admissions),
        "mean_service_ms": statistics.fmean(services),
        "num_slo_requests": len(slo_rows),
        "num_slo_met": sum(bool(row["deadline_met"]) for row in slo_rows),
        "slo_attainment_rate": (
            sum(bool(row["deadline_met"]) for row in slo_rows) / len(slo_rows)
            if slo_rows
            else None
        ),
        "total_tardiness_ms": sum(float(row["tardiness_ms"]) for row in slo_rows),
        "max_tardiness_ms": max(
            (float(row["tardiness_ms"]) for row in slo_rows), default=0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the Wan benchmark requires CUDA")

    process_started = time.perf_counter()
    world = StageWorldSpec.from_torchrun("wan_benchmark")
    torch.cuda.set_device(int(str(world.local_device).rsplit(":", 1)[-1]))
    output_dir = Path(args.output_dir).expanduser().resolve()
    specs = _request_specs(args)
    widths = tuple(
        width
        for width in range(1, world.world_size + 1)
        if world.world_size % width == 0
    )
    warmup_resolutions = tuple(
        sorted({(int(spec["height"]), int(spec["width"])) for spec in specs})
    )
    warmup_profiles = tuple(
        sorted(
            {
                (
                    int(spec["height"]),
                    int(spec["width"]),
                    int(spec["num_frames"]),
                )
                for spec in specs
            }
        )
    )
    pool = HotSwitchPoolConfig(
        policy=args.strategy,
        allowed_lane_widths=widths,
        switch_allowed_until_step=max(int(spec["num_steps"]) for spec in specs),
        pulse_steps=args.pulse_steps,
        balanced_k=not args.no_balanced_k,
        max_inflight_requests=max(4, len(specs)),
        max_pending_requests=max(16, len(specs)),
        warmup_resolutions=warmup_resolutions,
        warmup_steps=args.warmup_steps,
    )
    runtime = EmbeddedDiffusionRuntime.create(
        world=world,
        pool=pool,
        executor_factory=WanExecutorFactory(
            model_path=args.model_path,
            torch_dtype=torch.bfloat16,
            default_width=int(specs[0]["width"]),
            default_height=int(specs[0]["height"]),
            default_num_frames=int(specs[0]["num_frames"]),
            default_num_steps=int(specs[0]["num_steps"]),
            cfg_parallel=True,
            flow_shift=args.flow_shift,
            parallel_vae=not args.no_parallel_vae,
            vae_parallel_halo=args.vae_parallel_halo,
            warmup_profiles=warmup_profiles,
        ),
        config=EmbeddedRuntimeConfig(
            output_root=str(output_dir),
            record_timeline=True,
        ),
    )
    model_loaded = time.perf_counter()
    runtime.start()
    runtime_started = time.perf_counter()
    graceful = False
    video_executor: ThreadPoolExecutor | None = None
    try:
        if not runtime.is_leader:
            runtime.wait()
            return
        if args.warmup_only:
            print(f"WAN_WARMUP_REPORT={output_dir / 'epe_warmup.json'}", flush=True)
            graceful = True
            return

        benchmark_started_at = time.time()
        benchmark_started = time.monotonic()
        specs_by_id = {str(spec["request_id"]): spec for spec in specs}
        if len(specs_by_id) != len(specs):
            raise ValueError("trace request IDs must be unique")
        output_dir.mkdir(parents=True, exist_ok=True)
        video_dir = output_dir / "videos"
        video_paths = {
            str(spec["request_id"]): video_dir
            / f"{index:03d}-{_safe_request_id(str(spec['request_id']))}.mp4"
            for index, spec in enumerate(specs)
        }
        if args.save_videos:
            video_dir.mkdir(parents=True, exist_ok=True)
            video_executor = ThreadPoolExecutor(
                max_workers=args.video_workers,
                thread_name_prefix="wan-video-export",
            )
        video_futures: dict[str, Future[dict[str, Any]]] = {}
        pending: set[str] = set()
        admissions: dict[str, dict[str, float]] = {}
        per_request: dict[str, dict[str, float | bool | None]] = {}
        videos: dict[str, dict[str, Any]] = {}
        next_spec = 0
        timeout_at = (
            benchmark_started
            + max(float(spec["arrival_ms"]) for spec in specs) / 1000.0
            + args.timeout_s
        )
        while (next_spec < len(specs) or pending) and time.monotonic() < timeout_at:
            elapsed_ms = (time.monotonic() - benchmark_started) * 1000.0
            while (
                next_spec < len(specs)
                and float(specs[next_spec]["arrival_ms"]) <= elapsed_ms
            ):
                spec = specs[next_spec]
                arrival_ms = float(spec["arrival_ms"])
                before_ms = (time.monotonic() - benchmark_started) * 1000.0
                request_id = runtime.submit(
                    WanRequest(
                        request_id=str(spec["request_id"]),
                        prompt=str(spec["prompt"]),
                        negative_prompt=str(spec["negative_prompt"]),
                        width=int(spec["width"]),
                        height=int(spec["height"]),
                        num_frames=int(spec["num_frames"]),
                        num_steps=int(spec["num_steps"]),
                        guidance_scale=args.guidance_scale,
                        seed=int(spec["seed"]),
                        deadline_ms=spec["deadline_ms"],
                        priority=int(spec["priority"]),
                    )
                )
                pending.add(request_id)
                admissions[request_id] = {
                    "trace_arrival_ms": arrival_ms,
                    "submit_start_ms": before_ms,
                    "submit_finish_ms": (time.monotonic() - benchmark_started) * 1000.0,
                }
                next_spec += 1
                elapsed_ms = (time.monotonic() - benchmark_started) * 1000.0

            poll_timeout_s = 0.1
            if next_spec < len(specs):
                until_next_s = (
                    float(specs[next_spec]["arrival_ms"]) - elapsed_ms
                ) / 1000.0
                poll_timeout_s = max(0.0, min(poll_timeout_s, until_next_s))
            for completion in runtime.poll(timeout_s=poll_timeout_s):
                if completion.request_id not in pending:
                    continue
                if completion.status != "completed":
                    detail = (
                        None if completion.error is None else completion.error.message
                    )
                    raise RuntimeError(
                        f"request {completion.request_id} failed: {detail}"
                    )
                spec = specs_by_id[completion.request_id]
                assert completion.submitted_at is not None
                assert completion.started_at is not None
                assert completion.completed_at is not None
                finish_ms = (completion.completed_at - benchmark_started_at) * 1000.0
                latency_ms = finish_ms - float(spec["arrival_ms"])
                deadline_ms = spec["deadline_ms"]
                tardiness_ms = (
                    max(0.0, latency_ms - float(deadline_ms))
                    if deadline_ms is not None
                    else 0.0
                )
                video = np.asarray(completion.output)
                contiguous = np.ascontiguousarray(video)
                videos[completion.request_id] = {
                    "array_sha256": hashlib.sha256(
                        memoryview(contiguous).cast("B")
                    ).hexdigest(),
                    "shape": list(video.shape),
                    "dtype": str(video.dtype),
                }
                if video_executor is not None:
                    submitted_at = time.time()
                    videos[completion.request_id]["encode_submitted_at"] = submitted_at
                    video_futures[completion.request_id] = video_executor.submit(
                        _export_video_async,
                        contiguous,
                        video_paths[completion.request_id],
                        submitted_at=submitted_at,
                        fps=args.video_fps,
                    )
                per_request[completion.request_id] = {
                    **admissions[completion.request_id],
                    "submitted_at": completion.submitted_at,
                    "started_at": completion.started_at,
                    "completed_at": completion.completed_at,
                    "finish_ms": finish_ms,
                    "latency_ms": latency_ms,
                    "queue_delay_ms": (completion.started_at - completion.submitted_at)
                    * 1000.0,
                    "admission_delay_ms": admissions[completion.request_id][
                        "submit_finish_ms"
                    ]
                    - float(spec["arrival_ms"]),
                    "service_ms": (completion.completed_at - completion.started_at)
                    * 1000.0,
                    "deadline_ms": deadline_ms,
                    "absolute_deadline_ms": (
                        None
                        if deadline_ms is None
                        else float(spec["arrival_ms"]) + float(deadline_ms)
                    ),
                    "deadline_met": None
                    if deadline_ms is None
                    else tardiness_ms == 0.0,
                    "tardiness_ms": tardiness_ms,
                }
                pending.remove(completion.request_id)
                del contiguous, video, completion
        if next_spec < len(specs) or pending:
            raise TimeoutError(
                "benchmark timed out with "
                f"{len(specs) - next_spec} unsubmitted and pending requests: {pending}"
            )

        for request_id, future in video_futures.items():
            videos[request_id].update(future.result())

        report = {
            "strategy": args.strategy,
            "world_size": world.world_size,
            "allowed_lane_widths": list(widths),
            "workload": {
                "num_requests": len(specs),
                "resolutions": [list(value) for value in warmup_resolutions],
                "frames": sorted({int(spec["num_frames"]) for spec in specs}),
                "steps": sorted({int(spec["num_steps"]) for spec in specs}),
                "guidance_scale": args.guidance_scale,
                "arrival_pattern": "trace" if args.trace else "closed_batch",
                "trace": str(Path(args.trace).resolve()) if args.trace else None,
                "duration_ms": max(float(spec["arrival_ms"]) for spec in specs),
                "save_videos": args.save_videos,
                "video_workers": args.video_workers if args.save_videos else 0,
                "video_fps": args.video_fps,
            },
            "timing": {
                "model_load_ms": (model_loaded - process_started) * 1000.0,
                "startup_warmup_ms": (runtime_started - model_loaded) * 1000.0,
            },
            "summary": _summary(per_request),
            "per_request": per_request,
            "videos": videos,
            "warmup": runtime.warmup_report,
            "health_after_benchmark": asdict(runtime.health()),
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            "WAN_BENCHMARK_RESULT",
            json.dumps(report["summary"], sort_keys=True),
            flush=True,
        )
        print(f"WAN_BENCHMARK_METRICS={metrics_path}", flush=True)
        graceful = True
    finally:
        if video_executor is not None:
            video_executor.shutdown(wait=True, cancel_futures=False)
        runtime.stop(graceful=graceful)


if __name__ == "__main__":
    main()
