from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from chitu_diffusers import (
    EmbeddedDiffusionRuntime,
    EmbeddedRuntimeConfig,
    Flux1ExecutorFactory,
    Flux1Request,
    HotSwitchPoolConfig,
    StageWorldSpec,
)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _parse_resolutions(value: str) -> tuple[int, ...]:
    resolutions = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not resolutions:
        raise argparse.ArgumentTypeError("at least one resolution is required")
    if any(resolution < 16 or resolution % 16 for resolution in resolutions):
        raise argparse.ArgumentTypeError("resolutions must be positive multiples of 16")
    return resolutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FLUX.1 through the four-GPU embedded EPAC runtime"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--strategy",
        choices=("static_dp", "static_cp", "elastic"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--trace",
        help="Replay a Z-Image benchmark trace instead of the closed batch",
    )
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--resolutions", type=_parse_resolutions, default=(512, 1024))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--pulse-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="Run measured startup warmup, write epe_warmup.json, and exit",
    )
    parser.add_argument("--no-parallel-vae", action="store_true")
    parser.add_argument("--vae-parallel-halo", type=int, default=8)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    if args.num_requests < 1:
        parser.error("--num-requests must be positive")
    if min(args.steps, args.pulse_steps) < 1:
        parser.error("step counts must be positive")
    if args.warmup_steps < 3:
        parser.error("--warmup-steps must be >= 3")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    if args.vae_parallel_halo < 0:
        parser.error("--vae-parallel-halo must be non-negative")
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
                "width": int(entry["width"]),
                "height": int(entry["height"]),
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
        "A red cube on a white table, studio lighting",
        "A glass teapot beside green leaves, product photography",
        "A small cabin beside a frozen lake at sunrise",
        "A futuristic train entering a clean white station",
    )
    return [
        {
            "request_id": f"flux-bench-{index:03d}",
            "arrival_ms": 0.0,
            "prompt": prompts[index % len(prompts)],
            "width": args.resolutions[index % len(args.resolutions)],
            "height": args.resolutions[index % len(args.resolutions)],
            "seed": args.seed + index,
            "deadline_ms": None,
            "priority": 0,
        }
        for index in range(args.num_requests)
    ]


def _build_summary(
    completions: dict[str, Any],
    *,
    specs_by_id: dict[str, dict[str, Any]],
    benchmark_started_at: float,
    admissions: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_request: dict[str, Any] = {}
    latencies = []
    queue_delays = []
    service_times = []
    admission_delays = []
    finishes = []
    slo_results = []
    for request_id, completion in sorted(completions.items()):
        assert completion.submitted_at is not None
        assert completion.started_at is not None
        assert completion.completed_at is not None
        spec = specs_by_id[request_id]
        arrival_ms = float(spec["arrival_ms"])
        finish_ms = (completion.completed_at - benchmark_started_at) * 1000.0
        latency_ms = finish_ms - arrival_ms
        queue_delay_ms = (completion.started_at - completion.submitted_at) * 1000.0
        service_ms = (completion.completed_at - completion.started_at) * 1000.0
        admission_delay_ms = admissions[request_id]["submit_finish_ms"] - arrival_ms
        latencies.append(latency_ms)
        queue_delays.append(queue_delay_ms)
        service_times.append(service_ms)
        admission_delays.append(admission_delay_ms)
        finishes.append(finish_ms)
        deadline_ms = spec.get("deadline_ms")
        deadline_metrics = {}
        if deadline_ms is not None:
            tardiness_ms = max(0.0, latency_ms - float(deadline_ms))
            deadline_met = tardiness_ms == 0.0
            slo_results.append((deadline_met, tardiness_ms))
            deadline_metrics = {
                "deadline_ms": float(deadline_ms),
                "absolute_deadline_ms": arrival_ms + float(deadline_ms),
                "deadline_met": deadline_met,
                "tardiness_ms": tardiness_ms,
            }
        per_request[request_id] = {
            **admissions[request_id],
            "submitted_at": completion.submitted_at,
            "started_at": completion.started_at,
            "completed_at": completion.completed_at,
            "finish_ms": finish_ms,
            "latency_ms": latency_ms,
            "queue_delay_ms": queue_delay_ms,
            "admission_delay_ms": admission_delay_ms,
            "service_ms": service_ms,
            **deadline_metrics,
        }

    count = len(completions)
    benchmark_wall_ms = max(finishes) - min(
        float(spec["arrival_ms"]) for spec in specs_by_id.values()
    )
    summary = {
        "num_requests": count,
        "makespan_ms": benchmark_wall_ms,
        "throughput_req_per_s": count / (benchmark_wall_ms / 1000.0),
        "mean_latency_ms": statistics.fmean(latencies),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "max_latency_ms": max(latencies),
        "mean_queue_delay_ms": statistics.fmean(queue_delays),
        "p95_queue_delay_ms": _percentile(queue_delays, 0.95),
        "mean_admission_delay_ms": statistics.fmean(admission_delays),
        "mean_service_ms": statistics.fmean(service_times),
        "num_slo_requests": len(slo_results),
        "num_slo_met": sum(met for met, _ in slo_results),
        "slo_attainment_rate": (
            sum(met for met, _ in slo_results) / len(slo_results)
            if slo_results
            else None
        ),
        "total_tardiness_ms": sum(value for _, value in slo_results),
        "max_tardiness_ms": max((value for _, value in slo_results), default=0.0),
    }
    return summary, per_request


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the Flux.1 benchmark requires CUDA")

    process_started = time.perf_counter()
    world = StageWorldSpec.from_torchrun("flux1_benchmark")
    if world.world_size != 4:
        raise ValueError("the comparison benchmark requires exactly four ranks")
    torch.cuda.set_device(int(str(world.local_device).rsplit(":", 1)[-1]))

    output_dir = Path(args.output_dir).expanduser().resolve()
    widths = (1, 2, 4)
    specs = _request_specs(args)
    warmup_resolutions = tuple(
        sorted({(int(spec["height"]), int(spec["width"])) for spec in specs})
    )
    pool = HotSwitchPoolConfig(
        policy=args.strategy,
        allowed_lane_widths=widths,
        switch_allowed_until_step=args.steps,
        pulse_steps=args.pulse_steps,
        max_inflight_requests=max(4, len(specs)),
        max_pending_requests=max(16, len(specs)),
        warmup_resolutions=warmup_resolutions,
        warmup_steps=args.warmup_steps,
    )
    runtime = EmbeddedDiffusionRuntime.create(
        world=world,
        pool=pool,
        executor_factory=Flux1ExecutorFactory(
            model_path=args.model_path,
            torch_dtype=torch.bfloat16,
            default_width=int(specs[0]["width"]),
            default_height=int(specs[0]["height"]),
            default_num_steps=args.steps,
            parallel_vae=not args.no_parallel_vae,
            vae_parallel_halo=args.vae_parallel_halo,
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
    try:
        if not runtime.is_leader:
            runtime.wait()
            return

        if args.warmup_only:
            warmup_path = output_dir / "epe_warmup.json"
            print(f"FLUX1_WARMUP_REPORT={warmup_path}", flush=True)
            graceful = True
            return

        benchmark_started_at = time.time()
        benchmark_started = time.monotonic()
        request_ids = []
        admissions = {}
        for spec in specs:
            arrival_ms = float(spec["arrival_ms"])
            wait_s = benchmark_started + arrival_ms / 1000.0 - time.monotonic()
            if wait_s > 0:
                time.sleep(wait_s)
            before_ms = (time.monotonic() - benchmark_started) * 1000.0
            request_ids.append(
                runtime.submit(
                    Flux1Request(
                        request_id=str(spec["request_id"]),
                        prompt=str(spec["prompt"]),
                        width=int(spec["width"]),
                        height=int(spec["height"]),
                        num_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        seed=int(spec["seed"]),
                        deadline_ms=spec.get("deadline_ms"),
                        priority=int(spec.get("priority", 0)),
                    )
                )
            )
            admissions[str(spec["request_id"])] = {
                "trace_arrival_ms": arrival_ms,
                "submit_start_ms": before_ms,
                "submit_finish_ms": (
                    time.monotonic() - benchmark_started
                ) * 1000.0,
            }

        pending = set(request_ids)
        completions: dict[str, Any] = {}
        deadline = time.monotonic() + args.timeout_s
        while pending and time.monotonic() < deadline:
            for completion in runtime.poll(timeout_s=0.1):
                if completion.request_id not in pending:
                    continue
                if completion.status != "completed":
                    detail = (
                        None if completion.error is None else completion.error.message
                    )
                    raise RuntimeError(
                        f"request {completion.request_id} failed: {detail}"
                    )
                pending.remove(completion.request_id)
                completions[completion.request_id] = completion
        if pending:
            raise TimeoutError(f"benchmark timed out with pending requests: {pending}")
        output_dir.mkdir(parents=True, exist_ok=True)
        specs_by_id = {str(spec["request_id"]): spec for spec in specs}
        image_hashes = {}
        for request_id, completion in completions.items():
            image = completion.output
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            image_hashes[request_id] = {
                "pixel_sha256": digest,
                "size": list(image.size),
                "width": int(specs_by_id[request_id]["width"]),
                "height": int(specs_by_id[request_id]["height"]),
                "seed": int(specs_by_id[request_id]["seed"]),
            }
            if args.save_images:
                image.save(output_dir / f"{request_id}.png")

        summary, per_request = _build_summary(
            completions,
            specs_by_id=specs_by_id,
            benchmark_started_at=benchmark_started_at,
            admissions=admissions,
        )
        report = {
            "strategy": args.strategy,
            "world_size": world.world_size,
            "allowed_lane_widths": list(widths),
            "workload": {
                "num_requests": len(specs),
                "resolutions": [list(value) for value in warmup_resolutions],
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "base_seed": args.seed,
                "arrival_pattern": "trace" if args.trace else "closed_batch",
                "trace": (
                    str(Path(args.trace).expanduser().resolve()) if args.trace else None
                ),
                "duration_ms": max(float(spec["arrival_ms"]) for spec in specs),
            },
            "timing": {
                "model_load_ms": (model_loaded - process_started) * 1000.0,
                "startup_warmup_ms": (runtime_started - model_loaded) * 1000.0,
            },
            "summary": summary,
            "per_request": per_request,
            "images": image_hashes,
            "warmup": runtime.warmup_report,
            "health_after_benchmark": asdict(runtime.health()),
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("FLUX1_BENCHMARK_RESULT", json.dumps(summary, sort_keys=True), flush=True)
        print(f"FLUX1_BENCHMARK_METRICS={metrics_path}", flush=True)
        graceful = True
    finally:
        runtime.stop(graceful=graceful)


if __name__ == "__main__":
    main()
