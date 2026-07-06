#!/usr/bin/env python3
"""Server-side replayer for a Poisson client trace (end-to-end serving harness).

This is the *server*: it loads a client-arrival trace (see
``experiments/cp_dp_hot_switch/gen_client_trace.py``) and injects each request
into the ChituDiffusion task pool exactly when wall-clock reaches that request's
``arrival_ms``, while continuously driving the engine. This exposes the engine
to genuine dynamic load (queueing, overlapping in-flight requests) instead of
the front-loaded static batch that ``test/test_z_image.py`` uses -- which is the
scenario continuous batching (M3/M4) and the throughput scheduler (M7) must be
validated against.

It records per-request wall-clock timestamps (arrival / first-scheduled /
finish) and emits an end-to-end serving metrics JSON (throughput, latency
percentiles, queue delay, GPU busy ratio proxy) so runs with the engine's
``continuous_batch`` flag on vs off can be compared apples-to-apples.

Single-node, single-rank (SP=1 / world_size==1) for now: multi-rank idle
synchronization (ranks staying in lockstep while the pool is momentarily empty
between arrivals) needs an engine-level idle barrier and is deferred to the
dynamic-SP milestone. Drive via:

    CHITU_PYTHON_BIN=<venv-python> CHITU_SERVE_TRACE=<trace.json> \
        chitu run test/configs/z_image_serve.yaml
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from logging import getLogger
from pathlib import Path

import torch

from chitu_diffusion.core.config_loader import load_config_from_cli
from chitu_diffusion.core.schemas import ServeConfig
from chitu_diffusion.observability import Timer
from chitu_diffusion.runtime.main import (
    chitu_generate,
    chitu_init,
    chitu_is_terminated,
    chitu_start,
    chitu_terminate,
    warmup_diffusion_engine,
)
from chitu_diffusion.runtime.backend import DiffusionBackend
from chitu_diffusion.runtime.output_layout import metrics_dir, write_json
from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskPool,
    DiffusionTaskStatus,
    DiffusionUserParams,
    DiffusionUserRequest,
)
from run_context import DiffusionTestRunContext, should_record_metrics_on_rank

logger = getLogger(__name__)


def _load_trace(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    requests = list(data.get("requests", []))
    requests.sort(key=lambda r: float(r["arrival_ms"]))
    return requests


def _build_request(entry: dict, default_steps: int) -> DiffusionUserRequest:
    return DiffusionUserRequest(
        request_id=str(entry["request_id"]),
        params=DiffusionUserParams(
            role="client",
            prompt=str(entry["prompt"]),
            negative_prompt=entry.get("negative_prompt"),
            seed=int(entry.get("seed", 0)),
            frame_num=1,
            size=(int(entry["width"]), int(entry["height"])),
            num_inference_steps=int(entry.get("num_steps", default_steps)),
            n_sample=int(entry.get("n_sample", 1)),
            sample_solver=str(entry.get("sample_solver", "flowmatch_euler")),
        ),
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _summarize(records: dict[str, dict], wall_elapsed_s: float, num_images: int) -> dict:
    # Anchor client-perceived timings on the intended (trace) arrival, not the
    # moment the harness happened to inject the request into the pool. Injection
    # only occurs at engine stage boundaries (Z-Image denoise is re-entrant), so
    # a request may be admitted a bit after its true arrival; that gap is the
    # admission delay and is reported separately.
    completed = [r for r in records.values() if "finish_ms" in r]
    latencies = [r["finish_ms"] - r["trace_arrival_ms"] for r in completed]
    queue_delays = [
        r["first_sched_ms"] - r["trace_arrival_ms"]
        for r in completed
        if r.get("first_sched_ms") is not None
    ]
    admission_delays = [
        r["admit_ms"] - r["trace_arrival_ms"]
        for r in completed
        if r.get("admit_ms") is not None
    ]
    makespan_ms = 0.0
    if completed:
        first_arrival = min(r["trace_arrival_ms"] for r in completed)
        last_finish = max(r["finish_ms"] for r in completed)
        makespan_ms = last_finish - first_arrival
    thrpt_req = (len(completed) / (makespan_ms / 1000.0)) if makespan_ms > 0 else 0.0
    thrpt_img = (num_images / (makespan_ms / 1000.0)) if makespan_ms > 0 else 0.0
    return {
        "num_requests": len(records),
        "num_completed": len(completed),
        "num_images": num_images,
        "wall_elapsed_s": round(wall_elapsed_s, 4),
        "makespan_ms": round(makespan_ms, 3),
        "throughput_req_per_s": round(thrpt_req, 4),
        "throughput_img_per_s": round(thrpt_img, 4),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "p50_latency_ms": round(_percentile(latencies, 50), 3),
        "p95_latency_ms": round(_percentile(latencies, 95), 3),
        "p99_latency_ms": round(_percentile(latencies, 99), 3),
        "max_latency_ms": round(max(latencies), 3) if latencies else 0.0,
        "mean_queue_delay_ms": round(sum(queue_delays) / len(queue_delays), 3) if queue_delays else 0.0,
        "p95_queue_delay_ms": round(_percentile(queue_delays, 95), 3),
        "mean_admission_delay_ms": round(sum(admission_delays) / len(admission_delays), 3) if admission_delays else 0.0,
        "max_admission_delay_ms": round(max(admission_delays), 3) if admission_delays else 0.0,
    }


def main(args: ServeConfig):
    if args.models.name != "Z-Image":
        raise ValueError(f"test/test_z_image_serve.py expects models=Z-Image, got {args.models.name}.")

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            f"test_z_image_serve currently supports world_size==1 (SP=1 single GPU), got {world_size}. "
            "Multi-rank idle synchronization between arrivals is a future engine feature."
        )

    trace_path = os.getenv("CHITU_SERVE_TRACE", "").strip()
    if not trace_path:
        raise ValueError("Set CHITU_SERVE_TRACE=<client trace json> (see gen_client_trace.py).")

    logger.setLevel(logging.INFO)
    default_steps = int(args.models.sampler.sample_steps)
    run_context = DiffusionTestRunContext(args, logger, per_task_results=True, capture_stdio_log=True)
    initial_run_output_dir = run_context.build_initial_run_output_dir()
    run_context.activate_run(initial_run_output_dir)

    early_log_handler = None
    if getattr(args.output, "run_log", True) and should_record_metrics_on_rank():
        early_log_handler = run_context.attach_run_log_handler(initial_run_output_dir)
        os.environ["CHITU_RUN_LOG_ATTACHED"] = "1"

    try:
        chitu_init(args, logging_level=logging.INFO)
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        rank = torch.distributed.get_rank()

        trace = _load_trace(trace_path)
        reqs = [_build_request(entry, default_steps) for entry in trace]
        run_output_dir = run_context.build_run_output_dir(reqs)
        run_context.activate_run(run_output_dir)
        run_context.apply_request_output_dirs(run_output_dir, reqs)
        run_context.dump_run_metadata(run_output_dir, reqs)
        req_by_id = {req.request_id: req for req in reqs}
        num_images = sum(int(r.params.n_sample) for r in reqs)
        logger.info(
            "[serve] loaded trace=%s requests=%d images=%d duration=%.1fms",
            trace_path,
            len(trace),
            num_images,
            trace[-1]["arrival_ms"] if trace else 0.0,
        )

        run_context.dump_memory_snapshot(run_output_dir, "model_loaded")
        warmup_diffusion_engine(args)
        chitu_start()

        records: dict[str, dict] = {}
        start = time.time()
        injected = 0
        n_trace = len(trace)

        def _now_ms() -> float:
            return (time.time() - start) * 1000.0

        def _inject_due(now_ms: float) -> None:
            nonlocal injected
            while injected < n_trace and float(trace[injected]["arrival_ms"]) <= now_ms:
                entry = trace[injected]
                req = req_by_id[str(entry["request_id"])]
                DiffusionTaskPool.add(DiffusionTask(task_id=req.request_id, req=req))
                records[req.request_id] = {"admit_ms": now_ms, "trace_arrival_ms": float(entry["arrival_ms"])}
                logger.info("[serve] admit request=%s at %.1fms (%d/%d)", req.request_id, now_ms, injected + 1, n_trace)
                injected += 1

        def _harvest_completions(now_ms: float) -> None:
            for task_id, task in DiffusionTaskPool.pool.items():
                rec = records.get(task_id)
                if rec is None or "finish_ms" in rec:
                    continue
                if rec.get("first_sched_ms") is None and task.sched_ts is not None:
                    rec["first_sched_ms"] = now_ms  # first time we observe it scheduled
                if task.is_completed():
                    rec["finish_ms"] = now_ms

        scheduler = DiffusionBackend.scheduler
        with Timer.get_timer("overall"):
            while not chitu_is_terminated():
                now_ms = _now_ms()
                _inject_due(now_ms)
                # Mark first-scheduled the moment a pending task becomes schedulable.
                if scheduler.can_schedule():
                    for task_id in DiffusionTaskPool.pending_task_ids():
                        rec = records.get(task_id)
                        if rec is not None and rec.get("first_sched_ms") is None:
                            rec["first_sched_ms"] = now_ms
                    chitu_generate()
                    _harvest_completions(_now_ms())
                elif injected < n_trace:
                    # Idle: no work now but future arrivals pending -> wait briefly.
                    time.sleep(0.001)
                    continue
                else:
                    break
                if injected >= n_trace and DiffusionTaskPool.all_finished():
                    _harvest_completions(_now_ms())
                    break

        elapsed_s = time.time() - start
        summary = _summarize(records, elapsed_s, num_images)
        logger.info("[serve] DONE %s", json.dumps(summary))

        if rank == 0:
            out = {
                "trace": os.path.abspath(trace_path),
                "continuous_batch": bool(getattr(args.infer.diffusion, "continuous_batch", False)),
                "scheduling_policy": str(getattr(args.infer.diffusion, "scheduling_policy", "fifo")),
                "summary": summary,
                "per_request": records,
            }
            write_json(os.path.join(metrics_dir(run_output_dir), "serve_metrics.json"), out)
            logger.info("[serve] metrics -> %s", os.path.join(metrics_dir(run_output_dir), "serve_metrics.json"))
            if getattr(args.output, "timer", False):
                Timer.print_statistics()
                run_context.dump_timing_summary(run_output_dir, elapsed_s=elapsed_s)

        run_context.dump_memory_snapshot(run_output_dir, "final")
        chitu_terminate()
    finally:
        if early_log_handler is not None:
            run_context.close_run_log_handler(early_log_handler)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(load_config_from_cli())
