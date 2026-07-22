from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time

import torch
import torch.distributed as dist
import uvicorn

from ..epac.image_decoder import EmbeddedRuntimeConfig, StageWorldSpec
from .app import create_app
from .config import load_stage_service_config
from .embedded import EmbeddedDiffusionRuntime
from .zimage_runtime import EpeZImageServiceRuntime

logger = logging.getLogger(__name__)


def _start_http(runtime, host: str, port: int):
    server = uvicorn.Server(
        uvicorn.Config(create_app(runtime), host=host, port=port, log_level="info")
    )
    thread = threading.Thread(target=server.run, name="zimage-http", daemon=True)
    thread.start()
    deadline = time.monotonic() + 30.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError(f"HTTP server did not start on {host}:{port}")
    return server, thread


def run_config(config) -> None:
    logging.basicConfig(level=logging.INFO)
    runtime = EpeZImageServiceRuntime.from_config(config)

    run_runtime(runtime, config)


def _as_embedded_runtime(runtime, config) -> EmbeddedDiffusionRuntime:
    if isinstance(runtime, EmbeddedDiffusionRuntime):
        return runtime
    world = StageWorldSpec(
        stage_name=config.name,
        rank=runtime.rank,
        world_size=runtime.world_size,
        local_device=(
            f"cuda:{runtime.parallel.local_rank}"
            if torch.cuda.is_available()
            else "cpu"
        ),
        physical_device_ids=tuple(config.gpu),
        process_group=dist.group.WORLD if dist.is_initialized() else None,
        owns_process_group=False,
    )
    return EmbeddedDiffusionRuntime.from_backend(
        world=world,
        pool=runtime.pool,
        executor=runtime.executor,
        backend=runtime,
        config=EmbeddedRuntimeConfig(
            output_root=config.output_root,
            record_timeline=config.record_timeline,
            postprocess_workers=config.postprocess_workers,
        ),
    )


def run_runtime(
    runtime: EpeZImageServiceRuntime | EmbeddedDiffusionRuntime, config
) -> None:
    """Run an already-loaded pipeline/runtime on every torchrun rank."""
    embedded = _as_embedded_runtime(runtime, config)
    backend = embedded.service_backend
    rank = embedded.world.rank
    stop_requested = threading.Event()
    embedded.start()
    warmup_report = embedded.warmup_report
    assert warmup_report is not None

    def _stop(signum, _frame) -> None:
        logger.info("received signal %s", signum)
        stop_requested.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server = None
    server_thread = None
    device = (
        torch.device("cuda", backend.parallel.local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    startup = torch.ones(1, dtype=torch.int64, device=device)
    startup_error: BaseException | None = None
    if rank == 0:
        try:
            server, server_thread = _start_http(
                backend, config.service.host, config.service.port
            )
        except BaseException as exc:
            startup[0] = 0
            startup_error = exc
    if embedded.world.world_size > 1:
        dist.broadcast(startup, src=0)
    if int(startup.item()) == 0:
        embedded.stop(graceful=False)
        raise RuntimeError("rank 0 failed to start HTTP service") from startup_error

    if rank == 0:
        logger.info(
            "CHITU_API_READY %s",
            json.dumps(
                {
                    "stage": config.name,
                    "endpoint": config.service.endpoint,
                    "physical_gpu_ids": list(config.gpu),
                    "world_size": backend.world_size,
                    "allowed_lane_widths": list(backend.parallel.allowed_widths),
                    "epe": {
                        "policy": backend.epe.policy,
                        "cost_model": backend.epe.cost_model.source,
                        "cost_model_ready": backend.epe.cost_model.ready,
                        "online_calibration": backend.epe.calibrator.enabled,
                        "balanced_k": backend.epe.balanced_k,
                        "pulse_steps": backend.epe.pulse_steps,
                        "starvation_ms": backend.epe.starvation_ms,
                        "default_deadline_ms": (
                            config.parallelism.chitu_pool.default_deadline_ms
                        ),
                        "deadline_guard_ms": backend.epe.deadline_guard_ms,
                        "warmup": {
                            "steps": warmup_report["steps"],
                            "resolutions": warmup_report["resolutions"],
                            "total_wall_ms": warmup_report["total_wall_ms"],
                            "profile": [
                                {
                                    "resolution": row["resolution"],
                                    "image_tokens": row["image_tokens"],
                                    "width": row["width"],
                                    "latency_ms": row["latency_ms"],
                                }
                                for row in warmup_report["rows"]
                            ],
                        },
                    },
                    "runtime": "standalone_diffusers",
                    "execution_transport": (
                        "singleton_producer_consumer"
                        if backend.uses_singleton_worker_pool
                        else "full_world_producer_consumer"
                        if backend.uses_full_world_worker_pool
                        else "elastic_lane_exchange"
                    ),
                },
                sort_keys=True,
            ),
        )

    try:
        if rank == 0:
            while embedded.worker_alive and not stop_requested.is_set():
                time.sleep(0.01)
            if stop_requested.is_set():
                embedded.stop(graceful=True, timeout_s=60.0)
            else:
                embedded.wait()
        else:
            embedded.wait()
    except BaseException as exc:
        if rank == 0:
            backend.fail(exc)
        raise
    finally:
        if rank == 0 and server is not None:
            server.should_exit = True
        if rank == 0 and server_thread is not None:
            server_thread.join(timeout=10.0)
        if embedded.worker_alive:
            embedded.stop(graceful=False, timeout_s=60.0)
        else:
            embedded.stop(graceful=True, timeout_s=60.0)


def run(config_path: str) -> None:
    run_config(load_stage_service_config(config_path))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Standalone Diffusers Z-Image EPE service"
    )
    parser.add_argument("--stage-config", required=True)
    args = parser.parse_args()
    run(args.stage_config)


if __name__ == "__main__":
    main()
