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

from experiments.chitu_api.config import load_stage_service_config
from experiments.chitu_api.service import create_app

from .runtime import EpeZImageServiceRuntime

logger = logging.getLogger(__name__)

ACTION_RUN = 0
ACTION_IDLE = 1
ACTION_STOP = 2


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
    rank = runtime.rank
    stop_requested = threading.Event()
    if runtime.world_size > 1:
        runtime.parallel.barrier()

    def _stop(signum, _frame) -> None:
        logger.info("received signal %s", signum)
        stop_requested.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server = None
    server_thread = None
    device = (
        torch.device("cuda", runtime.parallel.local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    startup = torch.ones(1, dtype=torch.int64, device=device)
    startup_error: BaseException | None = None
    if rank == 0:
        try:
            server, server_thread = _start_http(runtime, config.service.host, config.service.port)
        except BaseException as exc:
            startup[0] = 0
            startup_error = exc
    if runtime.world_size > 1:
        dist.broadcast(startup, src=0)
    if int(startup.item()) == 0:
        runtime.close()
        raise RuntimeError("rank 0 failed to start HTTP service") from startup_error

    if rank == 0:
        logger.info(
            "CHITU_API_READY %s",
            json.dumps(
                {
                    "stage": config.name,
                    "endpoint": config.service.endpoint,
                    "physical_gpu_ids": list(config.gpu),
                    "world_size": runtime.world_size,
                    "allowed_lane_widths": list(runtime.parallel.allowed_widths),
                    "runtime": "standalone_diffusers",
                },
                sort_keys=True,
            ),
        )

    action = torch.zeros(1, dtype=torch.int64, device=device)
    try:
        while True:
            if rank == 0:
                if stop_requested.is_set():
                    action[0] = ACTION_STOP
                elif runtime.has_work():
                    action[0] = ACTION_RUN
                else:
                    action[0] = ACTION_IDLE
            if runtime.world_size > 1:
                dist.broadcast(action, src=0)
            current = int(action.item())
            if current == ACTION_STOP:
                break
            if current == ACTION_IDLE:
                if rank == 0:
                    time.sleep(0.001)
                continue
            runtime.process_next()
    except BaseException as exc:
        if rank == 0:
            runtime.fail(exc)
        raise
    finally:
        if rank == 0 and server is not None:
            server.should_exit = True
        if rank == 0 and server_thread is not None:
            server_thread.join(timeout=10.0)
        runtime.close()


def run(config_path: str) -> None:
    run_config(load_stage_service_config(config_path))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Standalone Diffusers Z-Image EPE service")
    parser.add_argument("--stage-config", required=True)
    args = parser.parse_args()
    run(args.stage_config)


if __name__ == "__main__":
    main()
