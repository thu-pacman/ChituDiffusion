from __future__ import annotations

import os

from ..epac.image_decoder import DiffusionBackend
from .config import EPACServeConfig, StageServiceConfig
from .diffusion_runtime import EpeDiffusionServiceRuntime
from .torchrun import run_runtime


def _physical_gpu_ids(world_size: int) -> list[int]:
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if len(visible) == world_size and all(value.isdigit() for value in visible):
        return [int(value) for value in visible]
    return list(range(world_size))


def serve_diffusion_backend(
    backend: DiffusionBackend,
    *,
    model_path: str,
    config: EPACServeConfig,
    stage_name: str = "diffusion",
) -> None:
    """Run an already-loaded model backend with the production EPE lifecycle."""
    config.cache.require_serve_available()
    parallel = backend.parallel_context
    widths = tuple(parallel.allowed_widths)
    epe = backend.scheduling_module
    epe.policy = config.schedule_strategy
    epe.switch_allowed_until_step = config.switch_allowed_until_step
    epe.pulse_steps = config.pulse_steps
    epe.balanced_k = config.balanced_k
    epe.starvation_ms = config.starvation_ms
    epe.deadline_guard_ms = config.deadline_guard_ms
    epe.max_active_requests = config.max_inflight_requests
    epe.calibrator.enabled = config.online_calibration

    service_config = StageServiceConfig.from_mapping(
        {
            "name": stage_name,
            "process": stage_name,
            "factory": backend.model_name,
            "gpu": _physical_gpu_ids(parallel.world_size),
            "terminal": True,
            "parallelism": {
                "sp": parallel.world_size,
                "chitu_pool": {
                    "policy": config.schedule_strategy,
                    "allowed_lane_widths": list(widths),
                    "switch_allowed_until_step": (config.switch_allowed_until_step),
                    "pulse_steps": config.pulse_steps,
                    "balanced_k": config.balanced_k,
                    "starvation_ms": config.starvation_ms,
                    "default_deadline_ms": config.default_deadline_ms,
                    "deadline_guard_ms": config.deadline_guard_ms,
                    "max_inflight_requests": config.max_inflight_requests,
                    "max_pending_requests": config.max_pending_requests,
                    "warmup_resolutions": [
                        list(resolution) for resolution in config.warmup_resolutions
                    ],
                    "warmup_steps": config.warmup_steps,
                    "online_calibration": config.online_calibration,
                },
            },
            "factory_args": {
                "model_path": model_path,
                "num_steps": config.default_num_steps,
                "default_width": config.default_width,
                "default_height": config.default_height,
                "attention_mode": getattr(epe, "attention_mode", "agkv"),
                "ulysses_degree": parallel.active_usp.ulysses_degree,
                "cfg_parallel": bool(getattr(backend, "cfg_parallel", False)),
                "parallel_vae": config.parallel_vae,
                "vae_parallel_halo": config.vae_parallel_halo,
            },
            "service": {
                "host": config.host,
                "port": config.port,
                "advertise_host": config.advertise_host,
            },
            "output_root": config.output_root,
            "record_timeline": config.record_timeline,
            "postprocess_workers": config.postprocess_workers,
        }
    )
    runtime = EpeDiffusionServiceRuntime(
        service_config,
        backend,
    )
    run_runtime(runtime, service_config)
