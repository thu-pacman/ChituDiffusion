from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import socket
from typing import Any, Mapping

import yaml


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


@dataclass(frozen=True)
class HotSwitchPoolConfig:
    policy: str = "slo_elastic"
    allowed_lane_widths: tuple[int, ...] = (1,)
    switch_allowed_until_step: int = 0
    phase_max_steps: int = 1
    max_inflight_requests: int = 64
    max_pending_requests: int = 256
    cost_model: str = "auto"
    online_calibration: bool = True

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, world_size: int
    ) -> "HotSwitchPoolConfig":
        widths = tuple(int(value) for value in raw.get("allowed_lane_widths", (1,)))
        if not widths:
            raise ValueError("allowed_lane_widths must not be empty")
        if tuple(sorted(set(widths))) != widths:
            raise ValueError("allowed_lane_widths must be sorted and unique")
        if widths[0] != 1:
            raise ValueError("allowed_lane_widths must include width=1")
        invalid = [width for width in widths if width < 1 or world_size % width != 0]
        if invalid:
            raise ValueError(
                f"lane widths {invalid} must be positive divisors of world size {world_size}"
            )

        phase_max_steps = int(raw.get("phase_max_steps", 1))
        max_inflight = int(raw.get("max_inflight_requests", 64))
        max_pending = int(raw.get("max_pending_requests", 256))
        switch_until = int(raw.get("switch_allowed_until_step", 0))
        if phase_max_steps < 1:
            raise ValueError("phase_max_steps must be >= 1")
        if max_inflight < 1 or max_pending < 1:
            raise ValueError("request limits must be >= 1")
        if switch_until < 0:
            raise ValueError("switch_allowed_until_step must be >= 0")

        return cls(
            policy=str(raw.get("policy", "slo_elastic")),
            allowed_lane_widths=widths,
            switch_allowed_until_step=switch_until,
            phase_max_steps=phase_max_steps,
            max_inflight_requests=max_inflight,
            max_pending_requests=max_pending,
            cost_model=str(raw.get("cost_model", "auto")),
            online_calibration=bool(raw.get("online_calibration", True)),
        )


@dataclass(frozen=True)
class ParallelismConfig:
    sp: int
    chitu_pool: HotSwitchPoolConfig

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, gpu_count: int
    ) -> "ParallelismConfig":
        # ``tp`` is accepted only as an old Omni StageConfig compatibility alias.
        sp = int(raw.get("sp", raw.get("tp", gpu_count)))
        if sp != gpu_count:
            raise ValueError(
                f"parallelism.sp ({sp}) must equal the number of stage GPUs ({gpu_count})"
            )
        pool_raw = _mapping(raw.get("chitu_pool", {}), "parallelism.chitu_pool")
        return cls(
            sp=sp,
            chitu_pool=HotSwitchPoolConfig.from_mapping(pool_raw, world_size=sp),
        )


@dataclass(frozen=True)
class ZImageFactoryConfig:
    model_path: str
    num_steps: int = 20
    sample_solver: str = "flowmatch_euler"
    default_width: int = 1024
    default_height: int = 1024

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ZImageFactoryConfig":
        model_path = str(raw.get("model_path", "")).strip()
        if not model_path:
            raise ValueError("factory_args.model_path is required")
        num_steps = int(raw.get("num_steps", 20))
        width = int(raw.get("default_width", 1024))
        height = int(raw.get("default_height", 1024))
        if num_steps < 1:
            raise ValueError("factory_args.num_steps must be >= 1")
        if width < 16 or height < 16 or width % 16 or height % 16:
            raise ValueError("default image dimensions must be positive multiples of 16")
        return cls(
            model_path=model_path,
            num_steps=num_steps,
            sample_solver=str(raw.get("sample_solver", "flowmatch_euler")),
            default_width=width,
            default_height=height,
        )


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 18080
    advertise_host: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ServiceConfig":
        port = int(raw.get("port", 18080))
        if not 1024 <= port <= 65535:
            raise ValueError("service.port must be between 1024 and 65535")
        advertise_host = raw.get("advertise_host")
        return cls(
            host=str(raw.get("host", "0.0.0.0")),
            port=port,
            advertise_host=str(advertise_host) if advertise_host else None,
        )

    @property
    def endpoint(self) -> str:
        host = self.advertise_host or (
            "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        )
        return f"http://{host}:{self.port}"


@dataclass(frozen=True)
class StageServiceConfig:
    name: str
    process: str
    factory: str
    gpu: tuple[int, ...]
    parallelism: ParallelismConfig
    factory_args: ZImageFactoryConfig
    service: ServiceConfig
    terminal: bool = True
    output_root: str = "outputs/chitu-api"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StageServiceConfig":
        gpu_raw = raw.get("gpu")
        if not isinstance(gpu_raw, (list, tuple)) or not gpu_raw:
            raise ValueError("gpu must be a non-empty list of physical GPU ids")
        gpu = tuple(int(value) for value in gpu_raw)
        if len(set(gpu)) != len(gpu):
            raise ValueError("gpu ids must be unique")
        if any(value < 0 for value in gpu):
            raise ValueError("gpu ids must be non-negative")

        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("name is required")
        process = str(raw.get("process", name)).strip()
        factory = str(raw.get("factory", "zimage")).strip()
        if factory not in {"zimage", "chitu_diffusion.zimage"}:
            raise ValueError(f"unsupported experimental factory: {factory}")

        return cls(
            name=name,
            process=process,
            factory=factory,
            gpu=gpu,
            parallelism=ParallelismConfig.from_mapping(
                _mapping(raw.get("parallelism", {}), "parallelism"),
                gpu_count=len(gpu),
            ),
            factory_args=ZImageFactoryConfig.from_mapping(
                _mapping(raw.get("factory_args", {}), "factory_args")
            ),
            service=ServiceConfig.from_mapping(
                _mapping(raw.get("service", {}), "service")
            ),
            terminal=bool(raw.get("terminal", True)),
            output_root=str(raw.get("output_root", "outputs/chitu-api")),
        )

    def chitu_overrides(self) -> list[str]:
        pool = self.parallelism.chitu_pool
        widths = ",".join(str(width) for width in pool.allowed_lane_widths)
        cost_model = pool.cost_model
        if cost_model == "auto":
            cost_model = "rtx4090"
        return [
            "models=Z-Image",
            f"models.ckpt_dir={self.factory_args.model_path}",
            "infer.diffusion.cfg_size=1",
            f"infer.diffusion.cp_size={self.parallelism.sp}",
            "infer.diffusion.up=1",
            "infer.diffusion.topology_mode=elastic",
            f"infer.diffusion.elastic_allowed_widths=[{widths}]",
            f"infer.diffusion.scheduling_policy={pool.policy}",
            "infer.diffusion.dynamic_sp=true",
            "infer.diffusion.epe.cfg_parallel_max=1",
            f"infer.diffusion.epe.phase_max_steps={pool.phase_max_steps}",
            f"infer.diffusion.epe.online_calibration={str(pool.online_calibration).lower()}",
            f"infer.diffusion.switch_allowed_until_step={pool.switch_allowed_until_step}",
            f"infer.diffusion.cost_model={cost_model}",
            f"models.sampler.sample_steps={self.factory_args.num_steps}",
            f"output.root_dir={self.output_root}",
            "output.run_log=true",
            "output.memory=false",
            "output.timer=true",
            "output.log_ranks=[0]",
        ]

    @classmethod
    def from_chitu_runtime(cls, args: Any, *, world_size: int) -> "StageServiceConfig":
        """Build the service view from overrides emitted by ``chitu run``."""
        diffusion = args.infer.diffusion
        widths = tuple(int(value) for value in diffusion.elastic_allowed_widths)
        if not widths:
            widths = tuple(width for width in range(1, world_size + 1) if world_size % width == 0)

        visible = [
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        ]
        if len(visible) == world_size and all(value.isdigit() for value in visible):
            gpu_ids = [int(value) for value in visible]
        else:
            # Under Slurm these are stage-local ids; placement remains owned by Slurm.
            gpu_ids = list(range(world_size))

        host = str(args.serve.host)
        advertise_host = os.environ.get("CHITU_API_ADVERTISE_HOST", "").strip()
        if not advertise_host and host in {"0.0.0.0", "::"}:
            advertise_host = socket.gethostname()
        mapping = {
            "name": os.environ.get("CHITU_API_STAGE_NAME", "image_decode"),
            "process": "chitu_image_decoder",
            "factory": "zimage",
            "gpu": gpu_ids,
            "terminal": True,
            "parallelism": {
                "sp": world_size,
                "chitu_pool": {
                    "policy": str(diffusion.scheduling_policy),
                    "allowed_lane_widths": list(widths),
                    "switch_allowed_until_step": int(
                        diffusion.switch_allowed_until_step
                    ),
                    "phase_max_steps": int(diffusion.epe.phase_max_steps),
                    "max_inflight_requests": int(args.infer.max_reqs),
                    "max_pending_requests": int(args.infer.max_reqs),
                    "cost_model": str(diffusion.cost_model),
                    "online_calibration": bool(diffusion.epe.online_calibration),
                },
            },
            "factory_args": {
                "model_path": str(args.models.ckpt_dir),
                "num_steps": int(args.models.sampler.sample_steps),
                "sample_solver": "flowmatch_euler",
            },
            "service": {
                "host": host,
                "port": int(args.serve.port),
                "advertise_host": advertise_host or None,
            },
            "output_root": str(args.output.root_dir),
        }
        return cls.from_mapping(mapping)


def load_stage_service_config(path: str | Path) -> StageServiceConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"stage config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return StageServiceConfig.from_mapping(_mapping(raw, "stage config"))
