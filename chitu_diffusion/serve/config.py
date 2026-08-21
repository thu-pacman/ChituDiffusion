from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from ..epac.cache import CacheConfig

_LEGACY_CONFIG_FIELDS = {
    "infer",
    "diffusion",
    "step_interleave",
    "dynamic_sp",
    "hotswitch_enabled",
    "tp",
}


@dataclass(frozen=True, slots=True)
class EPACServeConfig:
    host: str = "0.0.0.0"
    port: int = 18200
    advertise_host: str | None = None
    warmup_resolutions: tuple[tuple[int, int], ...] = (
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    )
    warmup_steps: int = 5
    pulse_steps: int = 5
    switch_allowed_until_step: int = 50
    balanced_k: bool = True
    starvation_ms: float = 30_000.0
    default_deadline_ms: float | None = None
    deadline_guard_ms: float = 3_000.0
    max_inflight_requests: int = 4
    max_pending_requests: int = 64
    default_num_steps: int = 50
    default_width: int = 1024
    default_height: int = 1024
    online_calibration: bool = True
    output_root: str = "outputs/chitu-api"
    record_timeline: bool = False
    postprocess_workers: int = 4
    cfg_parallel: bool = True
    parallel_vae: bool = True
    vae_parallel_halo: int = 8
    cache: CacheConfig = CacheConfig()
    schedule_strategy: Literal["elastic", "static_cp", "static_dp"] = "elastic"

    def __post_init__(self) -> None:
        self.cache.require_serve_available()
        if not 1024 <= self.port <= 65535:
            raise ValueError("port must be between 1024 and 65535")
        if (
            min(
                self.warmup_steps,
                self.pulse_steps,
                self.max_inflight_requests,
                self.max_pending_requests,
                self.default_num_steps,
                self.postprocess_workers,
            )
            <= 0
        ):
            raise ValueError("step counts and request limits must be positive")
        if self.warmup_steps < 3:
            raise ValueError("warmup_steps must be >= 3")
        if not self.warmup_resolutions:
            raise ValueError("warmup_resolutions must not be empty")
        if self.deadline_guard_ms < 0:
            raise ValueError("deadline_guard_ms must be >= 0")
        if self.default_deadline_ms is not None and self.default_deadline_ms <= 0:
            raise ValueError("default_deadline_ms must be positive")
        if self.vae_parallel_halo < 0:
            raise ValueError("vae_parallel_halo must be non-negative")
        if self.schedule_strategy not in {"elastic", "static_cp", "static_dp"}:
            raise ValueError(
                "schedule_strategy must be one of: elastic, static_cp, static_dp"
            )
        for height, width in self.warmup_resolutions:
            if min(height, width) < 16 or height % 16 or width % 16:
                raise ValueError(
                    "warmup resolutions must contain positive multiples of 16"
                )


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _reject_legacy_fields(raw: Mapping[str, Any], *, prefix: str = "") -> None:
    for key, value in raw.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key in _LEGACY_CONFIG_FIELDS:
            raise ValueError(
                f"legacy chitu_diffusion config field is not supported: {path}"
            )
        if isinstance(value, Mapping):
            _reject_legacy_fields(value, prefix=path)


def _resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        height = width = value
    elif isinstance(value, str):
        normalized = value.lower().replace(" ", "")
        if "x" not in normalized:
            height = width = int(normalized)
        else:
            left, right = normalized.split("x", 1)
            height, width = int(left), int(right)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
    else:
        raise ValueError(f"invalid resolution value: {value!r}")
    if height < 16 or width < 16 or height % 16 or width % 16:
        raise ValueError("warmup resolutions must be positive multiples of 16")
    return height, width


@dataclass(frozen=True)
class HotSwitchPoolConfig:
    policy: str = "elastic"
    allowed_lane_widths: tuple[int, ...] = (1,)
    switch_allowed_until_step: int = 0
    pulse_steps: int = 1
    balanced_k: bool = True
    starvation_ms: float = 30_000.0
    default_deadline_ms: float | None = None
    deadline_guard_ms: float = 3_000.0
    max_inflight_requests: int = 64
    max_pending_requests: int = 256
    warmup_resolutions: tuple[tuple[int, int], ...] = (
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    )
    warmup_steps: int = 5
    online_calibration: bool = True

    def __post_init__(self) -> None:
        if self.warmup_steps < 3:
            raise ValueError("warmup_steps must be >= 3")

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
        if widths[-1] != world_size:
            raise ValueError(
                "allowed_lane_widths must include the full stage world size"
            )

        policy = str(raw.get("policy", "elastic"))
        if policy not in {"elastic", "static_cp", "static_dp"}:
            raise ValueError(
                "chitu_pool.policy must be one of: elastic, static_cp, static_dp"
            )

        pulse_steps = int(raw.get("pulse_steps", 1))
        max_inflight = int(raw.get("max_inflight_requests", 64))
        max_pending = int(raw.get("max_pending_requests", 256))
        switch_until = int(raw.get("switch_allowed_until_step", 0))
        starvation_ms = float(raw.get("starvation_ms", 30_000.0))
        raw_default_deadline = raw.get("default_deadline_ms")
        default_deadline_ms = (
            None if raw_default_deadline is None else float(raw_default_deadline)
        )
        deadline_guard_ms = float(raw.get("deadline_guard_ms", 3_000.0))
        warmup_steps = int(raw.get("warmup_steps", 5))
        warmup_resolutions = tuple(
            _resolution(value)
            for value in raw.get(
                "warmup_resolutions",
                ((512, 512), (1024, 1024), (2048, 2048)),
            )
        )
        if pulse_steps < 1:
            raise ValueError("pulse_steps must be >= 1")
        if max_inflight < 1 or max_pending < 1:
            raise ValueError("request limits must be >= 1")
        if switch_until < 0:
            raise ValueError("switch_allowed_until_step must be >= 0")
        if starvation_ms < 0:
            raise ValueError("starvation_ms must be >= 0")
        if default_deadline_ms is not None and default_deadline_ms <= 0:
            raise ValueError("default_deadline_ms must be positive")
        if deadline_guard_ms < 0:
            raise ValueError("deadline_guard_ms must be >= 0")
        if warmup_steps < 3:
            raise ValueError("warmup_steps must be >= 3")
        if not warmup_resolutions:
            raise ValueError("warmup_resolutions must not be empty")

        return cls(
            policy=policy,
            allowed_lane_widths=widths,
            switch_allowed_until_step=switch_until,
            pulse_steps=pulse_steps,
            balanced_k=bool(raw.get("balanced_k", True)),
            starvation_ms=starvation_ms,
            default_deadline_ms=default_deadline_ms,
            deadline_guard_ms=deadline_guard_ms,
            max_inflight_requests=max_inflight,
            max_pending_requests=max_pending,
            warmup_resolutions=warmup_resolutions,
            warmup_steps=warmup_steps,
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
        sp = int(raw.get("sp", gpu_count))
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
class DiffusionFactoryConfig:
    model_path: str
    num_steps: int = 20
    sample_solver: str = "flowmatch_euler"
    default_width: int = 1024
    default_height: int = 1024
    num_frames: int = 17
    attention_mode: str = "agkv"
    ulysses_degree: int | None = None
    cfg_parallel: bool = True
    parallel_vae: bool = True
    vae_parallel_halo: int = 8

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DiffusionFactoryConfig":
        model_path = str(raw.get("model_path", "")).strip()
        if not model_path:
            raise ValueError("factory_args.model_path is required")
        num_steps = int(raw.get("num_steps", 20))
        width = int(raw.get("default_width", 1024))
        height = int(raw.get("default_height", 1024))
        attention_mode = str(raw.get("attention_mode", "agkv"))
        raw_degree = raw.get("ulysses_degree")
        ulysses_degree = None if raw_degree is None else int(raw_degree)
        if num_steps < 1:
            raise ValueError("factory_args.num_steps must be >= 1")
        if width < 16 or height < 16 or width % 16 or height % 16:
            raise ValueError(
                "default image dimensions must be positive multiples of 16"
            )
        if attention_mode not in {"agkv", "usp"}:
            raise ValueError("factory_args.attention_mode must be one of: agkv, usp")
        if ulysses_degree is not None and ulysses_degree < 1:
            raise ValueError("factory_args.ulysses_degree must be positive")
        vae_parallel_halo = int(raw.get("vae_parallel_halo", 8))
        num_frames = int(raw.get("num_frames", 17))
        if vae_parallel_halo < 0:
            raise ValueError("factory_args.vae_parallel_halo must be non-negative")
        if num_frames < 1 or (num_frames - 1) % 4:
            raise ValueError("factory_args.num_frames must equal 4n+1")
        return cls(
            model_path=model_path,
            num_steps=num_steps,
            sample_solver=str(raw.get("sample_solver", "flowmatch_euler")),
            default_width=width,
            default_height=height,
            num_frames=num_frames,
            attention_mode=attention_mode,
            ulysses_degree=ulysses_degree,
            cfg_parallel=bool(raw.get("cfg_parallel", True)),
            parallel_vae=bool(raw.get("parallel_vae", True)),
            vae_parallel_halo=vae_parallel_halo,
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
    factory_args: DiffusionFactoryConfig
    service: ServiceConfig
    terminal: bool = True
    output_root: str = "outputs/chitu-api"
    record_timeline: bool = False
    postprocess_workers: int = 4

    def __post_init__(self) -> None:
        if self.postprocess_workers < 1:
            raise ValueError("postprocess_workers must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StageServiceConfig":
        _reject_legacy_fields(raw)
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
        if factory not in {"zimage", "flux1", "qwen-image", "wan"}:
            raise ValueError(f"unsupported factory: {factory}")

        return cls(
            name=name,
            process=process,
            factory=factory,
            gpu=gpu,
            parallelism=ParallelismConfig.from_mapping(
                _mapping(raw.get("parallelism", {}), "parallelism"),
                gpu_count=len(gpu),
            ),
            factory_args=DiffusionFactoryConfig.from_mapping(
                _mapping(raw.get("factory_args", {}), "factory_args")
            ),
            service=ServiceConfig.from_mapping(
                _mapping(raw.get("service", {}), "service")
            ),
            terminal=bool(raw.get("terminal", True)),
            output_root=str(raw.get("output_root", "outputs/chitu-api")),
            record_timeline=bool(raw.get("record_timeline", False)),
            postprocess_workers=int(raw.get("postprocess_workers", 4)),
        )

def load_stage_service_config(path: str | Path) -> StageServiceConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"stage config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return StageServiceConfig.from_mapping(_mapping(raw, "stage config"))
