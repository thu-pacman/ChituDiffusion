from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from ..flexcache.config import CacheConfig

_LEGACY_CONFIG_FIELDS = {
    "infer",
    "diffusion",
    "step_interleave",
    "dynamic_sp",
    "hotswitch_enabled",
    "tp",
}


@dataclass(frozen=True, slots=True)
class EPEServeConfig:
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
    min_resize_gain_ms: float = 0.0
    resize_hysteresis_ms: float = 0.0
    resize_control_cost_ms: float = 0.0
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
        if min(
            self.min_resize_gain_ms,
            self.resize_hysteresis_ms,
            self.resize_control_cost_ms,
        ) < 0:
            raise ValueError("resize cost and hysteresis controls must be >= 0")
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
        if key in _LEGACY_CONFIG_FIELDS and path != "parallelism.tp":
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
    min_resize_gain_ms: float = 0.0
    resize_hysteresis_ms: float = 0.0
    resize_control_cost_ms: float = 0.0
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
        min_resize_gain_ms = float(raw.get("min_resize_gain_ms", 0.0))
        resize_hysteresis_ms = float(raw.get("resize_hysteresis_ms", 0.0))
        resize_control_cost_ms = float(raw.get("resize_control_cost_ms", 0.0))
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
        if min(
            min_resize_gain_ms,
            resize_hysteresis_ms,
            resize_control_cost_ms,
        ) < 0:
            raise ValueError("resize cost and hysteresis controls must be >= 0")
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
            min_resize_gain_ms=min_resize_gain_ms,
            resize_hysteresis_ms=resize_hysteresis_ms,
            resize_control_cost_ms=resize_control_cost_ms,
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
    tp: int = 1

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, gpu_count: int
    ) -> "ParallelismConfig":
        tp = int(raw.get("tp", 1))
        sp = int(raw.get("sp", gpu_count // max(tp, 1)))
        if tp < 1 or sp < 1 or sp * tp != gpu_count:
            raise ValueError(
                "parallelism.sp * parallelism.tp must equal the number of "
                f"stage GPUs ({gpu_count})"
            )
        pool_raw = _mapping(raw.get("chitu_pool", {}), "parallelism.chitu_pool")
        pool = HotSwitchPoolConfig.from_mapping(pool_raw, world_size=sp)
        if tp > 1 and pool.policy == "static_dp":
            raise ValueError("static_dp is not supported with tensor parallelism")
        return cls(
            sp=sp,
            chitu_pool=pool,
            tp=tp,
        )


@dataclass(frozen=True)
class MediaWarmupProfile:
    duration_s: float
    fps: int
    height: int
    width: int
    output_type: str = "mp4"
    prompt: str = "MiniMax-H3 startup warmup"
    first_frame_path: str | None = None
    last_frame_path: str | None = None
    profile_id: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MediaWarmupProfile":
        profile = cls(
            duration_s=float(raw.get("duration_s", 5.0)),
            fps=int(raw.get("fps", 24)),
            height=int(raw.get("height", 768)),
            width=int(raw.get("width", 1344)),
            output_type=str(raw.get("output_type", "mp4")),
            prompt=str(raw.get("prompt", "MiniMax-H3 startup warmup")),
            first_frame_path=(
                None
                if raw.get("first_frame_path") is None
                else str(raw["first_frame_path"])
            ),
            last_frame_path=(
                None
                if raw.get("last_frame_path") is None
                else str(raw["last_frame_path"])
            ),
            profile_id=(
                None if raw.get("profile_id") is None else str(raw["profile_id"])
            ),
        )
        if profile.duration_s <= 0 or profile.fps <= 0:
            raise ValueError("media warmup duration and fps must be positive")
        if (
            min(profile.height, profile.width) < 32
            or profile.height % 32
            or profile.width % 32
        ):
            raise ValueError("media warmup dimensions must be multiples of 32")
        if profile.output_type not in {"mp4", "latent"}:
            raise ValueError("media warmup output_type must be mp4 or latent")
        if not profile.prompt:
            raise ValueError("media warmup prompt must not be empty")
        return profile


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
    vae_parallel_degree: int | None = None
    vae_parallel_halo: int = 8
    attention_backend: str = "auto"
    flow_shift: float = 12.0
    audio_flow_shift: float = 3.0
    latent_t: int = 2
    audio_t: int = 3
    audio_channels: int = 2
    text_length: int = 32
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    processor_path: str | None = None
    video_vae_path: str | None = None
    audio_vae_path: str | None = None
    enable_native_media: bool = False
    qwen_num_layers: int = 50
    default_duration_s: float = 5.0
    default_fps: int = 24
    default_output_type: str = "latent"
    ffmpeg_path: str = "ffmpeg"
    warmup_media_profiles: tuple[MediaWarmupProfile, ...] = ()
    warmup_burnin_steps: int = 0
    cost_profile_path: str | None = None

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
        if attention_mode not in {"agkv", "ulysses"}:
            raise ValueError(
                "factory_args.attention_mode must be one of: agkv, ulysses"
            )
        if ulysses_degree is not None and ulysses_degree < 1:
            raise ValueError("factory_args.ulysses_degree must be positive")
        raw_vae_degree = raw.get("vae_parallel_degree")
        vae_parallel_degree = (
            None if raw_vae_degree is None else int(raw_vae_degree)
        )
        if vae_parallel_degree is not None and vae_parallel_degree < 1:
            raise ValueError("factory_args.vae_parallel_degree must be positive")
        vae_parallel_halo = int(raw.get("vae_parallel_halo", 8))
        num_frames = int(raw.get("num_frames", 17))
        attention_backend = str(raw.get("attention_backend", "auto"))
        flow_shift = float(raw.get("flow_shift", 12.0))
        audio_flow_shift = float(raw.get("audio_flow_shift", 3.0))
        latent_t = int(raw.get("latent_t", 2))
        audio_t = int(raw.get("audio_t", 3))
        audio_channels = int(raw.get("audio_channels", 2))
        text_length = int(raw.get("text_length", 32))
        default_duration_s = float(raw.get("default_duration_s", 5.0))
        default_fps = int(raw.get("default_fps", 24))
        default_output_type = str(raw.get("default_output_type", "latent"))
        ffmpeg_path = str(raw.get("ffmpeg_path", "ffmpeg")).strip()
        qwen_num_layers = int(raw.get("qwen_num_layers", 50))
        enable_native_media = bool(raw.get("enable_native_media", False))
        warmup_media_profiles = tuple(
            MediaWarmupProfile.from_mapping(
                _mapping(item, "factory_args.warmup_media_profiles[]")
            )
            for item in raw.get("warmup_media_profiles", ())
        )
        warmup_burnin_steps = int(raw.get("warmup_burnin_steps", 0))
        if warmup_burnin_steps < 0:
            raise ValueError("factory_args.warmup_burnin_steps must be non-negative")
        if vae_parallel_halo < 0:
            raise ValueError("factory_args.vae_parallel_halo must be non-negative")
        if num_frames < 1 or (num_frames - 1) % 4:
            raise ValueError("factory_args.num_frames must equal 4n+1")
        if attention_backend not in {"auto", "fa4", "flex", "sdpa"}:
            raise ValueError(
                "factory_args.attention_backend must be auto, fa4, flex, or sdpa"
            )
        if flow_shift <= 0 or audio_flow_shift <= 0:
            raise ValueError("factory_args flow shifts must be positive")
        if min(latent_t, audio_t, audio_channels, text_length) < 1:
            raise ValueError("MiniMax-H3 sequence dimensions must be positive")
        if default_duration_s <= 0 or default_fps <= 0:
            raise ValueError("default duration and fps must be positive")
        if default_output_type not in {"mp4", "latent"}:
            raise ValueError("default_output_type must be mp4 or latent")
        if not ffmpeg_path:
            raise ValueError("factory_args.ffmpeg_path must not be empty")
        if not 1 <= qwen_num_layers <= 64:
            raise ValueError("qwen_num_layers must be in [1, 64]")
        if enable_native_media:
            missing = [
                name
                for name in ("text_encoder_path", "video_vae_path", "audio_vae_path")
                if not str(raw.get(name, "")).strip()
            ]
            if missing:
                raise ValueError(
                    "native MiniMax-H3 component paths are missing: "
                    + ", ".join(missing)
                )
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
            vae_parallel_degree=vae_parallel_degree,
            vae_parallel_halo=vae_parallel_halo,
            attention_backend=attention_backend,
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
            latent_t=latent_t,
            audio_t=audio_t,
            audio_channels=audio_channels,
            text_length=text_length,
            text_encoder_path=(
                None
                if raw.get("text_encoder_path") is None
                else str(raw["text_encoder_path"])
            ),
            tokenizer_path=(
                None
                if raw.get("tokenizer_path") is None
                else str(raw["tokenizer_path"])
            ),
            processor_path=(
                None
                if raw.get("processor_path") is None
                else str(raw["processor_path"])
            ),
            video_vae_path=(
                None
                if raw.get("video_vae_path") is None
                else str(raw["video_vae_path"])
            ),
            audio_vae_path=(
                None
                if raw.get("audio_vae_path") is None
                else str(raw["audio_vae_path"])
            ),
            enable_native_media=enable_native_media,
            qwen_num_layers=qwen_num_layers,
            default_duration_s=default_duration_s,
            default_fps=default_fps,
            default_output_type=default_output_type,
            ffmpeg_path=ffmpeg_path,
            warmup_media_profiles=warmup_media_profiles,
            warmup_burnin_steps=warmup_burnin_steps,
            cost_profile_path=(
                None
                if raw.get("cost_profile_path") is None
                else str(raw["cost_profile_path"])
            ),
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
        if factory not in {
            "zimage",
            "flux1",
            "qwen-image",
            "wan",
            "minimax-h3",
        }:
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
