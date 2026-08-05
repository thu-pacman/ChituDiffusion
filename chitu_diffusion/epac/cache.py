from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any, Literal, Mapping

CacheStrategyName = Literal[
    "none",
    "magcache",
    "meancache",
    "teacache",
    "taylorseer",
    "pab",
]


def _finite_float_tuple(name: str, values: Any) -> tuple[float, ...]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of finite numbers") from exc
    if not items:
        raise ValueError(f"{name} must not be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in items
    ):
        raise ValueError(f"{name} must contain only finite numbers")
    return tuple(float(value) for value in items)


@dataclass(frozen=True, slots=True)
class CacheCommonConfig:
    warmup_steps: int = 0
    cooldown_steps: int = 0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0 or self.cooldown_steps < 0:
            raise ValueError("cache warmup_steps and cooldown_steps must be non-negative")


@dataclass(frozen=True, slots=True)
class MeanCacheConfig:
    fresh_steps: int = 25
    use_jvp: bool = True

    def __post_init__(self) -> None:
        if self.fresh_steps < 1:
            raise ValueError("MeanCache fresh_steps must be positive")


@dataclass(frozen=True, slots=True)
class MagCacheConfig:
    """Magnitude-ratio residual reuse.

    ``None`` values select the official profile defaults for the detected
    checkpoint. Explicit ratios are intended for separately calibrated
    checkpoints and require explicit policy parameters.
    """

    threshold: float | None = None
    max_skip_steps: int | None = None
    retention_ratio: float | None = None
    ratios: tuple[float, ...] | None = None
    reference_steps: int | None = None

    def __post_init__(self) -> None:
        if self.threshold is not None and self.threshold <= 0:
            raise ValueError("MagCache threshold must be positive")
        if self.max_skip_steps is not None and self.max_skip_steps < 1:
            raise ValueError("MagCache max_skip_steps must be positive")
        if self.retention_ratio is not None and not 0 <= self.retention_ratio < 1:
            raise ValueError("MagCache retention_ratio must be in [0, 1)")
        if self.ratios is not None:
            object.__setattr__(
                self,
                "ratios",
                _finite_float_tuple("MagCache ratios", self.ratios),
            )
            if self.reference_steps is None or self.reference_steps < 1:
                raise ValueError(
                    "custom MagCache ratios require a positive reference_steps"
                )
            if (
                self.threshold is None
                or self.max_skip_steps is None
                or self.retention_ratio is None
            ):
                raise ValueError(
                    "custom MagCache ratios require threshold, max_skip_steps, "
                    "and retention_ratio"
                )
        elif self.reference_steps is not None:
            raise ValueError("MagCache reference_steps requires custom ratios")


@dataclass(frozen=True, slots=True)
class TeaCacheConfig:
    """Reuse threshold for the rescaled modulation drift.

    ``coefficients`` defaults to the polynomial published for the detected
    checkpoint. Supplying it explicitly opts out of that calibration, in which
    case ``threshold`` has to be re-tuned against whatever rescale is used.
    """

    threshold: float = 0.2
    coefficients: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("TeaCache threshold must be positive")
        if self.coefficients is not None:
            object.__setattr__(
                self,
                "coefficients",
                _finite_float_tuple("TeaCache coefficients", self.coefficients),
            )


@dataclass(frozen=True, slots=True)
class TaylorSeerConfig:
    """Extrapolation window for pre-gate submodule outputs.

    ``fresh_threshold`` recomputes every Nth step. The reference ships 5 and 6,
    but those aggressive windows can introduce severe artifacts. The default
    is the measured 50-step Wan operating point near 2x DiT speedup; callers
    should revalidate quality when changing model family or step count.
    """

    fresh_threshold: int = 3
    max_order: int = 1
    first_enhance: int = 3

    def __post_init__(self) -> None:
        if self.fresh_threshold < 1:
            raise ValueError("TaylorSeer fresh_threshold must be positive")
        if self.max_order < 0 or self.first_enhance < 0:
            raise ValueError("TaylorSeer max_order and first_enhance must be non-negative")


@dataclass(frozen=True, slots=True)
class PABConfig:
    self_interval: int = 2
    cross_interval: int = 3
    joint_interval: int = 2

    def __post_init__(self) -> None:
        if min(self.self_interval, self.cross_interval, self.joint_interval) < 1:
            raise ValueError("PAB attention intervals must be positive")


CacheParams = (
    MagCacheConfig
    | MeanCacheConfig
    | TeaCacheConfig
    | TaylorSeerConfig
    | PABConfig
    | None
)
_PARAM_TYPES = {
    "magcache": MagCacheConfig,
    "meancache": MeanCacheConfig,
    "teacache": TeaCacheConfig,
    "taylorseer": TaylorSeerConfig,
    "pab": PABConfig,
}


@dataclass(frozen=True, slots=True)
class CacheConfig:
    strategy: CacheStrategyName = "none"
    common: CacheCommonConfig = field(default_factory=CacheCommonConfig)
    params: CacheParams = None

    def __post_init__(self) -> None:
        if self.strategy not in {
            "none",
            "magcache",
            "meancache",
            "teacache",
            "taylorseer",
            "pab",
        }:
            raise ValueError(
                "cache strategy must be one of: none, magcache, meancache, teacache, "
                "taylorseer, pab"
            )
        expected = _PARAM_TYPES.get(self.strategy)
        if expected is None:
            if self.params is not None:
                raise ValueError("cache strategy 'none' does not accept params")
            return
        if self.params is None:
            object.__setattr__(self, "params", expected())
        elif not isinstance(self.params, expected):
            raise ValueError(
                f"cache strategy {self.strategy!r} requires {expected.__name__}"
            )
        if self.strategy in {"magcache", "meancache"} and (
            self.common.warmup_steps or self.common.cooldown_steps
        ):
            raise ValueError(
                f"{self.strategy} uses its own retention/fresh schedule and does not "
                "accept common warmup/cooldown overrides"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "CacheConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("cache configuration must be a mapping")
        if not raw:
            return cls()
        allowed = {"strategy", "common", "params"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown cache fields: {sorted(unknown)}")
        strategy = str(raw.get("strategy", "none"))
        common_raw = raw.get("common", {})
        params_raw = raw.get("params", {})
        if not isinstance(common_raw, Mapping) or not isinstance(params_raw, Mapping):
            raise ValueError("cache common and params must be mappings")
        common_allowed = {"warmup_steps", "cooldown_steps"}
        unknown_common = set(common_raw) - common_allowed
        if unknown_common:
            raise ValueError(f"unknown cache common fields: {sorted(unknown_common)}")
        common = CacheCommonConfig(**common_raw)
        param_type = _PARAM_TYPES.get(strategy)
        if param_type is None:
            if params_raw:
                raise ValueError(f"cache strategy {strategy!r} does not accept params")
            params = None
        else:
            try:
                params = param_type(**params_raw)
            except TypeError as exc:
                raise ValueError(
                    f"invalid {strategy} cache params: {exc}"
                ) from exc
        return cls(strategy=strategy, common=common, params=params)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "common": asdict(self.common),
            "params": {} if self.params is None else asdict(self.params),
        }

    def validate_steps(self, total_steps: int) -> None:
        if total_steps <= 0:
            raise ValueError("cache total_steps must be positive")
        if self.strategy == "meancache" and total_steps != 50:
            raise ValueError("official MeanCache schedules require exactly 50 steps")
        if self.common.warmup_steps + self.common.cooldown_steps >= total_steps:
            raise ValueError("cache warmup_steps + cooldown_steps must be < total_steps")

    def require_generate_available(self) -> None:
        return None

    def require_serve_available(self) -> None:
        if self.strategy != "none":
            raise NotImplementedError(
                f"cache strategy {self.strategy!r} is generate-only; "
                "serve/EPE is not supported"
            )
