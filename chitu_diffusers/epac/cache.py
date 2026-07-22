from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CacheStrategyName = Literal["none", "meancache", "teacache"]


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Reserved public cache interface for the EPAC pipeline.

    MeanCache and TeaCache names are accepted by the schema so request and
    service contracts remain stable, but execution fails fast until their
    request-local implementations are connected.
    """

    strategy: CacheStrategyName = "none"
    warmup_steps: int | None = None
    cooldown_steps: int | None = None
    fresh_steps: int = 25
    use_jvp: bool = True
    teacache_threshold: float = 0.2
    teacache_coefficients: tuple[float, ...] = (1.0, 0.0)

    def __post_init__(self) -> None:
        if self.strategy not in {"none", "meancache", "teacache"}:
            raise ValueError("cache strategy must be one of: none, meancache, teacache")
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.cooldown_steps is not None and self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")
        if self.fresh_steps <= 0:
            raise ValueError("fresh_steps must be positive")
        if self.teacache_threshold <= 0:
            raise ValueError("teacache_threshold must be positive")
        if not self.teacache_coefficients:
            raise ValueError("teacache_coefficients must not be empty")

    def require_available(self) -> None:
        if self.strategy != "none":
            raise NotImplementedError(
                f"cache strategy {self.strategy!r} is reserved but not connected yet"
            )
