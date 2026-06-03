from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FlexCacheParams:
    """Base FlexCache parameters shared by all strategies."""
    strategy: Optional[str] = None
    warmup: int = 5
    cooldown: int = 5


@dataclass
class TeaCacheParams(FlexCacheParams):
    strategy: Optional[str] = "teacache"
    teacache_thresh: float = 0.2
    coefficients: Optional[list[float]] = None
    use_ref_steps: bool = True


@dataclass
class PABParams(FlexCacheParams):
    strategy: Optional[str] = "pab"
    skip_self_range: int = 2
    skip_cross_range: int = 3


@dataclass
class BlockDanceParams(FlexCacheParams):
    strategy: Optional[str] = "blockdance"
    boundary_block: int = 20
    group_size: int = 2
    start_fraction: float = 0.40
    end_fraction: float = 0.95


@dataclass
class CubicParams(FlexCacheParams):
    strategy: Optional[str] = "cubic"
    cache_ratio: float = 0.5
    tau_max: int = 8
    target_speedup: Optional[float] = None
    patch_size: Optional[Tuple[int, int, int]] = None
    num_layers: Optional[int] = None
    warmup_fraction: Optional[float] = None
    cooldown_fraction: Optional[float] = None
    anchor_interval: Optional[int] = None
    partition_mode: str = "wan13_832x480_uniform"
    min_block_size: int = 2
    max_block_size: int = 16
    cv_threshold: float = 0.5
    alpha: float = 1.0
    beta: float = 2.0
    curvature_contrast_gamma: float = 1.0


@dataclass
class TaylorSeerParams(FlexCacheParams):
    strategy: Optional[str] = "taylorseer"
    fresh_threshold: int = 5
    max_order: int = 1
    first_enhance: int = 1


@dataclass
class DiTangoParams(FlexCacheParams):
    strategy: Optional[str] = "ditango"
    cache_ratio: float = 0.5
    tau_max: int = 8
    curvature_interval_power: float = 1.0 / 3.0


FLEXCACHE_PARAM_CLASSES = {
    "teacache": TeaCacheParams,
    "pab": PABParams,
    "blockdance": BlockDanceParams,
    "cubic": CubicParams,
    "taylorseer": TaylorSeerParams,
    "ditango": DiTangoParams,
}
