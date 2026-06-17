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
    warmup: int = 1
    cooldown: int = 1
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
    target_speedup: float = 2.0
    tau_max: int = 8
    block_size: Optional[int] = None
    uniform_square_min_splits: Optional[int] = None


@dataclass
class MeanCacheParams(FlexCacheParams):
    strategy: Optional[str] = "meancache"
    fresh_steps: int = 25
    use_jvp: bool = True


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
    anchor_interval: Optional[int] = None
    tau_max: int = 8
    curvature_interval_power: float = 1.0 / 3.0
    intra_group_size_limit: Optional[int] = 1
    locality_group_compute_boost: float = 0.0
    groupwise_stagger_period: int = 0
    groupwise_stagger_fresh_count: int = 0
    groupwise_stagger_layer_start: int = 0
    groupwise_stagger_layer_end: int = -1
    groupwise_keep_local: bool = True
    groupwise_force_tail_full_layers: int = 1
    groupwise_reuse_stale_kv: bool = False
    groupwise_local_expand: int = -1
    groupwise_fixed_anchor_steps: str = ""
    groupwise_topk_mode: str = "none"
    groupwise_extra_topk: int = 0
    groupwise_state_align: bool = False
    groupwise_state_align_mode: str = "delta"
    groupwise_state_align_out_scale: float = 1.0
    groupwise_state_align_lse_scale: float = 1.0
    groupwise_state_align_distance_tau: float = 2.0


FLEXCACHE_PARAM_CLASSES = {
    "teacache": TeaCacheParams,
    "pab": PABParams,
    "blockdance": BlockDanceParams,
    "cubic": CubicParams,
    "meancache": MeanCacheParams,
    "taylorseer": TaylorSeerParams,
    "ditango": DiTangoParams,
}
