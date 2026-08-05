from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .spec import BlockSite, FlexCacheModelSpec, LeafSite
from .tree import TensorTree


@dataclass(frozen=True, slots=True)
class CacheStepContext:
    step_index: int
    total_steps: int
    branch_index: int
    model_spec: FlexCacheModelSpec


def _rate(hits: int, misses: int) -> float:
    total = hits + misses
    return hits / total if total else 0.0


@dataclass(slots=True)
class CacheStats:
    """Per-site-class counters.

    Hits are tracked separately per site class because the strategies reuse at
    different granularities: reporting a single blended rate would make a
    block-level strategy look idle and an attention-level one look saturated.
    """

    step_hits: int = 0
    step_misses: int = 0
    block_hits: int = 0
    block_misses: int = 0
    leaf_hits: int = 0
    leaf_misses: int = 0
    cache_bytes: int = 0
    branches: int = 1
    strategy: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "CacheStats") -> None:
        self.step_hits += other.step_hits
        self.step_misses += other.step_misses
        self.block_hits += other.block_hits
        self.block_misses += other.block_misses
        self.leaf_hits += other.leaf_hits
        self.leaf_misses += other.leaf_misses
        self.cache_bytes += other.cache_bytes
        if not self.strategy:
            self.strategy = dict(other.strategy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_hits": self.step_hits,
            "step_misses": self.step_misses,
            "step_hit_rate": _rate(self.step_hits, self.step_misses),
            "block_hits": self.block_hits,
            "block_misses": self.block_misses,
            "block_hit_rate": _rate(self.block_hits, self.block_misses),
            "leaf_hits": self.leaf_hits,
            "leaf_misses": self.leaf_misses,
            "leaf_hit_rate": _rate(self.leaf_hits, self.leaf_misses),
            "cache_bytes": self.cache_bytes,
            "branches": self.branches,
            "strategy": dict(self.strategy),
        }


class CacheStrategy(Protocol):
    def begin(self, *, total_steps: int, model_spec: FlexCacheModelSpec) -> None: ...

    def model_lookup(
        self,
        context: CacheStepContext,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]: ...

    def model_store(
        self,
        context: CacheStepContext,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: TensorTree,
    ) -> None: ...

    def block_lookup(
        self,
        context: CacheStepContext,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]: ...

    def block_store(
        self,
        context: CacheStepContext,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: TensorTree,
    ) -> None: ...

    def leaf_lookup(
        self,
        context: CacheStepContext,
        site: LeafSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]: ...

    def leaf_store(
        self,
        context: CacheStepContext,
        site: LeafSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: TensorTree,
    ) -> None: ...

    def end(self) -> CacheStats: ...
