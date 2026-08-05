from __future__ import annotations

from typing import Any

from ..contracts import CacheStats, CacheStepContext
from ..spec import BlockSite, FlexCacheModelSpec, LeafSite
from ..tree import TensorTree


class BaseCacheStrategy:
    def __init__(self) -> None:
        self.stats = CacheStats()
        self.model_spec: FlexCacheModelSpec | None = None
        self.total_steps = 0

    def begin(self, *, total_steps: int, model_spec: FlexCacheModelSpec) -> None:
        self.stats = CacheStats()
        self.model_spec = model_spec
        self.total_steps = total_steps

    def model_lookup(
        self,
        context: CacheStepContext,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]:
        self.stats.step_misses += 1
        return False, None

    def model_store(
        self,
        context: CacheStepContext,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: TensorTree,
    ) -> None:
        return None

    def block_lookup(
        self,
        context: CacheStepContext,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]:
        return False, None

    def block_store(
        self,
        context: CacheStepContext,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: TensorTree,
    ) -> None:
        return None

    def leaf_lookup(
        self,
        context: CacheStepContext,
        site: LeafSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]:
        return False, None

    def leaf_store(
        self,
        context: CacheStepContext,
        site: LeafSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: TensorTree,
    ) -> None:
        return None

    def end(self) -> CacheStats:
        return self.stats
