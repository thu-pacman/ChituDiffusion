from __future__ import annotations

from typing import Any

from chitu_diffusion.epac.cache import PABConfig

from ..contracts import CacheStepContext
from ..spec import FlexCacheModelSpec, LeafSite
from ..tree import TensorTree, tree_clone, tree_nbytes
from .base import BaseCacheStrategy


class PABStrategy(BaseCacheStrategy):
    def __init__(
        self,
        params: PABConfig,
        *,
        warmup_steps: int,
        cooldown_steps: int,
    ) -> None:
        super().__init__()
        self.params = params
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.cache: dict[str, TensorTree] = {}

    def begin(self, *, total_steps: int, model_spec: FlexCacheModelSpec) -> None:
        super().begin(total_steps=total_steps, model_spec=model_spec)
        if not model_spec.attentions:
            raise ValueError(
                f"{model_spec.family} exposes no attention sites required by PAB"
            )
        self.cache.clear()
        self.stats.strategy = {
            "self_interval": self.params.self_interval,
            "cross_interval": self.params.cross_interval,
            "joint_interval": self.params.joint_interval,
        }

    def _interval(self, site: LeafSite) -> int:
        return {
            "self": self.params.self_interval,
            "cross": self.params.cross_interval,
            "joint": self.params.joint_interval,
        }[site.kind]

    def _forced(self, context: CacheStepContext) -> bool:
        return (
            context.step_index < self.warmup_steps
            or context.step_index >= self.total_steps - self.cooldown_steps
        )

    def leaf_lookup(
        self,
        context: CacheStepContext,
        site: LeafSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]:
        del args, kwargs
        if not site.is_attention:
            return False, None
        cached = self.cache.get(site.site_id)
        if (
            cached is not None
            and not self._forced(context)
            and context.step_index % self._interval(site)
        ):
            self.stats.leaf_hits += 1
            return True, tree_clone(cached)
        self.stats.leaf_misses += 1
        return False, None

    def leaf_store(
        self,
        context: CacheStepContext,
        site: LeafSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: TensorTree,
    ) -> None:
        del context, args, kwargs
        if not site.is_attention:
            return None
        self.cache[site.site_id] = tree_clone(output)
        self.stats.cache_bytes = sum(
            tree_nbytes(value) for value in self.cache.values()
        )
        return None
