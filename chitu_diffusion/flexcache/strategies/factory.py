from __future__ import annotations

from chitu_diffusion.flexcache.config import (
    CacheConfig,
    MagCacheConfig,
    MeanCacheConfig,
    PABConfig,
    TaylorSeerConfig,
    TeaCacheConfig,
)

from .base import BaseCacheStrategy
from .magcache import MagCacheStrategy
from .meancache import MeanCacheStrategy
from .pab import PABStrategy
from .taylorseer import TaylorSeerStrategy
from .teacache import TeaCacheStrategy


def create_cache_strategy(config: CacheConfig) -> BaseCacheStrategy:
    config.require_generate_available()
    common = config.common
    if config.strategy == "none":
        return BaseCacheStrategy()
    if config.strategy == "magcache":
        assert isinstance(config.params, MagCacheConfig)
        return MagCacheStrategy(config.params)
    if config.strategy == "meancache":
        assert isinstance(config.params, MeanCacheConfig)
        return MeanCacheStrategy(config.params)
    if config.strategy == "teacache":
        assert isinstance(config.params, TeaCacheConfig)
        return TeaCacheStrategy(
            config.params,
            warmup_steps=common.warmup_steps,
            cooldown_steps=common.cooldown_steps,
        )
    if config.strategy == "taylorseer":
        assert isinstance(config.params, TaylorSeerConfig)
        return TaylorSeerStrategy(
            config.params,
            warmup_steps=common.warmup_steps,
            cooldown_steps=common.cooldown_steps,
        )
    assert isinstance(config.params, PABConfig)
    return PABStrategy(
        config.params,
        warmup_steps=common.warmup_steps,
        cooldown_steps=common.cooldown_steps,
    )
