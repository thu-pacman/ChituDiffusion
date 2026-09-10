"""Public APIs for ChituDiffusion."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .epe import EmbeddedRuntimeConfig, StageWorldSpec
    from .flexcache import (
        CacheCommonConfig,
        CacheConfig,
        FreeCacheConfig,
        FreeCacheProfile,
        MagCacheConfig,
        MeanCacheConfig,
        PABConfig,
        TaylorSeerConfig,
        TeaCacheConfig,
    )
    from .models.flux1 import Flux1Pipeline, Flux1Request
    from .models.flux2_klein import Flux2KleinCpPipeline
    from .models.llada_image import LLaDAImagePipeline, LLaDAImageRequest
    from .models.minimax_h3 import MiniMaxH3DiTConfig, MiniMaxH3DiTModel
    from .models.qwen_image import QwenImagePipeline, QwenImageRequest
    from .models.wan import WanPipeline, WanRequest
    from .models.zimage import ZImagePipeline, ZImageRequest
    from .serve import EmbeddedDiffusionRuntime, EPEServeConfig, HotSwitchPoolConfig

_EXPORT_MODULES = {
    "EmbeddedRuntimeConfig": ".epe",
    "StageWorldSpec": ".epe",
    "EmbeddedDiffusionRuntime": ".serve",
    "EPEServeConfig": ".serve",
    "HotSwitchPoolConfig": ".serve",
    "CacheCommonConfig": ".flexcache",
    "CacheConfig": ".flexcache",
    "FreeCacheConfig": ".flexcache",
    "FreeCacheProfile": ".flexcache",
    "MagCacheConfig": ".flexcache",
    "MeanCacheConfig": ".flexcache",
    "PABConfig": ".flexcache",
    "TaylorSeerConfig": ".flexcache",
    "TeaCacheConfig": ".flexcache",
    "Flux1Pipeline": ".models.flux1",
    "Flux1Request": ".models.flux1",
    "Flux2KleinCpPipeline": ".models.flux2_klein",
    "LLaDAImagePipeline": ".models.llada_image",
    "LLaDAImageRequest": ".models.llada_image",
    "MiniMaxH3DiTConfig": ".models.minimax_h3",
    "MiniMaxH3DiTModel": ".models.minimax_h3",
    "QwenImagePipeline": ".models.qwen_image",
    "QwenImageRequest": ".models.qwen_image",
    "WanPipeline": ".models.wan",
    "WanRequest": ".models.wan",
    "ZImagePipeline": ".models.zimage",
    "ZImageRequest": ".models.zimage",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "CacheCommonConfig",
    "CacheConfig",
    "FreeCacheConfig",
    "FreeCacheProfile",
    "EmbeddedDiffusionRuntime",
    "EmbeddedRuntimeConfig",
    "EPEServeConfig",
    "Flux1Pipeline",
    "Flux1Request",
    "Flux2KleinCpPipeline",
    "LLaDAImagePipeline",
    "LLaDAImageRequest",
    "MiniMaxH3DiTConfig",
    "MiniMaxH3DiTModel",
    "HotSwitchPoolConfig",
    "MagCacheConfig",
    "MeanCacheConfig",
    "PABConfig",
    "QwenImagePipeline",
    "QwenImageRequest",
    "StageWorldSpec",
    "TaylorSeerConfig",
    "TeaCacheConfig",
    "WanPipeline",
    "WanRequest",
    "ZImagePipeline",
    "ZImageRequest",
]
