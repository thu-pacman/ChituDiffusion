"""Diffusers model integrations provided by chitu_diffusion."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .flux1 import Flux1Pipeline, Flux1Request
    from .flux2_klein import Flux2KleinCpPipeline
    from .llada_image import LLaDAImagePipeline, LLaDAImageRequest
    from .minimax_h3 import MiniMaxH3DiTConfig, MiniMaxH3DiTModel
    from .qwen_image import QwenImagePipeline, QwenImageRequest
    from .wan import WanPipeline, WanRequest
    from .zimage import ZImagePipeline, ZImageRequest


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .. import __getattr__ as public_export

    value = public_export(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "Flux1Pipeline",
    "Flux1Request",
    "Flux2KleinCpPipeline",
    "LLaDAImagePipeline",
    "LLaDAImageRequest",
    "MiniMaxH3DiTConfig",
    "MiniMaxH3DiTModel",
    "QwenImagePipeline",
    "QwenImageRequest",
    "WanPipeline",
    "WanRequest",
    "ZImagePipeline",
    "ZImageRequest",
]
