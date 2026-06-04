from chitu_diffusion.runtime.adapter.base import (
    DiffusionModelSpec,
    DiffusionRuntimeAdapter,
    get_model_runtime_spec,
    register_model_runtime,
)

# Import concrete adapters for registration side effects.
from chitu_diffusion.runtime.adapter import flux1, flux2k, wan  # noqa: F401

__all__ = [
    "DiffusionModelSpec",
    "DiffusionRuntimeAdapter",
    "get_model_runtime_spec",
    "register_model_runtime",
]
