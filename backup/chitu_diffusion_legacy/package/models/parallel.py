from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParallelCapabilities:
    """Parallel features a model/runtime adapter can execute correctly.

    Model-specific sequence semantics belong to model code or adapter hooks. The
    generic parallel layer only consumes this contract to compose feature plans.
    """

    dit_context_parallel: bool = True
    dit_cfg_parallel: bool = True
    vae_tile_parallel: bool = True
    sampler_local_latent: bool = False
    model_specific_context_parallel: bool = False
