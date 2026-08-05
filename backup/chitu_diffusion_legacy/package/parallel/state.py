from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from chitu_diffusion.models.parallel import ModelParallelCapabilities


@dataclass
class DitParallelPlan:
    """Resolved DiT parallel execution plan for one backend instance."""

    cfg_size: int = 1
    cp_size: int = 1
    up: int = 1
    cp_backend: str = "none"

    @property
    def enabled(self) -> bool:
        return self.cp_size > 1 or self.cfg_size > 1

    @property
    def uses_agcp(self) -> bool:
        return self.cp_backend == "agcp"

    @property
    def uses_ucp(self) -> bool:
        return self.cp_backend == "ucp"


@dataclass
class VaeParallelPlan:
    """Resolved VAE parallel plan."""

    mode: str = "tile"
    enabled: bool = True


@dataclass
class SamplerParallelPlan:
    """Resolved sampler/latent update parallel plan."""

    mode: str = "local_latent"
    enabled: bool = False


@dataclass
class ParallelPlan:
    """Top-level plan that composes Chitu's parallel runtime features."""

    dit: DitParallelPlan = field(default_factory=DitParallelPlan)
    vae: VaeParallelPlan = field(default_factory=VaeParallelPlan)
    sampler: SamplerParallelPlan = field(default_factory=SamplerParallelPlan)


@dataclass
class ContextParallelLatentState:
    """Task-local metadata for latents sharded along the CP sequence axis."""

    image_offset: int
    image_seq_len: int
    is_local: bool = True

    def attention_kwargs(self) -> dict[str, int]:
        return {
            "image_offset": int(self.image_offset),
            "image_seq_len": int(self.image_seq_len),
        }


@dataclass
class ParallelTaskState:
    """Task-local state owned by the parallel feature layer."""

    cp_latents: Optional[ContextParallelLatentState] = None
