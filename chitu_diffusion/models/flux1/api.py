from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ...epe.api import (
    DiffusersEPEPipeline,
    build_diffusion_request,
    validate_image_request,
)
from ...epe.request import DiffusionRequest
from ...flexcache.config import CacheConfig
from .pipeline import EpeFlux1Pipeline


@dataclass(frozen=True, slots=True)
class Flux1Request:
    prompt: str | None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    prompt_2: str | None = None
    width: int = 1024
    height: int = 1024
    num_steps: int = 50
    guidance_scale: float = 3.5
    seed: int = 0
    deadline_ms: float | None = None
    priority: int = 0
    output_type: str = "pil"
    cache: CacheConfig = field(default_factory=CacheConfig)
    extra_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_image_request(self)

    def to_diffusion_request(self, *, device: torch.device) -> DiffusionRequest:
        self.cache.require_generate_available()
        self.cache.validate_steps(self.num_steps)
        return build_diffusion_request(
            self,
            device=device,
            model_inputs={"prompt": self.prompt, "prompt_2": self.prompt_2},
            reserved_fields={
                "prompt",
                "prompt_2",
                "width",
                "height",
                "guidance_scale",
                "generator",
                "num_inference_steps",
                "output_type",
            },
            metadata={"cache": self.cache.to_dict()},
        )


class Flux1Pipeline(DiffusersEPEPipeline):
    """Diffusers-style facade for static FLUX.1-dev generation."""

    pipeline_class = EpeFlux1Pipeline
    generation_error_prefix = "Flux.1 EPE"

    def _create_backend(self, config: Any | None = None) -> Any:
        from ...epe.scheduling.planner import EpeSchedulingModule
        from .executor import Flux1ImageDecoderExecutor

        parallel_vae = getattr(config, "parallel_vae", None)

        return Flux1ImageDecoderExecutor(
            self._pipeline,
            EpeSchedulingModule(
                self.parallel_context,
                policy=str(getattr(config, "schedule_strategy", "static_cp")),
            ),
            default_width=int(getattr(config, "default_width", 1024)),
            default_height=int(getattr(config, "default_height", 1024)),
            default_num_steps=int(getattr(config, "default_num_steps", 50)),
            parallel_vae=True if parallel_vae is None else bool(parallel_vae),
            vae_parallel_halo=int(getattr(config, "vae_parallel_halo", 8)),
        )
