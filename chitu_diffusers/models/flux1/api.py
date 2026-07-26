from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from ...epac.api import (
    DiffusersEPACPipeline,
    build_diffusion_request,
    validate_image_request,
)
from ...epac.request import DiffusionRequest
from .adapter import Flux1ModelAdapter
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
    extra_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_image_request(self)

    def to_diffusion_request(self, *, device: torch.device) -> DiffusionRequest:
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
        )


class Flux1EPACPipeline(DiffusersEPACPipeline):
    """Diffusers-style facade for static FLUX.1-dev generation."""

    pipeline_class = EpeFlux1Pipeline
    adapter_class = Flux1ModelAdapter
    generation_error_prefix = "Flux.1 EPAC"
