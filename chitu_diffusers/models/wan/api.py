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
from .adapter import WanModelAdapter
from .pipeline import EpeWanPipeline


@dataclass(frozen=True, slots=True)
class WanRequest:
    prompt: str | None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    negative_prompt: str | None = ""
    width: int = 832
    height: int = 480
    num_frames: int = 81
    num_steps: int = 50
    guidance_scale: float = 6.0
    seed: int = 0
    deadline_ms: float | None = None
    priority: int = 0
    output_type: str = "np"
    extra_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_image_request(self)
        if self.num_frames < 1 or (self.num_frames - 1) % 4:
            raise ValueError("Wan num_frames must be positive and equal to 4n+1")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        if self.output_type not in {"np", "pt", "pil", "latent"}:
            raise ValueError("Wan output_type must be np, pt, pil, or latent")

    def to_diffusion_request(self, *, device: torch.device) -> DiffusionRequest:
        return build_diffusion_request(
            self,
            device=device,
            model_inputs={
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "num_frames": self.num_frames,
            },
            reserved_fields={
                "prompt",
                "negative_prompt",
                "width",
                "height",
                "num_frames",
                "guidance_scale",
                "generator",
                "num_inference_steps",
                "output_type",
            },
            metadata={"num_frames": self.num_frames},
        )


class WanEPACPipeline(DiffusersEPACPipeline):
    """Diffusers-style facade for Wan2.1 T2V EPAC generation."""

    pipeline_class = EpeWanPipeline
    adapter_class = WanModelAdapter
    generation_error_prefix = "Wan EPAC"
