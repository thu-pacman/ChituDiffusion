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
from .pipeline import EpeQwenImagePipeline


@dataclass(frozen=True, slots=True)
class QwenImageRequest:
    prompt: str | None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    negative_prompt: str | None = " "
    width: int = 1024
    height: int = 1024
    num_steps: int = 50
    true_cfg_scale: float = 4.0
    seed: int = 0
    deadline_ms: float | None = None
    priority: int = 0
    output_type: str = "pil"
    cache: CacheConfig = field(default_factory=CacheConfig)
    extra_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_image_request(self)
        if self.width % 16 or self.height % 16:
            raise ValueError("Qwen-Image width and height must be multiples of 16")
        if self.true_cfg_scale < 0:
            raise ValueError("true_cfg_scale must be non-negative")

    @property
    def guidance_scale(self) -> None:
        """Qwen-Image is not guidance-distilled; traditional CFG is separate."""
        return None

    def to_diffusion_request(self, *, device: torch.device) -> DiffusionRequest:
        self.cache.require_generate_available()
        self.cache.validate_steps(self.num_steps)
        return build_diffusion_request(
            self,
            device=device,
            model_inputs={
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "true_cfg_scale": self.true_cfg_scale,
                "guidance_scale": None,
            },
            reserved_fields={
                "prompt",
                "negative_prompt",
                "width",
                "height",
                "true_cfg_scale",
                "guidance_scale",
                "generator",
                "num_inference_steps",
                "output_type",
            },
            metadata={"cache": self.cache.to_dict()},
        )


class QwenImagePipeline(DiffusersEPEPipeline):
    """Diffusers-style facade for static Qwen-Image generation."""

    pipeline_class = EpeQwenImagePipeline
    generation_error_prefix = "Qwen-Image EPE"

    def _create_backend(self, config: Any | None = None) -> Any:
        from ...epe.scheduling.planner import EpeSchedulingModule
        from .executor import QwenImageDecoderExecutor

        parallel_vae = getattr(config, "parallel_vae", None)

        return QwenImageDecoderExecutor(
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
