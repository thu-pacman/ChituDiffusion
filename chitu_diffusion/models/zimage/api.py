from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ...epe.api import (
    DiffusersEPEPipeline,
    build_diffusion_request,
    validate_image_request,
)
from ...epe.request import DiffusionRequest
from ...flexcache.config import CacheConfig
from ...parallel.vae import create_vae_parallel_placement
from .pipeline import EpeZImagePipeline

if TYPE_CHECKING:
    from ...serve.config import EPEServeConfig


@dataclass(frozen=True, slots=True)
class ZImageRequest:
    """Typed Z-Image request shared by offline and service-facing APIs."""

    prompt: str | None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    num_steps: int = 50
    guidance_scale: float = 5.0
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
            model_inputs={
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
            },
            reserved_fields={
                "prompt",
                "negative_prompt",
                "width",
                "height",
                "guidance_scale",
                "generator",
                "num_inference_steps",
                "output_type",
            },
            metadata={"cache": self.cache.to_dict()},
        )

    def to_service_payload(self) -> dict[str, Any]:
        self.cache.require_serve_available()
        if not self.prompt:
            raise ValueError("the HTTP service request requires a text prompt")
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "guidance_scale": self.guidance_scale,
            "deadline_ms": self.deadline_ms,
        }


class ZImagePipeline(DiffusersEPEPipeline):
    """Diffusers-style facade for static generation and elastic serving."""

    pipeline_class = EpeZImagePipeline

    def __init__(
        self,
        pipeline: EpeZImagePipeline,
        *,
        model_path: str,
        default_cache: CacheConfig | None = None,
    ) -> None:
        super().__init__(pipeline, model_path=model_path)
        self.default_cache = default_cache or CacheConfig()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        allowed_lane_widths: Sequence[int] | None = None,
        attention_mode: str = "agkv",
        ulysses_degree: int | None = None,
        cfg_parallel: bool = True,
        cache: CacheConfig | None = None,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> "ZImagePipeline":
        facade = super().from_pretrained(
            pretrained_model_name_or_path,
            allowed_lane_widths=allowed_lane_widths,
            attention_mode=attention_mode,
            ulysses_degree=ulysses_degree,
            cfg_parallel=cfg_parallel,
            device=device,
            **kwargs,
        )
        assert isinstance(facade, cls)
        facade.default_cache = cache or CacheConfig()
        return facade

    def _before_generate(self, request: ZImageRequest) -> None:
        cache = request.cache
        if cache.strategy == "none" and self.default_cache.strategy != "none":
            cache = self.default_cache
        cache.require_generate_available()
        cache.validate_steps(request.num_steps)

    def generate(self, request: ZImageRequest) -> Any:
        if request.cache.strategy == "none" and self.default_cache.strategy != "none":
            request = replace(request, cache=self.default_cache)
        return super().generate(request)

    def _create_backend(self, config: Any | None = None) -> Any:
        from .executor import ZImageImageDecoderExecutor

        return ZImageImageDecoderExecutor(
            self._pipeline,
            default_width=int(getattr(config, "default_width", 1024)),
            default_height=int(getattr(config, "default_height", 1024)),
            default_num_steps=int(getattr(config, "default_num_steps", 50)),
            cfg_parallel=bool(
                getattr(
                    config,
                    "cfg_parallel",
                    getattr(self._pipeline.transformer.epe, "cfg_parallel", True),
                )
            ),
            vae_placement=create_vae_parallel_placement(
                getattr(config, "vae_parallel_degree", None),
                halo=int(getattr(config, "vae_parallel_halo", 8)),
            ),
        )

    def serve(self, config: "EPEServeConfig | None" = None) -> None:
        """Warm up all configured lanes and run the blocking torchrun service."""
        self._ensure_open()
        from ...serve.config import EPEServeConfig
        from ...serve.runner import serve_diffusion_backend

        try:
            serve_config = config or EPEServeConfig()
            serve_config.cache.require_serve_available()
            serve_diffusion_backend(
                self._create_backend(serve_config),
                model_path=self.model_path,
                config=serve_config,
                stage_name="epe_zimage",
            )
        finally:
            self._closed = True
