from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import torch

from ...epe.api import (
    DiffusersEPEPipeline,
    build_diffusion_request,
    validate_image_request,
)
from ...epe.request import DiffusionRequest
from ...flexcache.config import CacheConfig
from ...parallel.vae import create_vae_parallel_placement
from .pipeline import LLaDAImageDiffusionPipeline

if TYPE_CHECKING:
    from diffusers.image_processor import PipelineImageInput

GenerationMode = Literal["text", "vq", "editing"]


@dataclass(frozen=True, slots=True)
class LLaDAImageRequest:
    """One LLaDA-Image request shared by the offline facade and executor."""

    prompt: str | None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    generation_mode: GenerationMode = "text"
    image: PipelineImageInput | None = None
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 4.5
    seed: int = 0
    max_sequence_length: int = 2048
    deadline_ms: float | None = None
    priority: int = 0
    output_type: str = "pil"
    cache: CacheConfig = field(default_factory=CacheConfig)
    extra_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_image_request(self)
        if self.prompt is not None and not isinstance(self.prompt, str):
            raise ValueError("LLaDA-Image currently supports one prompt per request")
        if self.generation_mode not in {"text", "vq", "editing"}:
            raise ValueError(
                "generation_mode must be one of 'text', 'vq', or 'editing'"
            )
        if self.generation_mode == "editing":
            if self.image is None:
                raise ValueError("image is required when generation_mode='editing'")
        elif self.image is not None:
            raise ValueError(
                f"image must be omitted when generation_mode='{self.generation_mode}'"
            )
        if isinstance(self.image, (list, tuple)):
            raise ValueError(
                "LLaDA-Image currently supports one source image per request"
            )
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        required_multiple = 32 if self.generation_mode == "editing" else 16
        if self.width % required_multiple or self.height % required_multiple:
            raise ValueError(
                f"{self.generation_mode} width and height must be multiples of "
                f"{required_multiple}"
            )
        if self.output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError(
                "output_type must be one of 'pil', 'np', 'pt', or 'latent'"
            )

    @property
    def num_steps(self) -> int:
        """Framework scheduler compatibility alias for the official argument."""

        return self.num_inference_steps

    def to_diffusion_request(self, *, device: torch.device) -> DiffusionRequest:
        self.cache.require_generate_available()
        self.cache.validate_steps(self.num_inference_steps)
        return build_diffusion_request(
            self,
            device=device,
            model_inputs={
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "generation_mode": self.generation_mode,
                "image": self.image,
                "max_sequence_length": self.max_sequence_length,
            },
            reserved_fields={
                "prompt",
                "negative_prompt",
                "generation_mode",
                "image",
                "width",
                "height",
                "guidance_scale",
                "generator",
                "num_inference_steps",
                "max_sequence_length",
                "output_type",
            },
            metadata={"cache": self.cache.to_dict()},
        )


class LLaDAImagePipeline(DiffusersEPEPipeline):
    """ChituDiffusion facade for LLaDA-Image text, VQ, and editing modes."""

    pipeline_class = LLaDAImageDiffusionPipeline
    generation_error_prefix = "LLaDA-Image EPE"

    def __init__(
        self,
        pipeline: LLaDAImageDiffusionPipeline,
        *,
        model_path: str,
        default_cache: CacheConfig | None = None,
        parallel_vae: bool = False,
        vae_parallel_halo: int = 8,
    ) -> None:
        super().__init__(pipeline, model_path=model_path)
        self.default_cache = default_cache or CacheConfig()
        self.parallel_vae = bool(parallel_vae)
        self.vae_parallel_halo = int(vae_parallel_halo)
        if self.vae_parallel_halo < 0:
            raise ValueError("vae_parallel_halo must be non-negative")

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
        parallel_vae: bool = False,
        vae_parallel_halo: int = 8,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> "LLaDAImagePipeline":
        if vae_parallel_halo < 0:
            raise ValueError("vae_parallel_halo must be non-negative")
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
        facade.parallel_vae = bool(parallel_vae)
        facade.vae_parallel_halo = int(vae_parallel_halo)
        return facade

    def _before_generate(self, request: LLaDAImageRequest) -> None:
        cache = request.cache
        if cache.strategy == "none" and self.default_cache.strategy != "none":
            cache = self.default_cache
        if cache.strategy != "none":
            raise NotImplementedError(
                "LLaDA-Image FlexCache requires a validated model contract"
            )

    def _create_backend(self, config: Any | None = None) -> Any:
        from .epe import LLaDAImageEpeModule
        from .executor import LLaDAImageDecoderExecutor

        parallel_vae = getattr(config, "parallel_vae", None)
        if parallel_vae is None:
            parallel_vae = self.parallel_vae

        scheduling_module = getattr(self._pipeline.transformer, "epe", None)
        if scheduling_module is None:
            scheduling_module = LLaDAImageEpeModule(
                self.parallel_context,
                policy=str(getattr(config, "schedule_strategy", "static_cp")),
            )
        return LLaDAImageDecoderExecutor(
            self._pipeline,
            scheduling_module,
            default_width=int(getattr(config, "default_width", 1024)),
            default_height=int(getattr(config, "default_height", 1024)),
            default_num_steps=int(getattr(config, "default_num_steps", 50)),
            vae_placement=create_vae_parallel_placement(
                getattr(config, "vae_parallel_degree", None) if parallel_vae else 1,
                halo=int(getattr(config, "vae_parallel_halo", self.vae_parallel_halo)),
            ),
        )

    def serve(self, config: Any | None = None) -> None:
        """Serve text requests with exact full-image VAE decode by default."""

        from ...serve.config import EPEServeConfig

        selected = config if config is not None else EPEServeConfig(parallel_vae=False)
        super().serve(selected)
