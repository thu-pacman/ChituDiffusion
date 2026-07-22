from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from ...epac.cache import CacheConfig
from ...epac.config import EngineConfig
from ...epac.engine import DiffusersEngine
from ...epac.request import DiffusionRequest, RequestStatus
from ...epac.scheduling import SchedulableRequest, StepPlan
from .adapter import ZImageModelAdapter
from .pipeline import EpeZImagePipeline

if TYPE_CHECKING:
    from ...serve.config import EPACServeConfig


@dataclass(frozen=True, slots=True)
class EPACRequest:
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
        if not self.prompt and "prompt_embeds" not in self.extra_inputs:
            raise ValueError("prompt or extra_inputs['prompt_embeds'] is required")
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.width < 16 or self.height < 16:
            raise ValueError("width and height must be at least 16")
        if self.width % 16 or self.height % 16:
            raise ValueError("width and height must be multiples of 16")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")

    def to_diffusion_request(self, *, device: torch.device) -> DiffusionRequest:
        self.cache.require_available()
        inputs = dict(self.extra_inputs)
        reserved = {
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "guidance_scale",
            "generator",
            "num_inference_steps",
            "output_type",
        }
        overlap = reserved.intersection(inputs)
        if overlap:
            raise ValueError(
                "extra_inputs contains reserved fields: " + ", ".join(sorted(overlap))
            )
        inputs.update(
            {
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "width": self.width,
                "height": self.height,
                "guidance_scale": self.guidance_scale,
                "generator": torch.Generator(device=device).manual_seed(self.seed),
            }
        )
        return DiffusionRequest(
            request_id=self.request_id,
            inputs=inputs,
            num_inference_steps=self.num_steps,
            output_type=self.output_type,
            deadline_ms=self.deadline_ms,
            priority=self.priority,
            metadata={"cache": self.cache.strategy},
        )

    def to_service_payload(self) -> dict[str, Any]:
        self.cache.require_available()
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


class _StaticFullWorldPolicy:
    def __init__(self, ranks: tuple[int, ...]) -> None:
        self.ranks = ranks

    def plan(
        self,
        requests: Sequence[SchedulableRequest],
        *,
        pulse_steps: int,
    ) -> Sequence[StepPlan]:
        if len(requests) != 1:
            raise RuntimeError("static generate supports exactly one active request")
        request = requests[0]
        return (
            StepPlan(
                request_id=request.request_id,
                steps=min(pulse_steps, request.profile.remaining_steps),
                lane_ranks=self.ranks,
            ),
        )


class EPACPipeline:
    """Diffusers-style facade for static generation and elastic serving."""

    def __init__(
        self,
        pipeline: EpeZImagePipeline,
        *,
        model_path: str,
        default_cache: CacheConfig,
    ) -> None:
        self._pipeline = pipeline
        self.model_path = str(model_path)
        self.default_cache = default_cache
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        allowed_lane_widths: Sequence[int] | None = None,
        attention_mode: str = "agkv",
        ulysses_degree: int | None = None,
        cache: CacheConfig | None = None,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> "EPACPipeline":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        widths = tuple(
            int(width)
            for width in (
                allowed_lane_widths
                or tuple(
                    width
                    for width in range(1, world_size + 1)
                    if world_size % width == 0
                )
            )
        )
        selected_device = torch.device(
            device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        )
        if selected_device.type == "cuda" and selected_device.index is None:
            selected_device = torch.device("cuda", local_rank)

        pipeline = EpeZImagePipeline.from_pretrained(
            pretrained_model_name_or_path,
            allowed_lane_widths=widths,
            attention_mode=attention_mode,
            ulysses_degree=ulysses_degree,
            **kwargs,
        ).to(selected_device)
        pipeline.set_progress_bar_config(disable=True)
        return cls(
            pipeline,
            model_path=str(pretrained_model_name_or_path),
            default_cache=cache or CacheConfig(),
        )

    @property
    def diffusers_pipeline(self) -> EpeZImagePipeline:
        return self._pipeline

    @property
    def parallel_context(self):
        return self._pipeline.parallel_context

    def generate(self, request: EPACRequest):
        self._ensure_open()
        cache = request.cache
        if cache.strategy == "none" and self.default_cache.strategy != "none":
            cache = self.default_cache
        cache.require_available()
        diffusion_request = request.to_diffusion_request(
            device=torch.device("cuda", self.parallel_context.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        world_ranks = tuple(range(self.parallel_context.world_size))
        engine = DiffusersEngine.from_pipeline(
            self._pipeline,
            adapter=ZImageModelAdapter(),
            config=EngineConfig(
                max_pending_requests=1,
                max_inflight_requests=1,
                pulse_steps=request.num_steps,
            ),
            policy=_StaticFullWorldPolicy(world_ranks),
        )
        result = engine.generate(diffusion_request)
        if result.status is not RequestStatus.COMPLETED:
            detail = result.error.message if result.error is not None else result.status
            raise RuntimeError(f"EPAC generation failed: {detail}")
        return result.output

    def serve(self, config: "EPACServeConfig | None" = None) -> None:
        """Warm up all configured lanes and run the blocking torchrun service."""
        self._ensure_open()
        from ...serve.config import EPACServeConfig
        from ...serve.zimage import serve_zimage_pipeline

        try:
            serve_zimage_pipeline(
                self._pipeline,
                model_path=self.model_path,
                default_cache=self.default_cache,
                config=config or EPACServeConfig(),
            )
        finally:
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._pipeline.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("EPACPipeline is closed")
