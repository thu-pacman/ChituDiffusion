from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import torch

from .config import EngineConfig
from .engine import DiffusersEngine
from .request import DiffusionRequest, RequestStatus
from .scheduling import SchedulableRequest, StepPlan


def validate_image_request(request: Any) -> None:
    """Validate fields shared by model-specific typed image requests."""

    if not request.prompt and "prompt_embeds" not in request.extra_inputs:
        raise ValueError("prompt or extra_inputs['prompt_embeds'] is required")
    if not request.request_id:
        raise ValueError("request_id must not be empty")
    if request.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if request.width < 16 or request.height < 16:
        raise ValueError("width and height must be at least 16")
    if request.width % 16 or request.height % 16:
        raise ValueError("width and height must be multiples of 16")
    if request.deadline_ms is not None and request.deadline_ms <= 0:
        raise ValueError("deadline_ms must be positive")


def build_diffusion_request(
    request: Any,
    *,
    device: torch.device,
    model_inputs: Mapping[str, Any],
    reserved_fields: set[str],
    metadata: Mapping[str, Any] | None = None,
) -> DiffusionRequest:
    """Build the engine request while protecting model-owned input fields."""

    inputs = dict(request.extra_inputs)
    overlap = reserved_fields.intersection(inputs)
    if overlap:
        raise ValueError(
            "extra_inputs contains reserved fields: " + ", ".join(sorted(overlap))
        )
    inputs.update(model_inputs)
    inputs.update(
        {
            "width": request.width,
            "height": request.height,
            "guidance_scale": request.guidance_scale,
            "generator": torch.Generator(device=device).manual_seed(request.seed),
        }
    )
    return DiffusionRequest(
        request_id=request.request_id,
        inputs=inputs,
        num_inference_steps=request.num_steps,
        output_type=request.output_type,
        deadline_ms=request.deadline_ms,
        priority=request.priority,
        metadata=dict(metadata or {}),
    )


class StaticFullWorldPolicy:
    """Single-request policy used by the synchronous Diffusers-style API."""

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


class DiffusersEPACPipeline:
    """Shared synchronous facade around a model pipeline and step adapter."""

    pipeline_class: ClassVar[type]
    adapter_class: ClassVar[type]
    generation_error_prefix: ClassVar[str] = "EPAC"

    def __init__(self, pipeline: Any, *, model_path: str) -> None:
        self._pipeline = pipeline
        self.model_path = str(model_path)
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        allowed_lane_widths: Sequence[int] | None = None,
        attention_mode: str = "agkv",
        ulysses_degree: int | None = None,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> "DiffusersEPACPipeline":
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
        pipeline = cls.pipeline_class.from_pretrained(
            pretrained_model_name_or_path,
            allowed_lane_widths=widths,
            attention_mode=attention_mode,
            ulysses_degree=ulysses_degree,
            **kwargs,
        ).to(selected_device)
        pipeline.set_progress_bar_config(disable=True)
        return cls(pipeline, model_path=str(pretrained_model_name_or_path))

    @property
    def diffusers_pipeline(self) -> Any:
        return self._pipeline

    @property
    def parallel_context(self) -> Any:
        return self._pipeline.parallel_context

    def _before_generate(self, request: Any) -> None:
        del request

    def generate(self, request: Any) -> Any:
        self._ensure_open()
        self._before_generate(request)
        device = (
            torch.device("cuda", self.parallel_context.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        diffusion_request = request.to_diffusion_request(device=device)
        world_ranks = tuple(range(self.parallel_context.world_size))
        engine = DiffusersEngine.from_pipeline(
            self._pipeline,
            adapter=self.adapter_class(),
            config=EngineConfig(
                max_pending_requests=1,
                max_inflight_requests=1,
                pulse_steps=request.num_steps,
            ),
            policy=StaticFullWorldPolicy(world_ranks),
        )
        result = engine.generate(diffusion_request)
        if result.status is not RequestStatus.COMPLETED:
            detail = result.error.message if result.error is not None else result.status
            raise RuntimeError(
                f"{self.generation_error_prefix} generation failed: {detail}"
            )
        return result.output

    def close(self) -> None:
        if self._closed:
            return
        self._pipeline.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
