from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import torch

from .request import DiffusionRequest


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


class DiffusersEPEPipeline:
    """Shared facade around one model pipeline and diffusion backend."""

    pipeline_class: ClassVar[type]
    generation_error_prefix: ClassVar[str] = "EPE"

    def __init__(self, pipeline: Any, *, model_path: str) -> None:
        self._pipeline = pipeline
        self.model_path = str(model_path)
        self._closed = False
        self.last_cache_stats: dict[str, Any] | None = None
        self.last_vae_stats: dict[str, Any] | None = None

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
    ) -> "DiffusersEPEPipeline":
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

    def _create_backend(self, config: Any | None = None) -> Any:
        del config
        raise NotImplementedError(
            f"{type(self).__name__} must provide a shared diffusion backend"
        )

    def generate(self, request: Any) -> Any:
        self._ensure_open()
        self._before_generate(request)
        try:
            backend = self._create_backend()
            output = backend.generate(request)
            self.last_cache_stats = backend.last_cache_stats
            placement = getattr(backend, "vae_placement", None)
            self.last_vae_stats = (
                None
                if placement is None
                else {
                    "parallel_vae": placement.sharded,
                    "vae_parallel_degree": placement.degree,
                    "vae_parallel_halo": placement.halo,
                }
            )
            decode_stats = getattr(
                getattr(self._pipeline, "vae", None), "_chitu_vae_decode_stats", None
            )
            if self.last_vae_stats is not None and decode_stats is not None:
                self.last_vae_stats.update(decode_stats)
            return output
        except Exception as exc:
            raise RuntimeError(
                f"{self.generation_error_prefix} generation failed: {exc}"
            ) from exc

    def serve(self, config: Any | None = None) -> None:
        """Run the loaded backend with the production EPE service lifecycle."""
        self._ensure_open()
        from ..serve.config import EPEServeConfig
        from ..serve.runner import serve_diffusion_backend

        selected = config or EPEServeConfig()
        selected.cache.require_serve_available()
        try:
            serve_diffusion_backend(
                self._create_backend(selected),
                model_path=self.model_path,
                config=selected,
                stage_name=type(self).__name__.removesuffix("Pipeline").lower()
                or "diffusion",
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
            raise RuntimeError(f"{type(self).__name__} is closed")
