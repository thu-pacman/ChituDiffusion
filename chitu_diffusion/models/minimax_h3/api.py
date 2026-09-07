from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from ...flexcache.config import CacheConfig


@dataclass(frozen=True, slots=True)
class MiniMaxH3Request:
    """Native FL2VA request, with an explicit legacy latent test mode."""

    conditioning_path: str | None = None
    prompt: str | None = None
    first_frame_path: str | None = None
    last_frame_path: str | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    width: int = 8
    height: int = 8
    latent_t: int = 2
    audio_t: int = 3
    audio_channels: int = 2
    text_length: int = 32
    num_steps: int = 50
    flow_shift: float | None = None
    audio_flow_shift: float | None = None
    synthetic: bool = False
    seed: int = 0
    deadline_ms: float | None = None
    priority: int = 0
    output_type: str = "latent"
    duration_s: float = 5.0
    fps: int = 24
    cache: CacheConfig = field(default_factory=CacheConfig)
    extra_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        native = not self.synthetic and self.conditioning_path is None
        if native and not (self.prompt or "").strip():
            raise ValueError("prompt is required for a native FL2VA request")
        for name in (
            "width",
            "height",
            "latent_t",
            "audio_t",
            "audio_channels",
            "text_length",
            "num_steps",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_steps < 2:
            raise ValueError("num_steps must contain at least two sigma points")
        alignment = 32 if native else 2
        if self.width % alignment or self.height % alignment:
            kind = "pixel" if native else "latent"
            raise ValueError(
                f"{kind} width and height must be divisible by {alignment}"
            )
        if self.flow_shift is not None and self.flow_shift <= 0:
            raise ValueError("flow_shift must be positive")
        if self.audio_flow_shift is not None and self.audio_flow_shift <= 0:
            raise ValueError("audio_flow_shift must be positive")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        if self.output_type not in {"latent", "mp4"}:
            raise ValueError("output_type must be latent or mp4")
        if self.output_type == "mp4" and not native:
            raise ValueError("mp4 output is only available for native FL2VA requests")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.fps != 24:
            raise ValueError("MiniMax-H3 FL2VA currently requires fps=24")
        if self.extra_inputs:
            raise ValueError("MiniMax-H3 request does not support extra_inputs")
        if self.cache.strategy != "none":
            raise ValueError("MiniMax-H3 latent backend does not support FlexCache")
