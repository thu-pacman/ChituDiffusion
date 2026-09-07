"""Public request type for Hunyuan Image 3 generation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

SUPPORTED_OUTPUT_TYPES = ("pil", "pt", "np", "latent")


@dataclass(frozen=True, slots=True)
class HunyuanImage3Request:
    """One fixed-size image request.

    ``height`` and ``width`` are snapped onto the released resolution group
    during preparation, so the accepted values are the ones the token grid can
    represent rather than arbitrary pixel sizes.
    """

    prompt: str | None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    image_paths: tuple[str, ...] = ()
    height: int = 1024
    width: int = 1024
    num_steps: int = 50
    guidance_scale: float = 5.0
    seed: int = 0
    system_prompt: str | None = None
    deadline_ms: float | None = None
    priority: int = 0
    output_type: str = "pil"
    extra_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not (self.prompt or "").strip():
            raise ValueError("Hunyuan Image 3 requires a prompt")
        if self.height < 16 or self.width < 16:
            raise ValueError("image dimensions must be at least 16 pixels")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.guidance_scale <= 1.0:
            raise ValueError("guidance_scale must be greater than 1")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        if self.output_type not in SUPPORTED_OUTPUT_TYPES:
            raise ValueError(
                "output_type must be one of " + ", ".join(SUPPORTED_OUTPUT_TYPES)
            )
        if self.extra_inputs:
            raise ValueError(
                "Hunyuan Image 3 does not accept extra_inputs; text generation, "
                "recaption, and automatic sizing are not part of this release"
            )


__all__ = ["SUPPORTED_OUTPUT_TYPES", "HunyuanImage3Request"]
