from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int = 0
    num_steps: int | None = None
    guidance_scale: float = 5.0
    deadline_ms: float | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "ImageGenerateRequest":
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be provided together")
        for name, value in (("width", self.width), ("height", self.height)):
            if value is not None and (value < 16 or value % 16):
                raise ValueError(f"{name} must be a positive multiple of 16")
        if self.num_steps is not None and self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        return self


RequestStatus = Literal[
    "pending",
    "running",
    "postprocessing",
    "completed",
    "cancelled",
    "failed",
]


class AdmissionResponse(BaseModel):
    request_id: str
    status: Literal["accepted"] = "accepted"
    status_url: str
    image_url: str


class RequestStatusResponse(BaseModel):
    request_id: str
    status: RequestStatus
    submitted_at: float
    first_scheduled_at: float | None = None
    completed_at: float | None = None
    queue_delay_ms: float | None = None
    latency_ms: float | None = None
    error: str | None = None
    image_url: str | None = None


class CancelResponse(BaseModel):
    request_id: str
    cancelled: bool
    detail: str


class HealthResponse(BaseModel):
    state: Literal["starting", "running", "stopping", "failed"]
    rank: int
    world_size: int
    queue_depth: int
    inflight: int
    completed: int
    fatal_error: str | None = None
