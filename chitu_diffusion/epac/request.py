from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class RequestStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            RequestStatus.COMPLETED,
            RequestStatus.CANCELLED,
            RequestStatus.FAILED,
        }


@dataclass(frozen=True, slots=True)
class DiffusionRequest:
    """Framework-neutral request whose inputs use Diffusers argument names."""

    inputs: Mapping[str, Any]
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    num_inference_steps: int = 50
    output_type: str = "pil"
    deadline_ms: float | None = None
    priority: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    submitted_at: float
    started_at: float | None
    completed_at: float
    steps_executed: int

    @property
    def queue_delay_ms(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.started_at - self.submitted_at) * 1000.0

    @property
    def latency_ms(self) -> float:
        return (self.completed_at - self.submitted_at) * 1000.0


@dataclass(frozen=True, slots=True)
class StructuredError:
    type: str
    message: str


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    request_id: str
    status: RequestStatus
    metrics: RequestMetrics
    output: Any | None = None
    error: StructuredError | None = None


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    request_id: str
    status: RequestStatus
    submitted_at: float
    started_at: float | None
    completed_at: float | None
    steps_executed: int
    error: StructuredError | None = None
