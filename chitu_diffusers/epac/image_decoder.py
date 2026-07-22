from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
import torch.distributed as dist

from .request import StructuredError
from .scheduling import RequestProfile


@dataclass(frozen=True, slots=True)
class StageWorldSpec:
    """Stage-local distributed world supplied by the embedding host."""

    stage_name: str
    rank: int
    world_size: int
    local_device: str
    physical_device_ids: tuple[int, ...]
    leader_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 0
    process_group: object | None = None
    owns_process_group: bool = True

    def __post_init__(self) -> None:
        if not self.stage_name:
            raise ValueError("stage_name must not be empty")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be within the stage world")
        if not 0 <= self.leader_rank < self.world_size:
            raise ValueError("leader_rank must be within the stage world")
        if len(self.physical_device_ids) != self.world_size:
            raise ValueError("physical_device_ids must contain one id per stage rank")
        if len(set(self.physical_device_ids)) != self.world_size:
            raise ValueError("physical_device_ids must be unique")
        if self.master_port < 0 or self.master_port > 65535:
            raise ValueError("master_port must be between 0 and 65535")

    @property
    def is_leader(self) -> bool:
        return self.rank == self.leader_rank

    @classmethod
    def from_torchrun(
        cls,
        stage_name: str,
        *,
        physical_device_ids: tuple[int, ...] | None = None,
        leader_rank: int = 0,
        owns_process_group: bool | None = None,
    ) -> "StageWorldSpec":
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if physical_device_ids is None:
            visible = tuple(
                int(value.strip())
                for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if value.strip().isdigit()
            )
            physical_device_ids = (
                visible if len(visible) == world_size else tuple(range(world_size))
            )
        initialized = dist.is_available() and dist.is_initialized()
        return cls(
            stage_name=stage_name,
            rank=rank,
            world_size=world_size,
            local_device=(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"),
            physical_device_ids=physical_device_ids,
            leader_rank=leader_rank,
            master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
            master_port=int(os.environ.get("MASTER_PORT", "0")),
            process_group=dist.group.WORLD if initialized else None,
            owns_process_group=(
                not initialized
                if owns_process_group is None
                else bool(owns_process_group)
            ),
        )


@dataclass(frozen=True, slots=True)
class TransferBundle:
    """Canonical tensors and metadata required to move one request state."""

    tensors: Mapping[str, torch.Tensor]
    step_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutorBuildContext:
    world: StageWorldSpec
    pool: Any


@dataclass(frozen=True, slots=True)
class EmbeddedRuntimeConfig:
    output_root: str = "outputs/chitu-diffusers"
    record_timeline: bool = False
    postprocess_workers: int = 4

    def __post_init__(self) -> None:
        if self.postprocess_workers < 1:
            raise ValueError("postprocess_workers must be positive")


@dataclass(frozen=True, slots=True)
class LaneSnapshot:
    request_id: str
    ranks: tuple[int, ...]
    current_step: int


@dataclass(frozen=True, slots=True)
class EmbeddedRuntimeHealth:
    state: str
    rank: int
    world_size: int
    queue_depth: int
    inflight: int
    completed: int
    active_lanes: tuple[LaneSnapshot, ...] = ()
    last_plan_version: int = -1
    fatal_error: str | None = None


@dataclass(frozen=True, slots=True)
class ImageDecodeCompletion:
    request_id: str
    status: str
    output: Any | None = None
    error: StructuredError | None = None
    submitted_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None


@runtime_checkable
class ImageDecoderExecutor(Protocol):
    """Model plugin consumed by the distributed EPAC worker runtime.

    The runtime owns scheduling and transfers. The executor owns request/state
    semantics and the model-specific Diffusers operations.
    """

    @property
    def parallel_context(self) -> Any: ...

    @property
    def scheduling_module(self) -> Any: ...

    def warmup(
        self,
        *,
        resolutions: tuple[tuple[int, int], ...],
        steps: int,
    ) -> dict[str, Any]: ...

    def normalize_request(self, request: Any) -> Any: ...

    def request_id(self, request: Any) -> str: ...

    def request_deadline_ms(self, request: Any) -> float | None: ...

    def validate_request(
        self,
        request: Any,
        *,
        warmup_resolutions: tuple[tuple[int, int], ...],
    ) -> None: ...

    def serialize_request(self, request: Any) -> Any: ...

    def deserialize_request(self, payload: Any) -> Any: ...

    def request_profile(
        self,
        request: Any,
        *,
        completed_steps: int,
    ) -> RequestProfile: ...

    def prepare_request(self, request: Any) -> Any: ...

    def profile(self, state: Any) -> RequestProfile: ...

    def denoise_step(
        self,
        state: Any,
        *,
        lane_ranks: tuple[int, ...],
    ) -> None: ...

    def synchronize_state(self, state: Any, *, step_index: int) -> None: ...

    def export_state(self, state: Any) -> TransferBundle: ...

    def finalize_gpu(
        self,
        state: Any,
        *,
        timings: dict[str, object],
    ) -> torch.Tensor: ...

    def postprocess(self, host_output: torch.Tensor) -> Any: ...

    def abort_request(self, request_id: str, state: Any | None) -> None: ...

    def close(self) -> None: ...


class ImageDecoderExecutorFactory(Protocol):
    def build(self, context: ExecutorBuildContext) -> ImageDecoderExecutor: ...
