from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist

from ..parallel import EpeParallelContext
from .image_decoder import ExecutorBuildContext, TransferBundle
from .scheduling import RequestProfile


def scheduling_options_from_pool(pool: Any) -> dict[str, Any]:
    """Translate embedded pool configuration into shared EPE options."""

    return {
        "policy": pool.policy,
        "switch_allowed_until_step": pool.switch_allowed_until_step,
        "pulse_steps": pool.pulse_steps,
        "balanced_k": pool.balanced_k,
        "starvation_ms": pool.starvation_ms,
        "deadline_guard_ms": pool.deadline_guard_ms,
        "max_active_requests": pool.max_inflight_requests,
        "online_calibration": pool.online_calibration,
    }


def build_stage_parallel_context(
    context: ExecutorBuildContext,
    *,
    attention_mode: str,
    ulysses_degree: int | None,
) -> tuple[EpeParallelContext, int]:
    """Validate a host stage world and attach EPAC to its process group."""

    world = context.world
    local_suffix = str(world.local_device).rsplit(":", 1)[-1]
    local_rank = int(local_suffix) if local_suffix.isdigit() else 0
    if dist.is_initialized():
        if (
            world.process_group is not None
            and world.process_group is not dist.group.WORLD
        ):
            raise NotImplementedError(
                "the embedded runtime supports an existing WORLD process group, "
                "not an arbitrary subgroup"
            )
        if dist.get_rank() != world.rank or dist.get_world_size() != world.world_size:
            raise ValueError(
                "initialized torch.distributed WORLD mismatches StageWorldSpec"
            )
    else:
        if world.process_group is not None:
            raise ValueError("process_group was supplied before torch.distributed init")
        if not world.owns_process_group:
            raise ValueError(
                "owns_process_group=False requires an initialized process group"
            )
        if world.world_size > 1 and world.master_port == 0:
            raise ValueError("master_port is required for a multi-rank stage world")
        os.environ["MASTER_ADDR"] = world.master_addr
        os.environ["MASTER_PORT"] = str(world.master_port)

    os.environ["RANK"] = str(world.rank)
    os.environ["WORLD_SIZE"] = str(world.world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    degree = ulysses_degree
    if degree is None:
        degree = 2 if attention_mode == "usp" else 1
    parallel = EpeParallelContext.from_torchrun(
        allowed_widths=context.pool.allowed_lane_widths,
        owns_process_group=world.owns_process_group,
        ulysses_degree=degree,
    )
    return parallel, local_rank


class DiffusersImageDecoderExecutor(ABC):
    """Reusable executor lifecycle with model-specific request and warmup hooks."""

    def __init__(
        self,
        pipeline: Any,
        scheduling_module: Any,
        *,
        default_width: int = 1024,
        default_height: int = 1024,
        default_num_steps: int = 50,
    ) -> None:
        self.pipeline = pipeline
        self.scheduling_module = scheduling_module
        self.default_width = int(default_width)
        self.default_height = int(default_height)
        self.default_num_steps = int(default_num_steps)

    @property
    def parallel_context(self) -> EpeParallelContext:
        return self.pipeline.parallel_context

    @abstractmethod
    def warmup(
        self,
        *,
        resolutions: tuple[tuple[int, int], ...],
        steps: int,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def normalize_request(self, request: Any) -> Any: ...

    @abstractmethod
    def serialize_request(self, request: Any) -> dict[str, Any]: ...

    @abstractmethod
    def prepare_request(self, request: Any) -> Any: ...

    @abstractmethod
    def _request_conditions(self, request: Any) -> int: ...

    @abstractmethod
    def _request_state_bytes(self, request: Any, image_tokens: int) -> int: ...

    @abstractmethod
    def _state_conditions(self, state: Any) -> int: ...

    def request_id(self, request: Any) -> str:
        return str(self.normalize_request(request).request_id)

    def request_deadline_ms(self, request: Any) -> float | None:
        return self.normalize_request(request).deadline_ms

    def validate_request(
        self,
        request: Any,
        *,
        warmup_resolutions: tuple[tuple[int, int], ...],
    ) -> None:
        normalized = self.normalize_request(request)
        shape = (int(normalized.height), int(normalized.width))
        if shape not in set(warmup_resolutions):
            formatted = ", ".join(
                f"{height}x{width}" for height, width in sorted(warmup_resolutions)
            )
            raise ValueError(
                f"unsupported resolution {shape[0]}x{shape[1]}; "
                f"warmup profile supports: {formatted}"
            )

    def deserialize_request(self, payload: Any) -> Any:
        return self.normalize_request(payload)

    def request_profile(
        self,
        request: Any,
        *,
        completed_steps: int,
    ) -> RequestProfile:
        normalized = self.normalize_request(request)
        height = int(normalized.height)
        width = int(normalized.width)
        image_tokens = (height // 16) * (width // 16)
        return RequestProfile(
            total_steps=int(normalized.num_steps),
            completed_steps=int(completed_steps),
            shape_key=(height, width),
            image_tokens=image_tokens,
            attributes={
                "batch_size": 1,
                "conditions": self._request_conditions(normalized),
                "state_bytes": self._request_state_bytes(
                    normalized,
                    image_tokens,
                ),
            },
        )

    def profile(self, state: Any) -> RequestProfile:
        return RequestProfile(
            total_steps=len(state.timesteps),
            completed_steps=int(state.step_index),
            shape_key=(int(state.height), int(state.width)),
            image_tokens=int(state.image_tokens),
            attributes={
                "batch_size": int(state.actual_batch_size),
                "conditions": self._state_conditions(state),
                "state_bytes": state.latents.numel() * state.latents.element_size(),
            },
        )

    def denoise_step(self, state: Any, *, lane_ranks: tuple[int, ...]) -> None:
        self.pipeline.denoise_step(state, lane_ranks=lane_ranks)

    def synchronize_state(self, state: Any, *, step_index: int) -> None:
        self.pipeline.synchronize_state(state, step_index=step_index)

    def export_state(self, state: Any) -> TransferBundle:
        return TransferBundle(
            tensors={"latents": state.latents},
            step_index=int(state.step_index),
        )

    def _decode_kwargs(self) -> dict[str, Any]:
        return {}

    def finalize_gpu(
        self,
        state: Any,
        *,
        lane_ranks: tuple[int, ...],
        timings: dict[str, object],
    ) -> torch.Tensor | None:
        with self.parallel_context.activate(lane_ranks) as topology:
            return self.pipeline.decode_request(
                state,
                topology=topology,
                timings=timings,
                **self._decode_kwargs(),
            )

    def postprocess(self, host_output: torch.Tensor) -> Any:
        return self.pipeline.image_processor.postprocess(
            host_output,
            output_type="pil",
        )

    def abort_request(self, request_id: str, state: Any | None) -> None:
        del request_id, state

    def close(self) -> None:
        self.pipeline.close()
