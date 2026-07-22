from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ...epac.image_decoder import (
    ExecutorBuildContext,
    ImageDecoderExecutor,
    TransferBundle,
)
from ...epac.scheduling import RequestProfile
from .api import EPACRequest
from ...parallel import EpeParallelContext
from .pipeline import EpeZImagePipeline, ZImageDenoiseState


class ZImageImageDecoderExecutor(ImageDecoderExecutor):
    """Z-Image model operations behind the generic decoder contract."""

    def __init__(
        self,
        pipeline: EpeZImagePipeline,
        *,
        default_width: int = 1024,
        default_height: int = 1024,
        default_num_steps: int = 50,
    ) -> None:
        self.pipeline = pipeline
        self.default_width = int(default_width)
        self.default_height = int(default_height)
        self.default_num_steps = int(default_num_steps)

    @property
    def parallel_context(self):
        return self.pipeline.parallel_context

    @property
    def scheduling_module(self):
        return self.pipeline.transformer.epe

    def warmup(self, *, resolutions, steps):
        return self.pipeline.warmup_epe(resolutions=resolutions, steps=steps)

    def normalize_request(self, request: Any) -> EPACRequest:
        if isinstance(request, EPACRequest):
            return request
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        if not isinstance(request, Mapping):
            raise TypeError("Z-Image requests must be EPACRequest or a mapping")
        return EPACRequest(
            prompt=request.get("prompt"),
            request_id=str(request.get("request_id") or uuid.uuid4().hex),
            negative_prompt=request.get("negative_prompt"),
            width=int(request.get("width") or self.default_width),
            height=int(request.get("height") or self.default_height),
            num_steps=int(request.get("num_steps") or self.default_num_steps),
            guidance_scale=float(request.get("guidance_scale", 5.0)),
            seed=int(request.get("seed", 0)),
            deadline_ms=(
                None
                if request.get("deadline_ms") is None
                else float(request["deadline_ms"])
            ),
            priority=int(request.get("priority", 0)),
        )

    def request_id(self, request: Any) -> str:
        return str(self.normalize_request(request).request_id)

    def request_deadline_ms(self, request: Any) -> float | None:
        return self.normalize_request(request).deadline_ms

    def _shape(self, request: EPACRequest) -> tuple[int, int]:
        return request.height, request.width

    def validate_request(self, request, *, warmup_resolutions) -> None:
        normalized = self.normalize_request(request)
        shape = self._shape(normalized)
        if shape not in set(warmup_resolutions):
            formatted = ", ".join(
                f"{height}x{width}" for height, width in sorted(warmup_resolutions)
            )
            raise ValueError(
                f"unsupported resolution {shape[0]}x{shape[1]}; "
                f"warmup profile supports: {formatted}"
            )

    def serialize_request(self, request: Any) -> dict[str, Any]:
        normalized = self.normalize_request(request)
        return {
            "request_id": normalized.request_id,
            "prompt": normalized.prompt,
            "negative_prompt": normalized.negative_prompt,
            "width": normalized.width,
            "height": normalized.height,
            "seed": normalized.seed,
            "num_steps": normalized.num_steps,
            "guidance_scale": normalized.guidance_scale,
            "deadline_ms": normalized.deadline_ms,
            "priority": normalized.priority,
        }

    def deserialize_request(self, payload: Any) -> EPACRequest:
        return self.normalize_request(payload)

    def request_profile(self, request, *, completed_steps) -> RequestProfile:
        normalized = self.normalize_request(request)
        height, width = self._shape(normalized)
        steps = normalized.num_steps
        return RequestProfile(
            total_steps=steps,
            completed_steps=int(completed_steps),
            image_tokens=(height // 16) * (width // 16),
            attributes={
                "batch_size": 1,
                "conditions": 2 if normalized.guidance_scale > 0 else 1,
            },
        )

    def prepare_request(self, request: Any) -> ZImageDenoiseState:
        normalized = self.normalize_request(request)
        height, width = self._shape(normalized)
        steps = normalized.num_steps
        device = (
            torch.device("cuda", self.parallel_context.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        generator = torch.Generator(device=device).manual_seed(normalized.seed)
        return self.pipeline.prepare_request(
            prompt=normalized.prompt,
            negative_prompt=normalized.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=normalized.guidance_scale,
            generator=generator,
        )

    def profile(self, state: ZImageDenoiseState) -> RequestProfile:
        return RequestProfile(
            total_steps=len(state.timesteps),
            completed_steps=state.step_index,
            image_tokens=state.image_tokens,
            attributes={
                "batch_size": state.actual_batch_size,
                "conditions": 2 if state.guidance_scale > 0 else 1,
            },
        )

    def denoise_step(self, state, *, lane_ranks) -> None:
        self.pipeline.denoise_step(state, lane_ranks=lane_ranks)

    def synchronize_state(self, state, *, step_index) -> None:
        self.pipeline.synchronize_state(state, step_index=step_index)

    def export_state(self, state: ZImageDenoiseState) -> TransferBundle:
        return TransferBundle(
            tensors={"latents": state.latents},
            step_index=state.step_index,
        )

    def finalize_gpu(self, state, *, timings):
        return self.pipeline.decode_request(state, timings=timings)

    def postprocess(self, host_output):
        return self.pipeline.image_processor.postprocess(
            host_output,
            output_type="pil",
        )

    def abort_request(self, request_id: str, state: Any | None) -> None:
        del request_id, state

    def close(self) -> None:
        self.pipeline.close()


@dataclass(frozen=True)
class ZImageExecutorFactory:
    model_path: str
    torch_dtype: torch.dtype | None = None
    local_files_only: bool = True
    default_width: int = 1024
    default_height: int = 1024
    default_num_steps: int = 50
    attention_mode: str = "agkv"
    ulysses_degree: int | None = None

    def build(self, context: ExecutorBuildContext) -> ZImageImageDecoderExecutor:
        world = context.world
        local_suffix = str(world.local_device).rsplit(":", 1)[-1]
        local_rank = int(local_suffix) if local_suffix.isdigit() else 0
        if dist.is_initialized():
            if (
                world.process_group is not None
                and world.process_group is not dist.group.WORLD
            ):
                raise NotImplementedError(
                    "the first embedded release supports an existing WORLD "
                    "process group, not an arbitrary subgroup"
                )
            if (
                dist.get_rank() != world.rank
                or dist.get_world_size() != world.world_size
            ):
                raise ValueError(
                    "initialized torch.distributed WORLD mismatches StageWorldSpec"
                )
        else:
            if world.process_group is not None:
                raise ValueError(
                    "process_group was supplied before torch.distributed init"
                )
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
        dtype = self.torch_dtype or (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        ulysses_degree = self.ulysses_degree
        if ulysses_degree is None:
            ulysses_degree = 2 if self.attention_mode == "usp" else 1
        parallel = EpeParallelContext.from_torchrun(
            allowed_widths=context.pool.allowed_lane_widths,
            owns_process_group=world.owns_process_group,
            ulysses_degree=ulysses_degree,
        )
        pipeline = EpeZImagePipeline.from_pretrained(
            self.model_path,
            parallel_context=parallel,
            epe_options={
                "policy": context.pool.policy,
                "switch_allowed_until_step": context.pool.switch_allowed_until_step,
                "pulse_steps": context.pool.pulse_steps,
                "balanced_k": context.pool.balanced_k,
                "starvation_ms": context.pool.starvation_ms,
                "deadline_guard_ms": context.pool.deadline_guard_ms,
                "max_active_requests": context.pool.max_inflight_requests,
                "online_calibration": context.pool.online_calibration,
                "attention_mode": self.attention_mode,
            },
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda", pipeline.parallel_context.local_rank))
        pipeline.set_progress_bar_config(disable=True)
        return ZImageImageDecoderExecutor(
            pipeline,
            default_width=self.default_width,
            default_height=self.default_height,
            default_num_steps=self.default_num_steps,
        )
