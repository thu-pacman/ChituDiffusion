from __future__ import annotations

import io
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist

from experiments.chitu_api.config import StageServiceConfig
from experiments.chitu_api.protocol import (
    AdmissionResponse,
    CancelResponse,
    HealthResponse,
    ImageGenerateRequest,
    RequestStatusResponse,
)

from .pipeline import EpeZImagePipeline, ZImageDenoiseState

logger = logging.getLogger(__name__)


@dataclass
class _RequestRecord:
    request: ImageGenerateRequest
    submitted_at: float
    status: str = "pending"
    first_scheduled_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    png: bytes | None = None


class EpeZImageServiceRuntime:
    """Instance-local EPE runtime with denoise-phase DP/CP hot switching."""

    def __init__(self, config: StageServiceConfig, pipeline: EpeZImagePipeline) -> None:
        self.config = config
        self.pipeline = pipeline
        self.parallel = pipeline.parallel_context
        self.rank = self.parallel.rank
        self.world_size = self.parallel.world_size
        self._records: dict[str, _RequestRecord] = {}
        self._pending: deque[str] = deque()
        self._states: dict[str, ZImageDenoiseState] = {}
        self._active_order: list[str] = []
        self._placements: dict[str, tuple[int, ...]] = {}
        self._lock = threading.RLock()
        self._accepting = self.rank == 0
        self._state = "running"
        self._fatal_error: str | None = None

    @classmethod
    def from_config(cls, config: StageServiceConfig) -> "EpeZImageServiceRuntime":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        pipeline = EpeZImagePipeline.from_pretrained(
            config.factory_args.model_path,
            allowed_lane_widths=config.parallelism.chitu_pool.allowed_lane_widths,
            torch_dtype=dtype,
            local_files_only=True,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda", pipeline.parallel_context.local_rank))
        pipeline.set_progress_bar_config(disable=True)
        return cls(config, pipeline)

    def submit(self, request: ImageGenerateRequest) -> AdmissionResponse:
        if self.rank != 0 or not self._accepting:
            raise RuntimeError("runtime is not accepting requests")
        request_id = request.request_id or uuid.uuid4().hex
        request = request.model_copy(update={"request_id": request_id})
        with self._lock:
            if request_id in self._records:
                raise KeyError(f"duplicate request_id: {request_id}")
            active = sum(
                record.status in {"pending", "running"}
                for record in self._records.values()
            )
            limit = self.config.parallelism.chitu_pool.max_pending_requests
            if active >= limit:
                raise OverflowError(f"request queue is full ({active}/{limit})")
            self._records[request_id] = _RequestRecord(
                request=request,
                submitted_at=time.time(),
            )
            self._pending.append(request_id)
        return AdmissionResponse(
            request_id=request_id,
            status_url=f"/v1/image-decode/{request_id}",
            image_url=f"/v1/image-decode/{request_id}/image",
        )

    def status(self, request_id: str) -> RequestStatusResponse | None:
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return None
            queue_delay_ms = (
                None
                if record.first_scheduled_at is None
                else (record.first_scheduled_at - record.submitted_at) * 1000.0
            )
            latency_ms = (
                None
                if record.completed_at is None
                else (record.completed_at - record.submitted_at) * 1000.0
            )
            return RequestStatusResponse(
                request_id=request_id,
                status=record.status,
                submitted_at=record.submitted_at,
                first_scheduled_at=record.first_scheduled_at,
                completed_at=record.completed_at,
                queue_delay_ms=queue_delay_ms,
                latency_ms=latency_ms,
                error=record.error,
                image_url=(
                    f"/v1/image-decode/{request_id}/image"
                    if record.status == "completed"
                    else None
                ),
            )

    def cancel(self, request_id: str) -> CancelResponse | None:
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return None
            if record.status != "pending":
                return CancelResponse(
                    request_id=request_id,
                    cancelled=False,
                    detail="runtime only supports pending-request cancellation",
                )
            record.status = "cancelled"
            record.completed_at = time.time()
            return CancelResponse(
                request_id=request_id,
                cancelled=True,
                detail="request cancelled before scheduling",
            )

    def image(self, request_id: str) -> bytes | None:
        with self._lock:
            record = self._records.get(request_id)
            return None if record is None else record.png

    def health(self) -> HealthResponse:
        with self._lock:
            statuses = [record.status for record in self._records.values()]
            return HealthResponse(
                state=self._state,
                rank=self.rank,
                world_size=self.world_size,
                queue_depth=sum(status == "pending" for status in statuses),
                inflight=sum(status == "running" for status in statuses),
                completed=sum(status == "completed" for status in statuses),
                fatal_error=self._fatal_error,
            )

    def has_work(self) -> bool:
        if self.rank != 0:
            return False
        with self._lock:
            has_pending = any(
                self._records[request_id].status == "pending"
                for request_id in self._pending
            )
            return has_pending or bool(self._active_order)

    def has_pending(self) -> bool:
        """Compatibility alias for the original service loop."""
        return self.has_work()

    def _take_requests(self) -> list[dict]:
        if self.rank != 0:
            return []
        requests: list[dict] = []
        pool = self.config.parallelism.chitu_pool
        if pool.policy in {"static_cp", "pure_sp"}:
            inflight_limit = 1
        else:
            inflight_limit = min(pool.max_inflight_requests, self.world_size)
        available_slots = max(0, inflight_limit - len(self._active_order))
        with self._lock:
            while self._pending and len(requests) < available_slots:
                request_id = self._pending.popleft()
                record = self._records[request_id]
                if record.status != "pending":
                    continue
                record.status = "running"
                record.first_scheduled_at = time.time()
                requests.append(record.request.model_dump())
        return requests

    def _canonical_lanes(self, width: int) -> list[tuple[int, ...]]:
        return [
            tuple(range(offset, offset + width))
            for offset in range(0, self.world_size, width)
        ]

    def _plan_phase(self) -> list[dict]:
        if self.rank != 0 or not self._active_order:
            return []
        pool = self.config.parallelism.chitu_pool
        if pool.policy in {"static_cp", "pure_sp"}:
            request_id = self._active_order[0]
            state = self._states[request_id]
            ranks = tuple(range(self.world_size))
            steps = min(
                pool.phase_max_steps,
                len(state.timesteps) - state.step_index,
            )
            plan = [
                {
                    "request_id": request_id,
                    "ranks": list(ranks),
                    "start_step": state.step_index,
                    "steps": steps,
                    "switched": False,
                }
            ]
            self._placements[request_id] = ranks
            logger.info(
                "static CP phase: %s:cp%d@%s step=%d+%d",
                request_id,
                self.world_size,
                list(ranks),
                state.step_index,
                steps,
            )
            return plan

        occupied: set[int] = set()
        planned: dict[str, tuple[int, ...]] = {}

        # Once the configured switch boundary is reached, keep the last lane.
        for request_id in self._active_order:
            state = self._states[request_id]
            placement = self._placements.get(request_id)
            if placement is not None and state.step_index >= pool.switch_allowed_until_step:
                planned[request_id] = placement
                occupied.update(placement)

        flexible = [
            request_id
            for request_id in self._active_order
            if request_id not in planned
        ]
        free_lanes: list[tuple[int, ...]] = []
        for width in reversed(self.parallel.allowed_widths):
            candidates = [
                lane
                for lane in self._canonical_lanes(width)
                if not occupied.intersection(lane)
            ]
            if len(candidates) >= len(flexible):
                free_lanes = candidates
                break
        if not free_lanes and flexible:
            free_lanes = [
                lane
                for lane in self._canonical_lanes(1)
                if not occupied.intersection(lane)
            ]
        for request_id, lane in zip(flexible, free_lanes):
            planned[request_id] = lane
            occupied.update(lane)

        plan = []
        for request_id in self._active_order:
            ranks = planned.get(request_id)
            if ranks is None:
                continue
            state = self._states[request_id]
            steps = min(
                pool.phase_max_steps,
                len(state.timesteps) - state.step_index,
            )
            previous = self._placements.get(request_id)
            plan.append(
                {
                    "request_id": request_id,
                    "ranks": list(ranks),
                    "start_step": state.step_index,
                    "steps": steps,
                    "switched": previous is not None and previous != ranks,
                }
            )
            self._placements[request_id] = ranks
        logger.info(
            "EPE phase layout: %s",
            ", ".join(
                f"{item['request_id']}:cp{len(item['ranks'])}@{item['ranks']}"
                f" step={item['start_step']}+{item['steps']}"
                f"{' switch' if item['switched'] else ''}"
                for item in plan
            ),
        )
        return plan

    def _prepare_request(self, raw: dict) -> None:
        request = ImageGenerateRequest.model_validate(raw)
        request_id = str(request.request_id)
        width = request.width or self.config.factory_args.default_width
        height = request.height or self.config.factory_args.default_height
        steps = request.num_steps or self.config.factory_args.num_steps
        device = (
            torch.device("cuda", self.parallel.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        generator = torch.Generator(device=device).manual_seed(request.seed)
        self._states[request_id] = self.pipeline.prepare_request(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
        )
        self._active_order.append(request_id)

    def _remove_active(self, request_id: str) -> None:
        self._states.pop(request_id, None)
        self._placements.pop(request_id, None)
        if request_id in self._active_order:
            self._active_order.remove(request_id)

    def _mark_failed(self, request_id: str, error: str) -> None:
        if self.rank != 0:
            return
        with self._lock:
            record = self._records[request_id]
            record.status = "failed"
            record.error = error
            record.completed_at = time.time()

    def _gather_objects(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def process_next(self) -> bool:
        payload = [self._take_requests() if self.rank == 0 else None]
        if self.world_size > 1:
            dist.broadcast_object_list(payload, src=0)
        new_requests = payload[0] or []

        local_prepare_errors: dict[str, str] = {}
        for raw in new_requests:
            request_id = str(raw["request_id"])
            try:
                self._prepare_request(raw)
            except Exception as exc:
                local_prepare_errors[request_id] = str(exc)
        prepare_errors = self._gather_objects(local_prepare_errors)
        failed_prepare = {
            request_id: error
            for errors in prepare_errors
            for request_id, error in (errors or {}).items()
        }
        for request_id, error in failed_prepare.items():
            self._remove_active(request_id)
            self._mark_failed(request_id, f"request preparation failed: {error}")

        payload = [self._plan_phase() if self.rank == 0 else None]
        if self.world_size > 1:
            dist.broadcast_object_list(payload, src=0)
        plan = payload[0] or []
        if not plan:
            return bool(new_requests)

        assignment = next(
            (item for item in plan if self.rank in item["ranks"]),
            None,
        )
        local_error = None
        if assignment is not None:
            request_id = assignment["request_id"]
            lane_ranks = tuple(int(rank) for rank in assignment["ranks"])
            try:
                state = self._states[request_id]
                for _ in range(int(assignment["steps"])):
                    self.pipeline.denoise_step(state, lane_ranks=lane_ranks)
            except Exception as exc:
                local_error = {"request_id": request_id, "error": str(exc)}

        errors = self._gather_objects(local_error)
        failed_run = {
            item["request_id"]: item["error"]
            for item in errors
            if item is not None
        }
        successful = [
            item for item in plan if item["request_id"] not in failed_run
        ]
        for item in successful:
            request_id = item["request_id"]
            state = self._states[request_id]
            if self.world_size > 1:
                dist.broadcast(state.latents, src=int(item["ranks"][0]))
            self.pipeline.synchronize_state(
                state,
                step_index=int(item["start_step"]) + int(item["steps"]),
            )

        for request_id, error in failed_run.items():
            self._mark_failed(request_id, f"denoise phase failed: {error}")

        finished = [
            item["request_id"]
            for item in successful
            if self._states[item["request_id"]].complete
        ]
        if self.rank == 0:
            for request_id in finished:
                try:
                    result = self.pipeline.finalize_request(
                        self._states[request_id], output_type="pil"
                    )
                    output = io.BytesIO()
                    result.images[0].save(output, format="PNG")
                    with self._lock:
                        record = self._records[request_id]
                        record.status = "completed"
                        record.png = output.getvalue()
                        record.completed_at = time.time()
                except Exception as exc:
                    with self._lock:
                        record = self._records[request_id]
                        record.status = "failed"
                        record.error = f"finalize failed: {exc}"
                        record.completed_at = time.time()

        for request_id in [*failed_run, *finished]:
            self._remove_active(request_id)
        return True

    def stop_admission(self) -> None:
        with self._lock:
            self._accepting = False
            if self._state != "failed":
                self._state = "stopping"

    def fail(self, error: BaseException) -> None:
        with self._lock:
            self._accepting = False
            self._state = "failed"
            self._fatal_error = str(error)

    def close(self) -> None:
        self.stop_admission()
        self.pipeline.close()
