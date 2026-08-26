from __future__ import annotations

import io
import json
import logging
import statistics
import threading
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from ..epe import (
    AsyncResultChannel,
    DistributedRankExchange,
    FullWorldLaneWorkerPool,
    LaneWorkItem,
    LaneWorkResult,
    PulseCoordinator,
    PulseLaneBroker,
    RankTimelineRecorder,
    SchedulableRequest,
    SingletonLaneWorkerPool,
)
from ..epe.contracts import (
    DiffusionBackendProtocol,
    ImageDecodeCompletion,
    LaneSnapshot,
    TerminalArtifact,
    normalize_terminal_artifact,
)
from ..epe.request import StructuredError
from ..models.zimage.executor import ZImageExecutorFactory
from .config import StageServiceConfig
from .protocol import (
    AdmissionResponse,
    CancelResponse,
    HealthResponse,
    ImageGenerateRequest,
    RequestStatusResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class _RequestRecord:
    request: object
    submitted_at: float
    status: str = "pending"
    first_scheduled_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    payload: bytes | None = None
    media_type: str = "image/png"
    output: object | None = None
    emitted: bool = False


class DiffusionServiceRuntime:
    """Model-neutral EPE runtime with denoise-phase DP/CP hot switching."""

    def __init__(self, config, executor: DiffusionBackendProtocol) -> None:
        self.config = config
        self.pool = (
            config.pool if hasattr(config, "pool") else config.parallelism.chitu_pool
        )
        self.executor = executor
        self.parallel = executor.parallel_context
        self.epe = executor.scheduling_module
        self.rank = self.parallel.rank
        self.world_size = self.parallel.world_size
        self.timeline = RankTimelineRecorder(
            rank=self.rank,
            output_root=config.output_root,
            enabled=config.record_timeline,
        )
        self._records: dict[str, _RequestRecord] = {}
        self._pending: deque[str] = deque()
        self._states: dict[str, object] = {}
        self._active_order: list[str] = []
        self._placements: dict[str, tuple[int, ...]] = {}
        self._scheduler_steps: dict[str, int] = {}
        self._lock = threading.RLock()
        self._postprocess_executor = (
            ThreadPoolExecutor(
                max_workers=config.postprocess_workers,
                thread_name_prefix="epe-image-postprocess",
            )
            if self.rank == 0
            else None
        )
        self._postprocess_futures: set[Future] = set()
        self._accepting = self.rank == 0
        self._state = "running"
        self._fatal_error: str | None = None
        self._warmup_report: dict | None = None
        scheduling_policy = self.epe.scheduling_policy
        self._pulse_coordinator = PulseCoordinator(
            world_size=self.world_size,
            policy=scheduling_policy,
            pulse_steps=self.pool.pulse_steps,
        )

    @classmethod
    def from_config(cls, config: StageServiceConfig) -> "DiffusionServiceRuntime":
        from ..epe.contracts import ExecutorBuildContext, StageWorldSpec
        from ..models.flux1.executor import Flux1ExecutorFactory
        from ..models.minimax_h3.executor import MiniMaxH3ExecutorFactory
        from ..models.qwen_image.executor import QwenImageExecutorFactory
        from ..models.wan.executor import WanExecutorFactory

        world = StageWorldSpec.from_torchrun(
            config.name,
            physical_device_ids=tuple(config.gpu),
        )
        common = {
            "model_path": config.factory_args.model_path,
            "default_width": config.factory_args.default_width,
            "default_height": config.factory_args.default_height,
            "default_num_steps": config.factory_args.num_steps,
            "attention_mode": config.factory_args.attention_mode,
            "ulysses_degree": config.factory_args.ulysses_degree,
            "parallel_vae": config.factory_args.parallel_vae,
            "vae_parallel_halo": config.factory_args.vae_parallel_halo,
        }
        if config.factory == "zimage":
            factory = ZImageExecutorFactory(
                **common,
                cfg_parallel=config.factory_args.cfg_parallel,
            )
        elif config.factory == "flux1":
            factory = Flux1ExecutorFactory(**common)
        elif config.factory == "qwen-image":
            factory = QwenImageExecutorFactory(
                **common,
                cfg_parallel=config.factory_args.cfg_parallel,
            )
        elif config.factory == "wan":
            factory = WanExecutorFactory(
                **common,
                default_num_frames=config.factory_args.num_frames,
                cfg_parallel=config.factory_args.cfg_parallel,
            )
        elif config.factory == "minimax-h3":
            factory = MiniMaxH3ExecutorFactory(
                model_path=config.factory_args.model_path,
                attention_backend=config.factory_args.attention_backend,
                attention_mode=config.factory_args.attention_mode,
                ulysses_degree=config.factory_args.ulysses_degree,
                flow_shift=config.factory_args.flow_shift,
                audio_flow_shift=config.factory_args.audio_flow_shift,
                default_width=config.factory_args.default_width,
                default_height=config.factory_args.default_height,
                default_latent_t=config.factory_args.latent_t,
                default_audio_t=config.factory_args.audio_t,
                default_audio_channels=config.factory_args.audio_channels,
                default_text_length=config.factory_args.text_length,
                default_num_steps=config.factory_args.num_steps,
                text_encoder_path=config.factory_args.text_encoder_path,
                tokenizer_path=config.factory_args.tokenizer_path,
                processor_path=config.factory_args.processor_path,
                video_vae_path=config.factory_args.video_vae_path,
                audio_vae_path=config.factory_args.audio_vae_path,
                enable_native_media=config.factory_args.enable_native_media,
                default_duration_s=config.factory_args.default_duration_s,
                default_fps=config.factory_args.default_fps,
                default_output_type=config.factory_args.default_output_type,
                ffmpeg_path=config.factory_args.ffmpeg_path,
                parallel_vae=config.factory_args.parallel_vae,
                vae_parallel_degree=config.factory_args.vae_parallel_degree,
                warmup_media_profiles=config.factory_args.warmup_media_profiles,
                warmup_burnin_steps=config.factory_args.warmup_burnin_steps,
                cost_profile_path=config.factory_args.cost_profile_path,
            )
        else:  # StageServiceConfig validates this before construction.
            raise ValueError(f"unsupported factory: {config.factory}")
        executor = factory.build(
            ExecutorBuildContext(
                world=world,
                pool=config.parallelism.chitu_pool,
                tensor_parallel_degree=config.parallelism.tp,
            )
        )
        return cls(config, executor)

    @property
    def warmup_report(self) -> dict | None:
        return self._warmup_report

    def warmup(self) -> dict:
        pool = self.pool
        report = self.executor.warmup(
            resolutions=pool.warmup_resolutions,
            steps=pool.warmup_steps,
        )
        transfer_rows = self._warmup_state_transfers(
            report.get("rows", ()),
            steps=pool.warmup_steps,
        )
        report["transfer_rows"] = transfer_rows
        initialize_transfer = getattr(self.epe, "initialize_transfer_cost_model", None)
        if initialize_transfer is not None:
            initialize_transfer(transfer_rows)
        self._warmup_report = report
        if self.rank == 0:
            output_dir = Path(self.config.output_root)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "epe_warmup.json"
            path.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            logger.info("Wrote EPE warmup report to %s", path)
        return report

    def _warmup_state_transfers(
        self,
        cost_rows: object,
        *,
        steps: int,
    ) -> list[dict[str, object]]:
        """Measure the exact state sizes used by supported request shapes."""
        if self.world_size <= 1 or not dist.is_initialized():
            return []
        if not isinstance(cost_rows, list):
            raise TypeError("warmup report rows must be a list")
        state_profiles = sorted(
            {
                (
                    int(row["state_bytes"]),
                    max(1, int(row.get("state_tensor_count", 1))),
                )
                for row in cost_rows
                if isinstance(row, dict) and int(row.get("state_bytes", 0)) > 0
            }
        )
        device = (
            torch.device("cuda", self.parallel.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        rows: list[dict[str, object]] = []
        barrier_kwargs = (
            {"device_ids": [self.parallel.local_rank]} if device.type == "cuda" else {}
        )
        for state_bytes, tensor_count in state_profiles:
            base_size, remainder = divmod(state_bytes, tensor_count)
            payloads = [
                torch.empty(
                    base_size + (1 if index < remainder else 0),
                    dtype=torch.uint8,
                    device=device,
                )
                for index in range(tensor_count)
            ]
            for source in range(self.world_size):
                for destination in range(self.world_size):
                    if source == destination:
                        continue
                    if (
                        source // self.parallel.cp_world_size
                        != destination // self.parallel.cp_world_size
                    ):
                        continue
                    samples_ms = []
                    for _ in range(steps):
                        dist.barrier(**barrier_kwargs)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        if self.rank == source:
                            for payload in payloads:
                                dist.send(payload, dst=destination)
                        elif self.rank == destination:
                            for payload in payloads:
                                dist.recv(payload, src=source)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        local_ms = (
                            (time.perf_counter() - started) * 1000.0
                            if self.rank in {source, destination}
                            else 0.0
                        )
                        critical = torch.tensor(
                            local_ms,
                            dtype=torch.float64,
                            device=device,
                        )
                        dist.all_reduce(critical, op=dist.ReduceOp.MAX)
                        if self.rank == 0:
                            samples_ms.append(float(critical.item()))
                    if self.rank == 0:
                        ordered_samples = sorted(samples_ms)
                        p95_index = 0.95 * (len(ordered_samples) - 1)
                        lower = int(p95_index)
                        upper = min(lower + 1, len(ordered_samples) - 1)
                        fraction = p95_index - lower
                        p95_ms = ordered_samples[lower] + fraction * (
                            ordered_samples[upper] - ordered_samples[lower]
                        )
                        rows.append(
                            {
                                "bytes": state_bytes,
                                "tensor_count": tensor_count,
                                "source": source,
                                "destination": destination,
                                "latency_ms": float(statistics.median(samples_ms)),
                                "p95_latency_ms": float(p95_ms),
                                "samples_ms": samples_ms,
                            }
                        )
            del payloads
        broadcast = [rows if self.rank == 0 else None]
        dist.broadcast_object_list(broadcast, src=0)
        assert broadcast[0] is not None
        return broadcast[0]

    def submit_request(self, request: object) -> str:
        if self.rank != 0 or not self._accepting:
            raise RuntimeError("runtime is not accepting requests")
        request = self.executor.normalize_request(request)
        request_id = self.executor.request_id(request)
        self.executor.validate_request(
            request,
            warmup_resolutions=self.pool.warmup_resolutions,
        )
        with self._lock:
            if request_id in self._records:
                raise KeyError(f"duplicate request_id: {request_id}")
            active = sum(
                record.status in {"pending", "running", "postprocessing"}
                for record in self._records.values()
            )
            limit = self.pool.max_pending_requests
            if active >= limit:
                raise OverflowError(f"request queue is full ({active}/{limit})")
            self._records[request_id] = _RequestRecord(
                request=request,
                submitted_at=time.time(),
            )
            self._pending.append(request_id)
            submitted_ns = time.time_ns()
            self.timeline.instant(
                "request_submit",
                unix_ns=submitted_ns,
                request_id=request_id,
                metadata={
                    "profile": self.executor.request_profile(
                        request,
                        completed_steps=0,
                    ).shape_key
                },
            )
        return request_id

    def submit(self, request: ImageGenerateRequest) -> AdmissionResponse:
        request_id = self.submit_request(request)
        return AdmissionResponse(
            request_id=request_id,
            status_url=f"/v1/image-decode/{request_id}",
            media_url=f"/v1/media/{request_id}",
            image_url=f"/v1/image-decode/{request_id}/image",
        )

    def poll_completions(self) -> list[ImageDecodeCompletion]:
        if self.rank != 0:
            return []
        with self._lock:
            completions = []
            for request_id, record in self._records.items():
                if record.status not in {"completed", "cancelled", "failed"}:
                    continue
                if record.emitted:
                    continue
                record.emitted = True
                completions.append(
                    ImageDecodeCompletion(
                        request_id=request_id,
                        status=record.status,
                        output=record.output,
                        error=(
                            None
                            if record.error is None
                            else StructuredError(
                                type="RequestExecutionError",
                                message=record.error,
                            )
                        ),
                        submitted_at=record.submitted_at,
                        started_at=record.first_scheduled_at,
                        completed_at=record.completed_at,
                    )
                )
            return completions

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
                media_url=(
                    f"/v1/media/{request_id}"
                    if record.status == "completed"
                    else None
                ),
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
            self.executor.abort_request(request_id, self._states.get(request_id))
            return CancelResponse(
                request_id=request_id,
                cancelled=True,
                detail="request cancelled before scheduling",
            )

    def image(self, request_id: str) -> bytes | None:
        with self._lock:
            record = self._records.get(request_id)
            return None if record is None else record.payload

    def media(self, request_id: str) -> bytes | None:
        return self.image(request_id)

    def media_type(self, request_id: str) -> str | None:
        with self._lock:
            record = self._records.get(request_id)
            return None if record is None else record.media_type

    def health(self) -> HealthResponse:
        with self._lock:
            statuses = [record.status for record in self._records.values()]
            return HealthResponse(
                state=self._state,
                rank=self.rank,
                world_size=self.world_size,
                queue_depth=sum(status == "pending" for status in statuses),
                inflight=sum(
                    status in {"running", "postprocessing"} for status in statuses
                ),
                completed=sum(status == "completed" for status in statuses),
                fatal_error=self._fatal_error,
            )

    def runtime_snapshot(self) -> tuple[tuple[LaneSnapshot, ...], int]:
        with self._lock:
            lanes = tuple(
                LaneSnapshot(
                    request_id=request_id,
                    ranks=tuple(ranks),
                    current_step=self._scheduler_steps.get(request_id, 0),
                )
                for request_id, ranks in sorted(self._placements.items())
                if self._records.get(request_id) is not None
                and self._records[request_id].status == "running"
            )
            return lanes, self._pulse_coordinator.last_plan_version

    def has_work(self) -> bool:
        if self.rank != 0:
            return False
        with self._lock:
            has_pending = any(
                self._records[request_id].status == "pending"
                for request_id in self._pending
            )
            return has_pending or bool(self._active_order)

    def _deadline_sort_key(self, request_id: str) -> tuple[bool, float, float]:
        record = self._records[request_id]
        deadline_ms = self._request_deadline_ms(record.request)
        return (
            deadline_ms is None,
            record.submitted_at + (deadline_ms or 0.0) / 1000.0,
            record.submitted_at,
        )

    def _request_deadline_ms(self, request) -> float | None:
        deadline_ms = self.executor.request_deadline_ms(request)
        if deadline_ms is not None:
            return deadline_ms
        return self.pool.default_deadline_ms

    @staticmethod
    def _request_priority(request: object) -> int:
        if isinstance(request, Mapping):
            return int(request.get("priority", 0))
        return int(getattr(request, "priority", 0))

    @property
    def uses_singleton_worker_pool(self) -> bool:
        return self.epe.policy == "static_dp"

    @property
    def uses_full_world_worker_pool(self) -> bool:
        return self.epe.policy == "static_cp"

    @property
    def uses_fixed_worker_pool(self) -> bool:
        return self.uses_singleton_worker_pool or self.uses_full_world_worker_pool

    @property
    def uses_elastic_worker_pool(self) -> bool:
        return self.epe.policy == "elastic"

    @property
    def uses_worker_pool(self) -> bool:
        return self.uses_fixed_worker_pool or self.uses_elastic_worker_pool

    def run_worker_pool(self, stop_requested: threading.Event) -> None:
        if self.uses_fixed_worker_pool:
            self.run_fixed_worker_pool(stop_requested)
            return
        if not self.uses_elastic_worker_pool:
            raise RuntimeError("unsupported worker-pool strategy")
        self.parallel.initialize_worker_control_group()
        self.parallel.initialize_worker_result_group()
        broker = (
            PulseLaneBroker(
                world_size=self.world_size,
                coordinator=self._pulse_coordinator,
                schedulable=self._elastic_schedulable,
                reserve=self._reserve_elastic_work,
                payload=self._elastic_payload,
                update_progress=self._update_elastic_progress,
                retire=self._retire_elastic_work,
                complete=self._complete_elastic_work,
                observe=self.epe.observe_schedulable,
                should_stop=stop_requested.is_set,
            )
            if self.rank == 0
            else None
        )
        exchange = DistributedRankExchange(
            rank=self.rank,
            world_size=self.world_size,
            control_group=self.parallel.worker_control_group(1),
            handler=None if broker is None else broker.exchange,
        )
        result_channel = AsyncResultChannel(
            rank=self.rank,
            world_size=self.world_size,
            control_group=(
                None if self.world_size == 1 else self.parallel.worker_result_group(1)
            ),
            complete=self._complete_elastic_work,
            on_transfer=self._record_result_transfer,
        )
        result_channel.start()
        try:
            exchange.run(
                lambda rank_exchange: self._elastic_worker_loop(
                    rank_exchange,
                    result_channel.submit,
                )
            )
        finally:
            result_channel.close()

    def run_fixed_worker_pool(self, stop_requested: threading.Event) -> None:
        if self.uses_singleton_worker_pool:
            self.run_singleton_worker_pool(stop_requested)
            return
        if not self.uses_full_world_worker_pool:
            raise RuntimeError("fixed worker pool requires a static strategy")
        self.parallel.initialize_worker_control_group()
        pool = FullWorldLaneWorkerPool(
            rank=self.rank,
            world_size=self.world_size,
            control_group=self.parallel.worker_control_group(1),
            claim=self._claim_full_world_work,
            execute=self._execute_full_world_work,
            complete=self._complete_lane_work,
            should_stop=stop_requested.is_set,
        )
        pool.run()

    def _elastic_schedulable(self, include_pending: bool) -> list[SchedulableRequest]:
        if self.rank != 0:
            raise RuntimeError("only rank 0 owns elastic scheduling state")
        with self._lock:
            running_ids = [
                request_id
                for request_id, record in self._records.items()
                if record.status == "running"
            ]
            available = max(
                0,
                self.pool.max_inflight_requests - len(running_ids),
            )
            pending_ids = [
                request_id
                for request_id in self._pending
                if self._records[request_id].status == "pending"
            ]
            pending_ids.sort(key=self._deadline_sort_key)
            visible_pending = (
                set(pending_ids) if include_pending else set(pending_ids[:available])
            )
            output = []
            for request_id, record in self._records.items():
                if record.status == "running":
                    if not include_pending:
                        continue
                    completed_steps = self._scheduler_steps.get(request_id, 0)
                    current_ranks = self._placements.get(request_id)
                elif include_pending and record.status == "pending":
                    if request_id not in visible_pending:
                        continue
                    completed_steps = 0
                    current_ranks = None
                elif not include_pending and record.status == "pending":
                    if request_id not in visible_pending:
                        continue
                    completed_steps = 0
                    current_ranks = None
                else:
                    continue
                request = record.request
                profile = self.executor.request_profile(
                    request,
                    completed_steps=completed_steps,
                )
                deadline_ms = self._request_deadline_ms(request)
                deadline_at = (
                    None
                    if deadline_ms is None
                    else record.submitted_at + deadline_ms / 1000.0
                )
                output.append(
                    SchedulableRequest(
                        request_id=request_id,
                        submitted_at=record.submitted_at,
                        deadline_at=deadline_at,
                        priority=self._request_priority(request),
                        profile=profile,
                        current_ranks=current_ranks,
                        admitted=record.status == "running",
                    )
                )
            return output

    def _reserve_elastic_work(self, request_id: str, ranks: tuple[int, ...]) -> None:
        with self._lock:
            record = self._records[request_id]
            if record.status == "pending":
                record.status = "running"
                record.first_scheduled_at = time.time()
                if request_id in self._pending:
                    self._pending.remove(request_id)
                if request_id not in self._active_order:
                    self._active_order.append(request_id)
                self._scheduler_steps[request_id] = 0
            self._placements[request_id] = tuple(ranks)

    def _elastic_payload(self, request_id: str) -> dict:
        with self._lock:
            return self.executor.serialize_request(self._records[request_id].request)

    def _update_elastic_progress(
        self, request_id: str, current_step: int, ranks: tuple[int, ...]
    ) -> None:
        with self._lock:
            self._scheduler_steps[request_id] = int(current_step)
            self._placements[request_id] = tuple(ranks)

    def _complete_elastic_work(self, result: LaneWorkResult) -> None:
        self._complete_lane_work(result)
        with self._lock:
            self._scheduler_steps.pop(result.request_id, None)
            self._placements.pop(result.request_id, None)
            if result.request_id in self._active_order:
                self._active_order.remove(result.request_id)

    def _retire_elastic_work(self, request_id: str) -> None:
        with self._lock:
            record = self._records[request_id]
            if record.status in {"pending", "running"}:
                record.status = "gpu_completed"
            self._scheduler_steps.pop(request_id, None)
            self._placements.pop(request_id, None)
            if request_id in self._active_order:
                self._active_order.remove(request_id)

    def _elastic_worker_loop(self, exchange, publish_result) -> None:
        pulse_sync = 0
        previous_report = None
        previous_epoch = -1
        while True:
            pulse_wait_started_ns = time.time_ns()
            pulse_report = (
                None
                if previous_report is None
                else {
                    key: value
                    for key, value in previous_report.items()
                    if key != "result"
                }
            )
            response = exchange(
                {
                    "kind": "pulse_report",
                    "epoch": previous_epoch,
                    "sync": pulse_sync,
                    "report": pulse_report,
                }
            )
            pulse_wait_ended_ns = time.time_ns()
            if response.get("action") == "pulse":
                self.timeline.record(
                    "pulse_wait",
                    start_unix_ns=pulse_wait_started_ns,
                    end_unix_ns=pulse_wait_ended_ns,
                    request_id=(
                        None
                        if previous_report is None
                        else previous_report.get("request_id")
                    ),
                    metadata={
                        "reported_epoch": previous_epoch,
                        "next_epoch": int(response["epoch"]),
                        **response.get("planner", {}),
                    },
                )
            pending_result = (
                None if previous_report is None else previous_report.get("result")
            )
            for request_id in response.get("retired_request_ids", ()):
                self._states.pop(str(request_id), None)
            previous_report = None
            previous_epoch = -1
            action = response.get("action")
            if action == "stop":
                if pending_result is not None:
                    publish_result(pending_result)
                return
            if action == "idle":
                if pending_result is not None:
                    publish_result(pending_result)
                time.sleep(0.01)
                pulse_sync += 1
                continue
            if action != "pulse":
                raise RuntimeError("elastic worker received an invalid pulse command")
            if self.rank == 0:
                logger.info(
                    "elastic pulse %d layout: %s",
                    int(response["epoch"]),
                    ", ".join(
                        f"{item['request_id']}:cp"
                        f"{len({int(rank) % self.parallel.cp_world_size for rank in item['ranks']})}@"
                        f"{item['ranks']} step={item['current_step']}+{item['steps']}"
                        for item in response["assignments"]
                    ),
                )
            dispatch_started_ns = time.time_ns()
            self._apply_elastic_dispatch(response)
            self.timeline.record(
                "elastic_dispatch",
                start_unix_ns=dispatch_started_ns,
                end_unix_ns=time.time_ns(),
                metadata={
                    "epoch": int(response["epoch"]),
                    "assignment_count": len(response["assignments"]),
                },
            )
            if pending_result is not None:
                publish_result(pending_result)
            lease = next(
                item for item in response["leases"] if self.rank in item["ranks"]
            )
            previous_report = self._run_elastic_lease(response, lease, exchange)
            previous_epoch = int(response["epoch"])
            pulse_sync += 1

    def _record_result_transfer(
        self,
        result: LaneWorkResult,
        started_ns: int,
        ended_ns: int,
    ) -> None:
        lane_ranks = tuple(int(rank) for rank in result.metadata["lane_ranks"])
        self.timeline.record(
            "result_transfer",
            start_unix_ns=started_ns,
            end_unix_ns=ended_ns,
            request_id=result.request_id,
            lane_ranks=lane_ranks,
            resource="cpu",
        )

    def _apply_elastic_dispatch(self, dispatch: dict) -> None:
        assignments = sorted(
            dispatch["assignments"], key=lambda item: item["request_id"]
        )
        for assignment in assignments:
            ranks = tuple(int(rank) for rank in assignment["ranks"])
            if self.rank not in ranks:
                continue
            request_id = assignment["request_id"]
            if request_id not in self._states:
                request = self.executor.deserialize_request(assignment["payload"])
                allocate = getattr(self.executor, "allocate_request", None)
                if assignment.get("old_ranks") is not None and callable(allocate):
                    execution_ranks = self.parallel.plane_lane(ranks)
                    with self.parallel.activate(execution_ranks):
                        self._states[request_id] = allocate(request)
                else:
                    self._states[request_id] = self._prepare_request_state(
                        request,
                        lane_ranks=ranks,
                        strategy="elastic",
                    )

        transfers = []
        for assignment in assignments:
            old_ranks = assignment.get("old_ranks")
            if old_ranks is None:
                continue
            for destination in assignment["ranks"]:
                destination = int(destination)
                if destination not in old_ranks:
                    destination_plane = (
                        destination // self.parallel.cp_world_size
                    )
                    source = next(
                        int(rank)
                        for rank in old_ranks
                        if int(rank) // self.parallel.cp_world_size
                        == destination_plane
                    )
                    transfers.append((assignment, source, destination))
        for assignment, source, destination in transfers:
            request_id = assignment["request_id"]
            state = self._states.get(request_id)
            transfer_started_ns = time.time_ns()
            transferred_bytes = 0
            if self.rank == source:
                if state is None:
                    raise RuntimeError(f"rank {source} lost state for {request_id}")
                bundle = self.executor.export_state(state)
                for tensor in bundle.tensors.values():
                    value = tensor.contiguous()
                    transferred_bytes += value.numel() * value.element_size()
                    dist.send(value, dst=destination)
            elif self.rank == destination:
                if state is None:
                    raise RuntimeError(
                        f"rank {destination} did not prepare state for {request_id}"
                    )
                bundle = self.executor.export_state(state)
                for tensor in bundle.tensors.values():
                    transferred_bytes += tensor.numel() * tensor.element_size()
                    dist.recv(tensor, src=source)
            if self.rank in {source, destination}:
                self.timeline.record(
                    "state_transfer",
                    start_unix_ns=transfer_started_ns,
                    end_unix_ns=time.time_ns(),
                    request_id=request_id,
                    lane_ranks=assignment["ranks"],
                    metadata={
                        "source": source,
                        "destination": destination,
                        "bytes": transferred_bytes,
                    },
                )

        for assignment in assignments:
            ranks = tuple(int(rank) for rank in assignment["ranks"])
            if self.rank in ranks:
                self.executor.synchronize_state(
                    self._states[assignment["request_id"]],
                    step_index=int(assignment["current_step"]),
                )

    def _run_elastic_lease(self, dispatch: dict, lease: dict, exchange) -> dict:
        epoch = int(dispatch["epoch"])
        ranks = tuple(int(rank) for rank in lease["ranks"])
        execution_ranks = self.parallel.plane_lane(ranks)
        request_id = lease.get("request_id")
        if request_id is None:
            return {
                "request_id": None,
                "current_step": None,
                "state_owner_rank": None,
                "completed_request_ids": [],
                "observed_step_ms": None,
            }
        assignment = next(
            item for item in dispatch["assignments"] if item["request_id"] == request_id
        )
        command = {
            "action": "run",
            "request_id": request_id,
            "steps": int(assignment["steps"]),
            "predicted_step_ms": float(assignment["predicted_step_ms"]),
            "payload": assignment["payload"],
        }
        completed_ids = []
        observations = []
        lane_sync = 0
        current_id = request_id
        current_step = int(assignment["current_step"])
        error = None
        while command.get("action") == "run":
            current_id = str(command["request_id"])
            if current_id not in self._states:
                request = self.executor.deserialize_request(command["payload"])
                self._states[current_id] = self._prepare_request_state(
                    request,
                    lane_ranks=execution_ranks,
                    strategy="elastic",
                )
            state = self._states[current_id]
            state_profile = self.executor.profile(state)
            steps = min(int(command["steps"]), state_profile.remaining_steps)
            local_error = None
            result = None
            measured_step_ms = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.parallel.local_rank)
                denoise_started_ns = time.time_ns()
                started_at = denoise_started_ns / 1_000_000_000.0
                started = time.perf_counter()
                start_step = state_profile.completed_steps
                for _ in range(steps):
                    self.executor.denoise_step(
                        state, lane_ranks=execution_ranks
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.parallel.local_rank)
                denoise_ended_ns = time.time_ns()
                measured_step_ms = (time.perf_counter() - started) * 1000.0 / steps
                self.timeline.record(
                    "denoise",
                    start_unix_ns=denoise_started_ns,
                    end_unix_ns=denoise_ended_ns,
                    request_id=current_id,
                    lane_ranks=ranks,
                    metadata={
                        "strategy": "elastic",
                        "epoch": epoch,
                        "lease_sync": lane_sync,
                        "step_begin": start_step,
                        "step_end": self.executor.profile(state).completed_steps,
                    },
                )
                observations.append(measured_step_ms)
                state_profile = self.executor.profile(state)
                current_step = state_profile.completed_steps
                if state_profile.remaining_steps == 0:
                    finalized = self._decode_to_host(
                        state,
                        request_id=current_id,
                        lane_ranks=execution_ranks,
                        result_leader_rank=ranks[0],
                    )
                    if self.rank == ranks[0]:
                        if finalized is None:
                            raise RuntimeError("lane leader did not receive VAE output")
                        host_artifact, finalize_profile = finalized
                        completed_at = time.time()
                        result = LaneWorkResult(
                            request_id=current_id,
                            output=host_artifact,
                            metadata={
                                "strategy": "elastic",
                                "lane_rank": ranks[0],
                                "lane_ranks": list(ranks),
                                "started_at": started_at,
                                "denoise_started_at": started_at,
                                "completed_at": completed_at,
                                "steps": steps,
                                "image_tokens": state_profile.image_tokens,
                                "batch_size": state_profile.attributes["batch_size"],
                                "cfg_conditions": state_profile.attributes[
                                    "conditions"
                                ],
                                "predicted_step_ms": float(
                                    command["predicted_step_ms"]
                                ),
                                "measured_step_ms": measured_step_ms,
                                "finalize_profile": finalize_profile,
                            },
                        )
            except Exception as exc:
                local_error = str(exc)
                error = local_error
            if self.executor.profile(state).remaining_steps == 0:
                finished_id = current_id
                completed_ids.append(finished_id)
                self._states.pop(finished_id, None)
                return {
                    "request_id": None,
                    "current_step": None,
                    "state_owner_rank": None,
                    "completed_request_ids": completed_ids,
                    "observed_step_ms": (
                        None
                        if not observations
                        else sum(observations) / len(observations)
                    ),
                    "error": error,
                    "finished_request_id": finished_id,
                    "result": result,
                }
            control_started_ns = time.time_ns()
            response = exchange(
                {
                    "kind": "lane_progress",
                    "epoch": epoch,
                    "sync": lane_sync,
                    "ranks": list(ranks),
                    "request_id": current_id,
                    "current_step": current_step,
                    "result": result,
                    "observed_step_ms": measured_step_ms,
                    "error": local_error,
                }
            )
            self.timeline.record(
                "control_wait",
                start_unix_ns=control_started_ns,
                end_unix_ns=time.time_ns(),
                request_id=current_id,
                lane_ranks=ranks,
                metadata={"epoch": epoch, "lease_sync": lane_sync},
            )
            if self.executor.profile(state).remaining_steps == 0:
                completed_ids.append(current_id)
                self._states.pop(current_id, None)
                current_id = None
                current_step = None
            elif local_error is not None:
                self._states.pop(current_id, None)
                current_id = None
                current_step = None
            lane_sync += 1
            command = response
        return {
            "request_id": current_id,
            "current_step": current_step,
            "state_owner_rank": None if current_id is None else ranks[0],
            "completed_request_ids": completed_ids,
            "observed_step_ms": (
                None if not observations else sum(observations) / len(observations)
            ),
            "error": error,
        }

    def run_singleton_worker_pool(self, stop_requested: threading.Event) -> None:
        """Run static-DP as independent producer-consumer singleton lanes."""
        if not self.uses_singleton_worker_pool:
            raise RuntimeError("singleton worker pool requires static_dp strategy")
        self.parallel.initialize_worker_control_group()
        pool = SingletonLaneWorkerPool(
            rank=self.rank,
            world_size=self.world_size,
            control_group=self.parallel.worker_control_group,
            claim=self._claim_singleton_work,
            execute=self._execute_singleton_work,
            complete=self._complete_singleton_work,
            should_stop=stop_requested.is_set,
        )
        pool.run()

    def _claim_singleton_work(self, lane_rank: int) -> LaneWorkItem | None:
        return self._claim_lane_work((lane_rank,), "static_dp")

    def _claim_full_world_work(self, _leader_rank: int) -> LaneWorkItem | None:
        return self._claim_lane_work(tuple(range(self.world_size)), "static_cp")

    def _claim_lane_work(
        self, lane_ranks: tuple[int, ...], strategy: str
    ) -> LaneWorkItem | None:
        if self.rank != 0:
            raise RuntimeError("only rank 0 may claim admitted work")
        with self._lock:
            inflight = sum(
                record.status == "running" for record in self._records.values()
            )
            limit = min(
                self.world_size // len(lane_ranks),
                self.pool.max_inflight_requests,
            )
            if inflight >= limit:
                return None
            candidates = [
                request_id
                for request_id in self._pending
                if self._records[request_id].status == "pending"
            ]
            if not candidates:
                return None
            candidates.sort(key=self._deadline_sort_key)
            request_id = candidates[0]
            self._pending.remove(request_id)
            record = self._records[request_id]
            record.status = "running"
            record.first_scheduled_at = time.time()
            self._placements[request_id] = tuple(lane_ranks)
            logger.info(
                "%s lane %s claimed %s queue_ms=%.2f",
                strategy,
                list(lane_ranks),
                request_id,
                (record.first_scheduled_at - record.submitted_at) * 1000.0,
            )
            return LaneWorkItem(
                request_id=request_id,
                payload=self.executor.serialize_request(record.request),
                metadata={
                    "strategy": strategy,
                    "lane_rank": lane_ranks[0],
                    "lane_ranks": list(lane_ranks),
                    "submitted_at": record.submitted_at,
                    "scheduled_at": record.first_scheduled_at,
                },
            )

    def _execute_singleton_work(
        self, item: LaneWorkItem, lane_rank: int
    ) -> LaneWorkResult:
        return self._execute_lane_work(item, (lane_rank,))

    def _decode_to_host(
        self,
        state: object,
        *,
        request_id: str,
        lane_ranks: tuple[int, ...],
        result_leader_rank: int | None = None,
    ) -> tuple[TerminalArtifact, dict[str, object]] | None:
        timings: dict[str, object] = {}
        finalize_started = time.perf_counter()
        decoded = self.executor.finalize_gpu(
            state,
            lane_ranks=lane_ranks,
            timings=timings,
        )
        self.timeline.record(
            "vae",
            start_unix_ns=int(timings["vae_start_unix_ns"]),
            end_unix_ns=int(timings["vae_end_unix_ns"]),
            request_id=request_id,
            lane_ranks=lane_ranks,
        )
        leader = lane_ranks[0] if result_leader_rank is None else result_leader_rank
        if self.rank != leader:
            return None
        if decoded is None:
            raise RuntimeError("lane leader did not receive VAE output")
        artifact = normalize_terminal_artifact(decoded)
        assert artifact is not None
        d2h_started_ns = time.time_ns()
        d2h_started = time.perf_counter()
        host_artifact = TerminalArtifact(
            tensors={
                name: tensor.to(device="cpu")
                for name, tensor in artifact.tensors.items()
            },
            metadata=artifact.metadata,
        )
        d2h_ended_ns = time.time_ns()
        timings["d2h_ms"] = (time.perf_counter() - d2h_started) * 1000.0
        timings["finalize_total_ms"] = (time.perf_counter() - finalize_started) * 1000.0
        self.timeline.record(
            "d2h",
            start_unix_ns=d2h_started_ns,
            end_unix_ns=d2h_ended_ns,
            request_id=request_id,
            lane_ranks=lane_ranks,
            metadata={
                "tensor_names": list(host_artifact.tensors),
                "bytes": sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in host_artifact.tensors.values()
                ),
            },
        )
        return host_artifact, timings

    def _postprocess_to_png(
        self,
        host_artifact: TerminalArtifact,
        *,
        request_id: str,
        lane_ranks: tuple[int, ...],
    ) -> tuple[bytes | None, object, dict[str, object]]:
        timings: dict[str, object] = {}
        postprocess_started_ns = time.time_ns()
        postprocess_started = time.perf_counter()
        postprocess_artifact = getattr(
            self.executor, "postprocess_artifact", None
        )
        if postprocess_artifact is not None:
            images = postprocess_artifact(host_artifact)
        else:
            if len(host_artifact.tensors) != 1:
                raise TypeError(
                    "executor must implement postprocess_artifact for "
                    "multiple terminal tensors"
                )
            images = self.executor.postprocess(
                next(iter(host_artifact.tensors.values()))
            )
        postprocess_ended_ns = time.time_ns()
        timings["postprocess_ms"] = (time.perf_counter() - postprocess_started) * 1000.0
        png_started_ns = time.time_ns()
        png_started = time.perf_counter()
        package_output = getattr(self.executor, "package_postprocessed_output", None)
        if package_output is None:
            output = io.BytesIO()
            images[0].save(output, format="PNG")
            output_bytes = output.getvalue()
            decoded_output = images[0]
        else:
            output_bytes, decoded_output = package_output(images)
        png_ended_ns = time.time_ns()
        timings["png_encode_ms"] = (time.perf_counter() - png_started) * 1000.0
        timings["png_bytes"] = 0 if output_bytes is None else len(output_bytes)
        self.timeline.record(
            "cpu_postprocess",
            start_unix_ns=postprocess_started_ns,
            end_unix_ns=postprocess_ended_ns,
            request_id=request_id,
            lane_ranks=lane_ranks,
            resource="cpu",
        )
        self.timeline.record(
            "cpu_png",
            start_unix_ns=png_started_ns,
            end_unix_ns=png_ended_ns,
            request_id=request_id,
            lane_ranks=lane_ranks,
            resource="cpu",
            metadata={"bytes": timings["png_bytes"]},
        )
        return output_bytes, decoded_output, timings

    def _execute_full_world_work(
        self, item: LaneWorkItem, lane_ranks: tuple[int, ...]
    ) -> LaneWorkResult:
        return self._execute_lane_work(item, lane_ranks)

    def _execute_lane_work(
        self, item: LaneWorkItem, lane_ranks: tuple[int, ...]
    ) -> LaneWorkResult:
        execution_ranks = self.parallel.plane_lane(lane_ranks)
        request = self.executor.deserialize_request(item.payload)
        strategy = str(item.metadata["strategy"])
        lane_rank = lane_ranks[0]
        started_at = time.time()
        try:
            state = self._prepare_request_state(
                request,
                lane_ranks=execution_ranks,
                strategy=strategy,
            )
            state_profile = self.executor.profile(state)
            steps = state_profile.remaining_steps
            conditions = int(state_profile.attributes["conditions"])
            predicted_step_ms = self.epe.cost_model.predict_step_ms(
                sequence_length=state_profile.image_tokens,
                width=len(execution_ranks),
                batch_size=int(state_profile.attributes["batch_size"]),
                conditions=conditions,
            )
            logger.info(
                "%s lane run: request=%s ranks=%s start_unix_ms=%.3f "
                "steps=%d predicted=%.2fms/step",
                strategy,
                item.request_id,
                list(lane_ranks),
                started_at * 1000.0,
                steps,
                predicted_step_ms,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.parallel.local_rank)
            denoise_started_ns = time.time_ns()
            denoise_started_at = denoise_started_ns / 1_000_000_000.0
            denoise_started = time.perf_counter()
            start_step = state_profile.completed_steps
            while self.executor.profile(state).remaining_steps > 0:
                self.executor.denoise_step(
                    state, lane_ranks=execution_ranks
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.parallel.local_rank)
            denoise_ended_ns = time.time_ns()
            denoise_ms = (time.perf_counter() - denoise_started) * 1000.0
            self.timeline.record(
                "denoise",
                start_unix_ns=denoise_started_ns,
                end_unix_ns=denoise_ended_ns,
                request_id=item.request_id,
                lane_ranks=lane_ranks,
                metadata={
                    "strategy": strategy,
                    "step_begin": start_step,
                    "step_end": self.executor.profile(state).completed_steps,
                },
            )
            measured_step_ms = denoise_ms / steps
            finalized = self._decode_to_host(
                state,
                request_id=item.request_id,
                lane_ranks=execution_ranks,
                result_leader_rank=lane_ranks[0],
            )
            host_artifact = None
            finalize_profile: dict[str, object] = {}
            if self.rank == lane_rank:
                if finalized is None:
                    raise RuntimeError("lane leader did not receive VAE output")
                host_artifact, finalize_profile = finalized
            completed_at = time.time()
            return LaneWorkResult(
                request_id=item.request_id,
                output=host_artifact,
                metadata={
                    **item.metadata,
                    "lane_rank": lane_rank,
                    "lane_ranks": list(lane_ranks),
                    "started_at": started_at,
                    "denoise_started_at": denoise_started_at,
                    "completed_at": completed_at,
                    "steps": steps,
                    "image_tokens": state_profile.image_tokens,
                    "batch_size": state_profile.attributes["batch_size"],
                    "cfg_conditions": conditions,
                    "predicted_step_ms": predicted_step_ms,
                    "measured_step_ms": measured_step_ms,
                    "denoise_ms": denoise_ms,
                    "finalize_profile": finalize_profile,
                },
            )
        except Exception as exc:
            return LaneWorkResult(
                request_id=item.request_id,
                error=str(exc),
                metadata={
                    **item.metadata,
                    "lane_rank": lane_rank,
                    "started_at": started_at,
                    "completed_at": time.time(),
                },
            )

    def _complete_singleton_work(self, result: LaneWorkResult) -> None:
        self._complete_lane_work(result)

    def _submit_postprocess(self, result: LaneWorkResult) -> None:
        if self._postprocess_executor is None:
            raise RuntimeError("postprocess executor is only available on rank 0")
        artifact = normalize_terminal_artifact(result.output)
        if artifact is None or any(
            tensor.device.type != "cpu" for tensor in artifact.tensors.values()
        ):
            raise TypeError(
                "lane result must contain terminal artifact CPU tensors"
            )
        lane_ranks = tuple(int(rank) for rank in result.metadata["lane_ranks"])
        future = self._postprocess_executor.submit(
            self._postprocess_to_png,
            artifact,
            request_id=result.request_id,
            lane_ranks=lane_ranks,
        )
        with self._lock:
            self._postprocess_futures.add(future)
        future.add_done_callback(
            lambda completed: self._publish_postprocess(result, completed)
        )

    def _publish_postprocess(
        self,
        result: LaneWorkResult,
        future: Future,
    ) -> None:
        strategy = str(result.metadata.get("strategy", self.epe.policy))
        completed_at = time.time()
        try:
            output_bytes, decoded_output, timings = future.result()
            with self._lock:
                record = self._records[result.request_id]
                record.status = "completed"
                record.payload = output_bytes
                media_type_for_output = getattr(
                    self.executor, "media_type_for_output", None
                )
                record.media_type = (
                    str(media_type_for_output(output_bytes, decoded_output))
                    if callable(media_type_for_output)
                    else str(
                        getattr(self.executor, "output_media_type", "image/png")
                    )
                )
                record.output = decoded_output
                record.completed_at = completed_at
            self.timeline.instant(
                "request_complete",
                unix_ns=int(completed_at * 1_000_000_000),
                request_id=result.request_id,
                metadata={"strategy": strategy},
            )
            logger.info(
                "%s image complete: request=%s postprocess=%.2fms "
                "package=%.2fms encoded_bytes=%d",
                strategy,
                result.request_id,
                float(timings["postprocess_ms"]),
                float(timings["png_encode_ms"]),
                int(timings["png_bytes"]),
            )
        except BaseException as exc:
            with self._lock:
                record = self._records[result.request_id]
                record.status = "failed"
                record.error = f"{strategy} postprocess failed: {exc}"
                record.completed_at = completed_at
            self.timeline.instant(
                "request_failed",
                unix_ns=int(completed_at * 1_000_000_000),
                request_id=result.request_id,
                metadata={"strategy": strategy, "stage": "postprocess"},
            )
            logger.exception(
                "%s postprocess failed for request %s",
                strategy,
                result.request_id,
            )
        finally:
            with self._lock:
                self._postprocess_futures.discard(future)

    def _complete_lane_work(self, result: LaneWorkResult) -> None:
        if self.rank != 0:
            raise RuntimeError("only rank 0 may publish lane results")
        metadata = result.metadata
        strategy = str(metadata.get("strategy", self.epe.policy))
        gpu_completed_at = float(metadata.get("completed_at", time.time()))
        with self._lock:
            record = self._records[result.request_id]
            self._placements.pop(result.request_id, None)
            self._scheduler_steps.pop(result.request_id, None)
            if result.error is not None:
                record.status = "failed"
                record.error = f"{strategy} lane failed: {result.error}"
                record.completed_at = gpu_completed_at
            else:
                record.status = "postprocessing"
        if result.error is not None:
            self.timeline.instant(
                "request_failed",
                unix_ns=int(gpu_completed_at * 1_000_000_000),
                request_id=result.request_id,
                metadata={"strategy": strategy, "stage": "gpu"},
            )
            logger.error(
                "%s lane %s failed request %s: %s",
                strategy,
                metadata.get("lane_ranks", [metadata.get("lane_rank")]),
                result.request_id,
                result.error,
            )
            return
        self.timeline.instant(
            "gpu_complete",
            unix_ns=int(gpu_completed_at * 1_000_000_000),
            request_id=result.request_id,
            metadata={"strategy": strategy},
        )
        try:
            self._submit_postprocess(result)
        except BaseException as exc:
            failed_at = time.time()
            with self._lock:
                record = self._records[result.request_id]
                record.status = "failed"
                record.error = f"{strategy} postprocess submission failed: {exc}"
                record.completed_at = failed_at
            self.timeline.instant(
                "request_failed",
                unix_ns=int(failed_at * 1_000_000_000),
                request_id=result.request_id,
                metadata={"strategy": strategy, "stage": "postprocess_submit"},
            )
            logger.exception(
                "%s could not submit postprocess for request %s",
                strategy,
                result.request_id,
            )
            return
        if strategy != "elastic":
            assignment = {
                "ranks": [int(rank) for rank in metadata["lane_ranks"]],
                "image_tokens": int(metadata["image_tokens"]),
                "batch_size": int(metadata["batch_size"]),
                "cfg_conditions": int(metadata["cfg_conditions"]),
            }
            with self._lock:
                self.epe.observe_assignment(
                    assignment,
                    float(metadata["measured_step_ms"]),
                )
        finalize = metadata.get("finalize_profile", {})
        if not isinstance(finalize, dict):
            finalize = {}
        finalize_log_values = (
            float(finalize.get("finalize_total_ms", 0.0)),
            float(finalize.get("vae_decode_ms", 0.0)),
            float(finalize.get("d2h_ms", 0.0)),
            float(finalize.get("cuda_allocated_before_mb", 0.0)),
            float(finalize.get("cuda_peak_allocated_mb", 0.0)),
            bool(finalize.get("latent_contiguous", False)),
        )
        if strategy == "static_dp":
            logger.info(
                "static_dp lane complete: request=%s rank=%d start_unix_ms=%.3f "
                "denoise_start_unix_ms=%.3f end_unix_ms=%.3f steps=%d "
                "predicted=%.2fms/step measured=%.2fms/step "
                "gpu_finalize=%.2fms vae=%.2fms d2h=%.2fms "
                "cuda_alloc_before=%.1fMiB "
                "cuda_peak=%.1fMiB latent_contiguous=%s",
                result.request_id,
                int(metadata["lane_rank"]),
                float(metadata["started_at"]) * 1000.0,
                float(metadata["denoise_started_at"]) * 1000.0,
                gpu_completed_at * 1000.0,
                int(metadata["steps"]),
                float(metadata["predicted_step_ms"]),
                float(metadata["measured_step_ms"]),
                *finalize_log_values,
            )
        elif strategy == "static_cp":
            logger.info(
                "static_cp lane complete: request=%s ranks=%s "
                "start_unix_ms=%.3f denoise_start_unix_ms=%.3f "
                "end_unix_ms=%.3f steps=%d predicted=%.2fms/step "
                "measured=%.2fms/step gpu_finalize=%.2fms vae=%.2fms "
                "d2h=%.2fms "
                "cuda_alloc_before=%.1fMiB cuda_peak=%.1fMiB "
                "latent_contiguous=%s",
                result.request_id,
                metadata["lane_ranks"],
                float(metadata["started_at"]) * 1000.0,
                float(metadata["denoise_started_at"]) * 1000.0,
                gpu_completed_at * 1000.0,
                int(metadata["steps"]),
                float(metadata["predicted_step_ms"]),
                float(metadata["measured_step_ms"]),
                *finalize_log_values,
            )
        else:
            logger.info(
                "elastic lane complete: request=%s ranks=%s end_unix_ms=%.3f "
                "step=%d measured=%.2fms/step gpu_finalize=%.2fms "
                "vae=%.2fms d2h=%.2fms "
                "cuda_alloc_before=%.1fMiB cuda_peak=%.1fMiB "
                "latent_contiguous=%s",
                result.request_id,
                metadata["lane_ranks"],
                gpu_completed_at * 1000.0,
                int(metadata["steps"]),
                float(metadata["measured_step_ms"]),
                *finalize_log_values,
            )

    def _prepare_request_state(
        self,
        request: object,
        *,
        lane_ranks: tuple[int, ...],
        strategy: str,
    ) -> object:
        started_ns = time.time_ns()
        execution_ranks = self.parallel.plane_lane(lane_ranks)
        try:
            with self.parallel.activate(execution_ranks):
                state = self.executor.prepare_request(request)
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.parallel.local_rank)
            return state
        finally:
            ended_ns = time.time_ns()
            self.timeline.record(
                "prepare",
                start_unix_ns=started_ns,
                end_unix_ns=ended_ns,
                request_id=self.executor.request_id(request),
                lane_ranks=lane_ranks,
                metadata={"strategy": strategy},
            )

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
        try:
            if self._postprocess_executor is not None:
                self._postprocess_executor.shutdown(wait=True, cancel_futures=False)
            self.timeline.close()
        finally:
            self.executor.close()
