from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

from .adapters import AdapterRegistry, DiffusersModelAdapter
from .config import EngineConfig
from .executor import DenoiseStepExecutor
from .optimization import OptimizationChain
from .request import (
    DiffusionRequest,
    DiffusionResult,
    RequestMetrics,
    RequestSnapshot,
    RequestStatus,
    StructuredError,
)
from .scheduling import (
    RoundRobinPolicy,
    SchedulableRequest,
    SchedulingPolicy,
    StepPlan,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _RequestRecord:
    request: DiffusionRequest
    submitted_at: float
    status: RequestStatus = RequestStatus.PENDING
    started_at: float | None = None
    completed_at: float | None = None
    state: Any = None
    steps_executed: int = 0
    cancel_requested: bool = False
    result: DiffusionResult | None = None
    emitted: bool = False


class DiffusersEngine:
    """Synchronous prototype of a request-local, pulse-driven step engine.

    The service/launcher owns threads and distributed command broadcast. This
    class owns only request state, model steps, optimization hooks and policy.
    """

    def __init__(
        self,
        pipeline: Any,
        adapter: DiffusersModelAdapter,
        *,
        config: EngineConfig | None = None,
        policy: SchedulingPolicy | None = None,
        optimizations: OptimizationChain | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.adapter = adapter
        self.config = config or EngineConfig()
        self.policy = policy or RoundRobinPolicy()
        self.optimizations = optimizations or OptimizationChain()
        self.executor = DenoiseStepExecutor(
            pipeline,
            adapter,
            self.optimizations,
        )
        self._records: dict[str, _RequestRecord] = {}
        self._pending: deque[str] = deque()
        self._active_order: list[str] = []
        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def from_pipeline(
        cls,
        pipeline: Any,
        *,
        adapter: DiffusersModelAdapter | None = None,
        registry: AdapterRegistry | None = None,
        config: EngineConfig | None = None,
        policy: SchedulingPolicy | None = None,
        optimizations: OptimizationChain | None = None,
    ) -> "DiffusersEngine":
        if adapter is None:
            registry = registry or AdapterRegistry()
            adapter = registry.resolve(pipeline)
        if not adapter.supports(pipeline):
            raise TypeError(
                f"adapter {adapter.name!r} does not support {type(pipeline).__name__}"
            )
        return cls(
            pipeline,
            adapter,
            config=config,
            policy=policy,
            optimizations=optimizations,
        )

    def submit(self, request: DiffusionRequest) -> str:
        with self._lock:
            self._ensure_open()
            if request.request_id in self._records:
                raise KeyError(f"duplicate request_id: {request.request_id}")
            queued = sum(
                not record.status.terminal for record in self._records.values()
            )
            if queued >= self.config.max_pending_requests:
                raise OverflowError(
                    f"request queue is full ({queued}/{self.config.max_pending_requests})"
                )
            record = _RequestRecord(request=request, submitted_at=time.time())
            self._records[request.request_id] = record
            self._pending.append(request.request_id)
            return request.request_id

    def pulse(self) -> int:
        """Run one scheduler pulse and return executed denoise steps."""
        with self._lock:
            self._ensure_open()
            self._apply_cancellations()
            self._admit_pending()
            schedulable = self._schedulable_requests()
            if not schedulable:
                return 0
            plans = tuple(
                self.policy.plan(
                    schedulable,
                    pulse_steps=self.config.pulse_steps,
                )
            )
            self._validate_plans(plans)
            executed = 0
            for plan in plans:
                record = self._records[plan.request_id]
                for _ in range(plan.steps):
                    if record.status is not RequestStatus.RUNNING:
                        break
                    if record.cancel_requested:
                        self._cancel_record(record)
                        break
                    executed += self._execute_step(record, plan)
                    if record.status.terminal:
                        break
            self._active_order = [
                request_id
                for request_id in self._active_order
                if self._records[request_id].status is RequestStatus.RUNNING
            ]
            return executed

    def generate(self, request: DiffusionRequest) -> DiffusionResult:
        self.submit(request)
        while True:
            result = self.result(request.request_id)
            if result is not None:
                return result
            progressed = self.pulse()
            if (
                progressed == 0
                and self.status(request.request_id).status is RequestStatus.RUNNING
            ):
                raise RuntimeError(
                    "scheduling policy returned no work for a running request"
                )

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            record = self._records.get(request_id)
            if record is None or record.status.terminal:
                return False
            if record.status is RequestStatus.PENDING:
                self._cancel_record(record)
            else:
                record.cancel_requested = True
            return True

    def status(self, request_id: str) -> RequestSnapshot:
        with self._lock:
            record = self._get_record(request_id)
            error = record.result.error if record.result is not None else None
            return RequestSnapshot(
                request_id=request_id,
                status=record.status,
                submitted_at=record.submitted_at,
                started_at=record.started_at,
                completed_at=record.completed_at,
                steps_executed=record.steps_executed,
                error=error,
            )

    def result(self, request_id: str) -> DiffusionResult | None:
        with self._lock:
            return self._get_record(request_id).result

    def poll(self) -> list[DiffusionResult]:
        """Return each newly terminal result exactly once."""
        with self._lock:
            results = []
            for record in self._records.values():
                if record.result is not None and not record.emitted:
                    results.append(record.result)
                    record.emitted = True
            return results

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for record in self._records.values():
                if not record.status.terminal:
                    self._cancel_record(record)
            self._closed = True

    def _admit_pending(self) -> None:
        available = self.config.max_inflight_requests - len(self._active_order)
        if available <= 0:
            return
        candidates = [
            self._records[request_id]
            for request_id in self._pending
            if self._records[request_id].status is RequestStatus.PENDING
        ]
        candidates.sort(
            key=lambda record: (
                self._request_deadline_ms(record.request) is None,
                record.submitted_at
                + (self._request_deadline_ms(record.request) or 0.0) / 1000.0,
                -record.request.priority,
                record.submitted_at,
            )
        )
        selected = candidates[:available]
        selected_ids = {record.request.request_id for record in selected}
        self._pending = deque(
            request_id
            for request_id in self._pending
            if request_id not in selected_ids
            and self._records[request_id].status is RequestStatus.PENDING
        )
        for record in selected:
            record.status = RequestStatus.RUNNING
            record.started_at = time.time()
            try:
                record.state = self.adapter.prepare_request(
                    self.pipeline, record.request
                )
            except Exception as exc:
                self._fail_record(record, exc)
                continue
            self._active_order.append(record.request.request_id)

    def _schedulable_requests(self) -> list[SchedulableRequest]:
        output = []
        for request_id in self._active_order:
            record = self._records[request_id]
            if record.status is not RequestStatus.RUNNING or record.cancel_requested:
                continue
            deadline_ms = self._request_deadline_ms(record.request)
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
                    priority=record.request.priority,
                    profile=self.adapter.profile(record.state),
                )
            )
        return output

    def _request_deadline_ms(self, request: DiffusionRequest) -> float | None:
        if request.deadline_ms is not None:
            return request.deadline_ms
        return self.config.default_deadline_ms

    def _validate_plans(self, plans: Sequence[StepPlan]) -> None:
        seen: set[str] = set()
        for plan in plans:
            if plan.request_id in seen:
                raise ValueError(
                    f"policy scheduled request {plan.request_id!r} more than once"
                )
            seen.add(plan.request_id)
            record = self._records.get(plan.request_id)
            if record is None or record.status is not RequestStatus.RUNNING:
                raise ValueError(
                    f"policy scheduled non-running request {plan.request_id!r}"
                )

    def _execute_step(self, record: _RequestRecord, plan: StepPlan) -> int:
        try:
            self.executor.execute(record.request, record.state, plan)
            record.steps_executed += 1
            if self.adapter.is_complete(record.state):
                output = self.adapter.finalize_request(
                    self.pipeline, record.state, record.request
                )
                self._complete_record(record, output)
            return 1
        except Exception as exc:
            self._fail_record(record, exc)
            return 0

    def _apply_cancellations(self) -> None:
        for request_id in tuple(self._active_order):
            record = self._records[request_id]
            if record.cancel_requested and record.status is RequestStatus.RUNNING:
                self._cancel_record(record)
        self._active_order = [
            request_id
            for request_id in self._active_order
            if self._records[request_id].status is RequestStatus.RUNNING
        ]

    def _complete_record(self, record: _RequestRecord, output: Any) -> None:
        self._finish_record(record, RequestStatus.COMPLETED, output=output)

    def _cancel_record(self, record: _RequestRecord) -> None:
        if record.state is not None:
            try:
                self.adapter.abort_request(self.pipeline, record.state)
            except Exception:
                logger.exception(
                    "failed to abort cancelled request %s",
                    record.request.request_id,
                )
        self._finish_record(record, RequestStatus.CANCELLED)

    def _fail_record(self, record: _RequestRecord, exc: Exception) -> None:
        if record.state is not None:
            try:
                self.adapter.abort_request(self.pipeline, record.state)
            except Exception:
                logger.exception(
                    "failed to abort failed request %s",
                    record.request.request_id,
                )
        self._finish_record(
            record,
            RequestStatus.FAILED,
            error=StructuredError(type=type(exc).__name__, message=str(exc)),
        )

    def _finish_record(
        self,
        record: _RequestRecord,
        status: RequestStatus,
        *,
        output: Any = None,
        error: StructuredError | None = None,
    ) -> None:
        if record.status.terminal:
            return
        record.status = status
        record.completed_at = time.time()
        record.result = DiffusionResult(
            request_id=record.request.request_id,
            status=status,
            output=output,
            error=error,
            metrics=RequestMetrics(
                submitted_at=record.submitted_at,
                started_at=record.started_at,
                completed_at=record.completed_at,
                steps_executed=record.steps_executed,
            ),
        )
        try:
            self.executor.on_request_end(record.request.request_id)
        except Exception:
            logger.exception(
                "request-end hook failed for %s",
                record.request.request_id,
            )

    def _get_record(self, request_id: str) -> _RequestRecord:
        try:
            return self._records[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request_id: {request_id}") from exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("engine is closed")
