from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist

from ..epe.contracts import (
    DiffusionBackendFactory,
    EmbeddedRuntimeConfig,
    EmbeddedRuntimeHealth,
    ExecutorBuildContext,
    ImageDecodeCompletion,
    StageWorldSpec,
)
from .runtime import DiffusionServiceRuntime


@dataclass(frozen=True, slots=True)
class _BackendConfig:
    pool: Any
    output_root: str
    record_timeline: bool
    postprocess_workers: int


class EmbeddedDiffusionRuntime:
    """Host-embedded lifecycle around the distributed EPE worker runtime.

    This class does not create an HTTP server or spawn GPU processes. Every
    stage rank constructs and starts one instance inside its existing process.
    """

    def __init__(
        self,
        *,
        world: StageWorldSpec,
        pool: Any,
        executor,
        config: EmbeddedRuntimeConfig,
        backend=None,
    ) -> None:
        self.world = world
        self.pool = pool
        self.executor = executor
        self.config = config
        parallel = executor.parallel_context
        if parallel.rank != world.rank or parallel.world_size != world.world_size:
            raise ValueError("executor distributed world does not match StageWorldSpec")
        expected_local = str(world.local_device).rsplit(":", 1)[-1]
        if expected_local.isdigit() and parallel.local_rank != int(expected_local):
            raise ValueError(
                "executor local rank does not match StageWorldSpec.local_device"
            )
        if world.world_size > 1 and dist.is_initialized():
            signature = (
                world.stage_name,
                world.world_size,
                world.leader_rank,
                world.physical_device_ids,
            )
            signatures = [None] * world.world_size
            dist.all_gather_object(signatures, signature)
            if any(item != signature for item in signatures):
                raise ValueError("StageWorldSpec differs across stage ranks")
        self._backend = backend or DiffusionServiceRuntime(
            _BackendConfig(
                pool=pool,
                output_root=config.output_root,
                record_timeline=config.record_timeline,
                postprocess_workers=config.postprocess_workers,
            ),
            executor,
        )
        self._stop_requested = threading.Event()
        self._worker: threading.Thread | None = None
        self._state = "created"
        self._lock = threading.RLock()
        self._worker_error: BaseException | None = None

    @classmethod
    def create(
        cls,
        *,
        world: StageWorldSpec,
        pool: Any,
        executor_factory: DiffusionBackendFactory,
        config: EmbeddedRuntimeConfig | None = None,
    ) -> "EmbeddedDiffusionRuntime":
        executor = executor_factory.build(ExecutorBuildContext(world=world, pool=pool))
        return cls(
            world=world,
            pool=pool,
            executor=executor,
            config=config or EmbeddedRuntimeConfig(),
        )

    @classmethod
    def from_executor(
        cls,
        *,
        world: StageWorldSpec,
        pool: Any,
        executor,
        config: EmbeddedRuntimeConfig | None = None,
    ) -> "EmbeddedDiffusionRuntime":
        return cls(
            world=world,
            pool=pool,
            executor=executor,
            config=config or EmbeddedRuntimeConfig(),
        )

    @classmethod
    def from_backend(
        cls,
        *,
        world: StageWorldSpec,
        pool: Any,
        executor,
        backend,
        config: EmbeddedRuntimeConfig | None = None,
    ) -> "EmbeddedDiffusionRuntime":
        return cls(
            world=world,
            pool=pool,
            executor=executor,
            config=config or EmbeddedRuntimeConfig(),
            backend=backend,
        )

    @property
    def is_leader(self) -> bool:
        return self.world.is_leader

    @property
    def warmup_report(self) -> dict | None:
        return self._backend.warmup_report

    @property
    def service_backend(self):
        """Connector-facing backend for HTTP status and image retrieval."""
        return self._backend

    @property
    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._state != "created":
                raise RuntimeError(f"cannot start runtime in state {self._state}")
            self._state = "starting"
        try:
            self._backend.warmup()
            if self.world.world_size > 1:
                self.executor.parallel_context.barrier()
            worker = threading.Thread(
                target=self._worker_main,
                name=f"epe-worker-rank{self.world.rank}",
                daemon=False,
            )
            self._worker = worker
            worker.start()
            with self._lock:
                self._state = "running"
        except BaseException:
            with self._lock:
                self._state = "failed"
            self._backend.close()
            raise

    def _worker_main(self) -> None:
        try:
            self._backend.run_worker_pool(self._stop_requested)
        except BaseException as exc:
            self._worker_error = exc
            self._backend.fail(exc)
            with self._lock:
                self._state = "failed"

    def submit(self, request: object) -> str:
        self._require_running()
        if not self.is_leader:
            raise RuntimeError("only the stage leader may submit requests")
        return self._backend.submit_request(request)

    def poll(self, timeout_s: float = 0.0) -> list[ImageDecodeCompletion]:
        self._require_running(allow_stopping=True)
        if not self.is_leader:
            return []
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = time.monotonic() + timeout_s
        while True:
            completions = self._backend.poll_completions()
            if completions or time.monotonic() >= deadline:
                return completions
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def cancel(self, request_id: str, reason: str | None = None) -> bool:
        del reason
        self._require_running()
        if not self.is_leader:
            raise RuntimeError("only the stage leader may cancel requests")
        response = self._backend.cancel(request_id)
        return bool(response is not None and response.cancelled)

    def health(self) -> EmbeddedRuntimeHealth:
        backend = self._backend.health()
        active_lanes, last_plan_version = self._backend.runtime_snapshot()
        with self._lock:
            state = self._state
        return EmbeddedRuntimeHealth(
            state=state,
            rank=backend.rank,
            world_size=backend.world_size,
            queue_depth=backend.queue_depth,
            inflight=backend.inflight,
            completed=backend.completed,
            active_lanes=active_lanes,
            last_plan_version=last_plan_version,
            fatal_error=backend.fatal_error,
        )

    def wait(self, timeout_s: float | None = None) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.join(timeout=timeout_s)
        if worker.is_alive():
            raise TimeoutError("embedded runtime worker did not stop in time")
        if self._worker_error is not None:
            raise RuntimeError("embedded runtime worker failed") from self._worker_error

    def stop(self, *, graceful: bool = True, timeout_s: float = 30.0) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = time.monotonic() + timeout_s
        with self._lock:
            if self._state in {"stopped", "closed"}:
                return
            if self._state == "created":
                self._backend.close()
                self._state = "stopped"
                return
            if self._state != "failed":
                self._state = "stopping"
        self._backend.stop_admission()
        if graceful and self.is_leader:
            while time.monotonic() < deadline:
                health = self._backend.health()
                if health.queue_depth == 0 and health.inflight == 0:
                    break
                time.sleep(0.01)
        self._stop_requested.set()
        wait_error = None
        try:
            self.wait(max(0.0, deadline - time.monotonic()))
        except BaseException as exc:
            wait_error = exc
        finally:
            self._backend.close()
        with self._lock:
            self._state = "failed" if wait_error is not None else "stopped"
        if wait_error is not None:
            raise wait_error

    def _require_running(self, *, allow_stopping: bool = False) -> None:
        with self._lock:
            allowed = {"running"}
            if allow_stopping:
                allowed.add("stopping")
            if self._state not in allowed:
                raise RuntimeError(f"runtime is not running (state={self._state})")
