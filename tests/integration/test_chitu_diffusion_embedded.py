from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from chitu_diffusion.epe import (
    EmbeddedRuntimeConfig,
    ImageDecodeCompletion,
    StageWorldSpec,
)
from chitu_diffusion.serve import EmbeddedDiffusionRuntime


@dataclass
class FakeParallel:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    barriers: int = 0

    def barrier(self) -> None:
        self.barriers += 1


class FakeExecutor:
    def __init__(self, parallel: FakeParallel) -> None:
        self.parallel_context = parallel
        self.scheduling_module = SimpleNamespace(scheduling_policy=object())
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeExecutorFactory:
    def __init__(self, executor: FakeExecutor) -> None:
        self.executor = executor

    def build(self, context):
        assert context.world.stage_name == "image_decode"
        return self.executor


class FakeBackend:
    def __init__(self) -> None:
        self.warmup_report = None
        self.closed = False
        self.accepting = True
        self.failed = None
        self.submitted = []
        self.completions = []

    def warmup(self):
        self.warmup_report = {"steps": 5, "rows": []}
        return self.warmup_report

    def run_worker_pool(self, stop_requested: threading.Event) -> None:
        stop_requested.wait()

    def submit_request(self, request) -> str:
        assert self.accepting
        self.submitted.append(request)
        return str(request["request_id"])

    def poll_completions(self):
        output, self.completions = self.completions, []
        return output

    def cancel(self, request_id: str):
        return SimpleNamespace(cancelled=request_id == "pending")

    def health(self):
        return SimpleNamespace(
            rank=0,
            world_size=1,
            queue_depth=0,
            inflight=0,
            completed=len(self.completions),
            fatal_error=self.failed,
        )

    def runtime_snapshot(self):
        return (), -1

    def stop_admission(self) -> None:
        self.accepting = False

    def fail(self, error: BaseException) -> None:
        self.failed = str(error)

    def close(self) -> None:
        self.closed = True


def _world(*, rank: int = 0, world_size: int = 1) -> StageWorldSpec:
    return StageWorldSpec(
        stage_name="image_decode",
        rank=rank,
        world_size=world_size,
        local_device=f"cuda:{rank}",
        physical_device_ids=tuple(range(world_size)),
    )


def test_stage_world_rejects_inconsistent_placement() -> None:
    with pytest.raises(ValueError, match="one id per stage rank"):
        StageWorldSpec(
            stage_name="image_decode",
            rank=0,
            world_size=2,
            local_device="cuda:0",
            physical_device_ids=(0,),
        )


def test_create_builds_executor_without_http_or_process_spawn(tmp_path) -> None:
    executor = FakeExecutor(FakeParallel())
    runtime = EmbeddedDiffusionRuntime.create(
        world=_world(),
        pool=SimpleNamespace(pulse_steps=1),
        executor_factory=FakeExecutorFactory(executor),
        config=EmbeddedRuntimeConfig(output_root=str(tmp_path)),
    )

    runtime.stop()

    assert executor.closed


def test_embedded_runtime_lifecycle_and_exactly_once_poll() -> None:
    backend = FakeBackend()
    executor = FakeExecutor(FakeParallel())
    runtime = EmbeddedDiffusionRuntime.from_backend(
        world=_world(),
        pool=SimpleNamespace(),
        executor=executor,
        backend=backend,
        config=EmbeddedRuntimeConfig(),
    )

    runtime.start()
    assert runtime.health().state == "running"
    assert runtime.submit({"request_id": "req"}) == "req"
    backend.completions.append(
        ImageDecodeCompletion("req", "completed", output=b"image")
    )
    assert [item.request_id for item in runtime.poll()] == ["req"]
    assert runtime.poll() == []
    assert runtime.cancel("pending")
    assert not runtime.cancel("running")

    runtime.stop()

    assert runtime.health().state == "stopped"
    assert backend.closed
    assert not runtime.worker_alive


def test_follower_rejects_ingress_and_emits_no_completions() -> None:
    backend = FakeBackend()
    parallel = FakeParallel(rank=1, world_size=2, local_rank=1)
    runtime = EmbeddedDiffusionRuntime.from_backend(
        world=_world(rank=1, world_size=2),
        pool=SimpleNamespace(),
        executor=FakeExecutor(parallel),
        backend=backend,
    )
    runtime.start()

    with pytest.raises(RuntimeError, match="leader"):
        runtime.submit({"request_id": "req"})
    assert runtime.poll() == []

    runtime.stop(graceful=False)
    assert parallel.barriers == 1
