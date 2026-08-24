from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import torch
from PIL import Image

from chitu_diffusion.epac.timeline import RankTimelineRecorder
from chitu_diffusion.epac.worker_pool import AsyncResultChannel, LaneWorkResult
from chitu_diffusion.serve.protocol import ImageGenerateRequest
from chitu_diffusion.serve.diffusion_runtime import (
    EpeDiffusionServiceRuntime,
    _RequestRecord,
)


class BlockingImageProcessor:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def postprocess(self, image: torch.Tensor, *, output_type: str):
        assert image.device.type == "cpu"
        assert output_type == "pil"
        self.started.set()
        assert self.release.wait(timeout=2.0)
        return [Image.new("RGB", (4, 4), color=(10, 20, 30))]


class FakeEpe:
    policy = "static_dp"

    def observe_assignment(self, assignment, measured_step_ms: float) -> None:
        assert assignment["ranks"] == [0]
        assert measured_step_ms == 1.0


def test_lane_releases_before_cpu_postprocess_completes(tmp_path) -> None:
    processor = BlockingImageProcessor()
    runtime = object.__new__(EpeDiffusionServiceRuntime)
    runtime.executor = SimpleNamespace(
        postprocess=lambda host_image: processor.postprocess(
            host_image,
            output_type="pil",
        )
    )
    runtime.timeline = RankTimelineRecorder(
        rank=0,
        output_root=tmp_path,
        enabled=True,
    )
    runtime.epe = FakeEpe()
    runtime.rank = 0
    runtime._lock = threading.RLock()
    runtime._postprocess_executor = ThreadPoolExecutor(max_workers=1)
    runtime._postprocess_futures = set()
    runtime._placements = {"req": (0,)}
    runtime._scheduler_steps = {"req": 1}
    request = ImageGenerateRequest(request_id="req", prompt="test")
    runtime._records = {
        "req": _RequestRecord(
            request=request,
            submitted_at=time.time(),
            status="running",
        )
    }
    now = time.time()
    result = LaneWorkResult(
        request_id="req",
        output=torch.zeros(1, 3, 4, 4),
        metadata={
            "strategy": "static_dp",
            "lane_rank": 0,
            "lane_ranks": [0],
            "started_at": now,
            "denoise_started_at": now,
            "completed_at": now,
            "steps": 1,
            "image_tokens": 1,
            "batch_size": 1,
            "cfg_conditions": 1,
            "predicted_step_ms": 1.0,
            "measured_step_ms": 1.0,
            "finalize_profile": {
                "finalize_total_ms": 2.0,
                "vae_decode_ms": 1.0,
                "d2h_ms": 1.0,
                "latent_contiguous": True,
            },
        },
    )

    runtime._complete_lane_work(result)

    assert processor.started.wait(timeout=1.0)
    assert runtime._records["req"].status == "postprocessing"
    assert runtime._records["req"].payload is None

    # Saturating the CPU executor must only queue later outputs. In particular,
    # publishing a GPU completion must never block the scheduler thread.
    request_2 = ImageGenerateRequest(request_id="req-2", prompt="test")
    runtime._records["req-2"] = _RequestRecord(
        request=request_2,
        submitted_at=time.time(),
        status="running",
    )
    runtime._placements["req-2"] = (0,)
    runtime._scheduler_steps["req-2"] = 1
    queued_at = time.monotonic()
    runtime._complete_lane_work(
        replace(result, request_id="req-2", output=torch.ones(1, 3, 4, 4))
    )
    assert time.monotonic() - queued_at < 0.1
    assert runtime._records["req-2"].status == "postprocessing"

    processor.release.set()
    deadline = time.monotonic() + 2.0
    while any(
        runtime._records[request_id].status == "postprocessing"
        for request_id in ("req", "req-2")
    ):
        assert time.monotonic() < deadline
        time.sleep(0.01)

    runtime._postprocess_executor.shutdown(wait=True)
    runtime.timeline.close()
    record = runtime._records["req"]
    assert record.status == "completed"
    assert record.payload is not None and record.payload.startswith(b"\x89PNG")
    assert record.media_type == "image/png"
    stages = {
        json.loads(line)["stage"]
        for line in (tmp_path / "timeline-rank0.jsonl").read_text().splitlines()
    }
    assert {"cpu_postprocess", "cpu_png", "request_complete"} <= stages


def test_idle_poll_does_not_resend_the_completed_lane_result() -> None:
    runtime = object.__new__(EpeDiffusionServiceRuntime)
    runtime.rank = 0
    runtime.timeline = SimpleNamespace(record=lambda *_args, **_kwargs: None)
    runtime._states = {"req": object()}
    runtime._apply_elastic_dispatch = lambda _response: None
    result = LaneWorkResult(
        request_id="req",
        output=torch.zeros(1),
        metadata={"lane_ranks": [0]},
    )
    runtime._run_elastic_lease = lambda _response, _lease, _exchange: {
        "request_id": None,
        "current_step": None,
        "state_owner_rank": None,
        "completed_request_ids": ["req"],
        "observed_step_ms": 1.0,
        "error": None,
        "finished_request_id": "req",
        "result": result,
    }
    pulse_reports = []
    published_results = []

    def exchange(event):
        pulse_reports.append(event["report"])
        if len(pulse_reports) == 1:
            return {
                "action": "pulse",
                "retired_request_ids": [],
                "epoch": 0,
                "planner": {},
                "assignments": [],
                "leases": [{"ranks": [0], "request_id": "req"}],
            }
        if len(pulse_reports) == 2:
            return {"action": "idle", "retired_request_ids": ["req"]}
        return {"action": "stop", "retired_request_ids": []}

    runtime._elastic_worker_loop(exchange, published_results.append)

    assert published_results == [result]
    assert pulse_reports[1]["finished_request_id"] == "req"
    assert pulse_reports[2] is None


def test_async_result_channel_drains_remote_results_without_blocking_submit() -> None:
    to_root = queue.Queue()
    to_peer = queue.Queue()
    completed = []
    transferred = []

    root = AsyncResultChannel(
        rank=0,
        world_size=2,
        control_group=None,
        complete=completed.append,
        send=lambda value, _destination: to_peer.put(value),
        recv=lambda _source: to_root.get(timeout=2.0),
    )
    peer = AsyncResultChannel(
        rank=1,
        world_size=2,
        control_group=None,
        complete=lambda _result: None,
        on_transfer=lambda result, _start, _end: transferred.append(result.request_id),
        send=lambda value, _destination: to_root.put(value),
        recv=lambda _source: to_peer.get(timeout=2.0),
    )
    root.start()
    peer.start()
    result = LaneWorkResult("remote", output=torch.ones(1024))

    submitted_at = time.monotonic()
    peer.submit(result)
    assert time.monotonic() - submitted_at < 0.1
    peer.close()
    root.close()

    assert completed == [result]
    assert transferred == ["remote"]
