from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch
from PIL import Image

from chitu_diffusers.epac.timeline import RankTimelineRecorder
from chitu_diffusers.epac.worker_pool import LaneWorkResult
from chitu_diffusers.serve.protocol import ImageGenerateRequest
from chitu_diffusers.serve.zimage_runtime import (
    EpeZImageServiceRuntime,
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
    runtime = object.__new__(EpeZImageServiceRuntime)
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
    runtime._postprocess_slots = threading.BoundedSemaphore(2)
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
    assert runtime._records["req"].png is None

    processor.release.set()
    deadline = time.monotonic() + 2.0
    while runtime._records["req"].status == "postprocessing":
        assert time.monotonic() < deadline
        time.sleep(0.01)

    runtime._postprocess_executor.shutdown(wait=True)
    runtime.timeline.close()
    record = runtime._records["req"]
    assert record.status == "completed"
    assert record.png is not None and record.png.startswith(b"\x89PNG")
    stages = {
        json.loads(line)["stage"]
        for line in (tmp_path / "timeline-rank0.jsonl").read_text().splitlines()
    }
    assert {"cpu_postprocess", "cpu_png", "request_complete"} <= stages
