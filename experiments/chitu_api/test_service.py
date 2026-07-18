from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from experiments.chitu_api.config import StageServiceConfig
from experiments.chitu_api.protocol import (
    AdmissionResponse,
    CancelResponse,
    HealthResponse,
    ImageGenerateRequest,
    RequestStatusResponse,
)
from experiments.chitu_api.service import create_app


def _config() -> StageServiceConfig:
    return StageServiceConfig.from_mapping(
        {
            "name": "image_decode",
            "process": "image_decoder",
            "factory": "zimage",
            "gpu": [4, 5, 6, 7],
            "parallelism": {
                "sp": 4,
                "chitu_pool": {"allowed_lane_widths": [1, 2, 4]},
            },
            "factory_args": {"model_path": "/models/Z-Image", "num_steps": 20},
            "service": {"host": "0.0.0.0", "port": 18080},
        }
    )


def test_stage_config_builds_zimage_epe_overrides() -> None:
    config = _config()
    assert config.parallelism.sp == 4
    assert config.service.endpoint == "http://127.0.0.1:18080"
    overrides = config.chitu_overrides()
    assert "infer.diffusion.cfg_size=1" in overrides
    assert "infer.diffusion.epe.cfg_parallel_max=1" in overrides
    assert "infer.diffusion.elastic_allowed_widths=[1,2,4]" in overrides


def test_stage_config_rejects_non_divisor_lane_width() -> None:
    raw = {
        "name": "image_decode",
        "factory": "zimage",
        "gpu": [0, 1, 2, 3],
        "parallelism": {
            "sp": 4,
            "chitu_pool": {"allowed_lane_widths": [1, 3]},
        },
        "factory_args": {"model_path": "/models/Z-Image"},
    }
    with pytest.raises(ValueError, match="divisors"):
        StageServiceConfig.from_mapping(raw)


class _FakeBackend:
    def __init__(self):
        self.requests: dict[str, RequestStatusResponse] = {}
        self.images: dict[str, bytes] = {}

    def submit(self, request: ImageGenerateRequest) -> AdmissionResponse:
        request_id = request.request_id or "generated"
        if request_id in self.requests:
            raise KeyError("duplicate")
        self.requests[request_id] = RequestStatusResponse(
            request_id=request_id,
            status="pending",
            submitted_at=1.0,
        )
        return AdmissionResponse(
            request_id=request_id,
            status_url=f"/v1/image-decode/{request_id}",
            image_url=f"/v1/image-decode/{request_id}/image",
        )

    def status(self, request_id: str):
        return self.requests.get(request_id)

    def cancel(self, request_id: str):
        if request_id not in self.requests:
            return None
        return CancelResponse(request_id=request_id, cancelled=True, detail="cancelled")

    def image(self, request_id: str):
        return self.images.get(request_id)

    def health(self) -> HealthResponse:
        return HealthResponse(
            state="running",
            rank=0,
            world_size=4,
            queue_depth=1,
            inflight=0,
            completed=0,
        )


def test_http_admission_status_and_not_ready_image() -> None:
    backend = _FakeBackend()
    client = TestClient(create_app(backend))
    response = client.post(
        "/v1/image-decode",
        json={"request_id": "req-1", "prompt": "test", "width": 1024, "height": 1024},
    )
    assert response.status_code == 202
    assert response.json()["request_id"] == "req-1"
    assert client.get("/v1/image-decode/req-1").json()["status"] == "pending"
    assert client.get("/v1/image-decode/req-1/image").status_code == 409


def test_http_returns_completed_png() -> None:
    backend = _FakeBackend()
    backend.requests["req-2"] = RequestStatusResponse(
        request_id="req-2",
        status="completed",
        submitted_at=1.0,
        completed_at=2.0,
        image_url="/v1/image-decode/req-2/image",
    )
    backend.images["req-2"] = b"\x89PNG\r\n\x1a\n"
    client = TestClient(create_app(backend))
    response = client.get("/v1/image-decode/req-2/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
