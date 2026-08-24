from __future__ import annotations

from fastapi.testclient import TestClient

from chitu_diffusion.models.llada_image.executor import LLaDAImageExecutorFactory
from chitu_diffusion.serve.app import create_app
from chitu_diffusion.serve.config import StageServiceConfig
from chitu_diffusion.serve.protocol import AdmissionResponse
from chitu_diffusion.serve.runtime import DiffusionServiceRuntime


class _AdmissionBackend:
    def __init__(self) -> None:
        self.requests = []

    def submit(self, request):
        self.requests.append(request)
        return AdmissionResponse(
            request_id=request.request_id or "generated",
            status_url="/v1/image-decode/generated",
            image_url="/v1/image-decode/generated/image",
        )


def _llada_stage_config() -> StageServiceConfig:
    return StageServiceConfig.from_mapping(
        {
            "name": "llada-image",
            "factory": "llada-image",
            "gpu": [0],
            "parallelism": {
                "sp": 1,
                "chitu_pool": {"allowed_lane_widths": [1]},
            },
            "factory_args": {
                "model_path": "/models/llada-image",
                "num_steps": 20,
                "cfg_parallel": True,
            },
        }
    )


def test_http_app_accepts_text_and_rejects_image_modes() -> None:
    backend = _AdmissionBackend()
    client = TestClient(create_app(backend))

    response = client.post(
        "/v1/image-decode",
        json={
            "request_id": "text-request",
            "prompt": "a red fox",
            "generation_mode": "text",
        },
    )
    assert response.status_code == 202
    assert [request.generation_mode for request in backend.requests] == ["text"]

    for payload in (
        {
            "prompt": "a red fox",
            "generation_mode": "vq",
        },
        {
            "prompt": "edit this image",
            "generation_mode": "editing",
            "image_url": "https://example.invalid/image.png",
        },
        {
            "prompt": "a red fox",
            "image_base64": "not-accepted",
        },
    ):
        response = client.post("/v1/image-decode", json=payload)
        assert response.status_code == 422

    assert len(backend.requests) == 1


def test_stage_runtime_builds_llada_image_factory(monkeypatch) -> None:
    config = _llada_stage_config()
    built = {}
    fake_executor = object()

    def fake_build(self, context):
        built["factory"] = self
        built["context"] = context
        return fake_executor

    def fake_runtime_init(self, selected_config, executor):
        self.config = selected_config
        self.executor = executor

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(LLaDAImageExecutorFactory, "build", fake_build)
    monkeypatch.setattr(DiffusionServiceRuntime, "__init__", fake_runtime_init)

    runtime = DiffusionServiceRuntime.from_config(config)

    assert runtime.config is config
    assert runtime.executor is fake_executor
    assert built["factory"].model_path == "/models/llada-image"
    assert built["factory"].default_num_steps == 20
    assert built["factory"].cfg_parallel is True
    assert built["factory"].parallel_vae is False
    assert built["context"].world.physical_device_ids == (0,)
