from __future__ import annotations

from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient
from PIL import Image

from chitu_diffusion.epac.image_decoder import (
    TerminalArtifact,
    normalize_terminal_artifact,
)
from chitu_diffusion.epac.model_executor import DiffusersBackend
from chitu_diffusion.serve.app import create_app
from chitu_diffusion.serve.protocol import (
    AdmissionResponse,
    HealthResponse,
    RequestStatusResponse,
)
from chitu_diffusion.serve.diffusion_runtime import EpeDiffusionServiceRuntime


def test_legacy_tensor_normalizes_to_named_terminal_artifact() -> None:
    tensor = torch.ones(2)

    artifact = normalize_terminal_artifact(tensor)

    assert artifact is not None
    assert artifact.tensors == {"output": tensor}
    assert artifact.metadata == {}


def test_default_artifact_adapter_preserves_executor_postprocess_default() -> None:
    class LegacyExecutor:
        def postprocess(self, tensor: torch.Tensor, *, output_type: str = "np"):
            return tensor, output_type

    tensor = torch.ones(1)
    output, output_type = DiffusersBackend.postprocess_artifact(
        LegacyExecutor(),
        TerminalArtifact(tensors={"output": tensor}),
    )

    assert output is tensor
    assert output_type == "np"


def test_runtime_copies_every_terminal_tensor_to_host() -> None:
    runtime = object.__new__(EpeDiffusionServiceRuntime)
    runtime.rank = 0
    runtime.executor = SimpleNamespace(
        finalize_gpu=lambda *_args, **timings: _terminal_output(
            timings["timings"]
        )
    )
    events = []
    runtime.timeline = SimpleNamespace(
        record=lambda stage, **values: events.append((stage, values))
    )

    artifact, _timings = runtime._decode_to_host(
        object(),
        request_id="req",
        lane_ranks=(0,),
    )

    assert set(artifact.tensors) == {"video", "audio"}
    assert all(tensor.device.type == "cpu" for tensor in artifact.tensors.values())
    assert artifact.metadata == {"kind": "av"}
    assert events[-1][0] == "d2h"
    assert events[-1][1]["metadata"]["tensor_names"] == ["video", "audio"]


def _terminal_output(timings: dict[str, object]) -> TerminalArtifact:
    timings["vae_start_unix_ns"] = 1
    timings["vae_end_unix_ns"] = 2
    return TerminalArtifact(
        tensors={
            "video": torch.zeros(1, 2),
            "audio": torch.ones(1, 3),
        },
        metadata={"kind": "av"},
    )


def test_artifact_postprocess_and_package_run_off_gpu() -> None:
    seen = []

    def postprocess(artifact: TerminalArtifact):
        seen.append(artifact)
        assert set(artifact.tensors) == {"rgb", "mask"}
        assert all(
            tensor.device.type == "cpu" for tensor in artifact.tensors.values()
        )
        return [Image.new("RGB", (2, 2), color=(1, 2, 3))]

    runtime = object.__new__(EpeDiffusionServiceRuntime)
    runtime.executor = SimpleNamespace(postprocess_artifact=postprocess)
    runtime.timeline = SimpleNamespace(record=lambda *_args, **_kwargs: None)
    artifact = TerminalArtifact(
        tensors={
            "rgb": torch.zeros(1, 3, 2, 2),
            "mask": torch.ones(1, 1, 2, 2),
        }
    )

    payload, output, _timings = runtime._postprocess_to_png(
        artifact,
        request_id="req",
        lane_ranks=(0,),
    )

    assert seen == [artifact]
    assert payload is not None and payload.startswith(b"\x89PNG")
    assert output.size == (2, 2)


class _MediaBackend:
    def submit(self, _request) -> AdmissionResponse:
        return AdmissionResponse(
            request_id="req",
            status_url="/v1/image-decode/req",
            media_url="/v1/media/req",
            image_url="/v1/image-decode/req/image",
        )

    def status(self, request_id: str) -> RequestStatusResponse | None:
        if request_id != "req":
            return None
        return RequestStatusResponse(
            request_id=request_id,
            status="completed",
            submitted_at=1.0,
            completed_at=2.0,
            media_url="/v1/media/req",
            image_url="/v1/image-decode/req/image",
        )

    def cancel(self, _request_id):
        return None

    def image(self, request_id: str) -> bytes | None:
        return self.media(request_id)

    def media(self, request_id: str) -> bytes | None:
        return b"payload" if request_id == "req" else None

    def media_type(self, request_id: str) -> str | None:
        return "video/mp4" if request_id == "req" else None

    def health(self) -> HealthResponse:
        return HealthResponse(
            state="running",
            rank=0,
            world_size=1,
            queue_depth=0,
            inflight=0,
            completed=1,
        )


def test_media_endpoint_is_canonical_and_image_route_remains_alias() -> None:
    client = TestClient(create_app(_MediaBackend()))

    media = client.get("/v1/media/req")
    image_alias = client.get("/v1/image-decode/req/image")

    assert media.status_code == 200
    assert media.content == b"payload"
    assert media.headers["content-type"] == "video/mp4"
    assert image_alias.content == media.content
