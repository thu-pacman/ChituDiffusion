from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from chitu_diffusion.epe.api import DiffusersEPEPipeline
from chitu_diffusion.epe.executor import scheduling_options_from_pool
from chitu_diffusion.epe.scheduling.planner import EpeSchedulingModule
from chitu_diffusion.models.llada_image.api import (
    LLaDAImagePipeline,
    LLaDAImageRequest,
)
from chitu_diffusion.models.llada_image.epe import LLaDAImageEpeModule
from chitu_diffusion.models.llada_image.executor import LLaDAImageDecoderExecutor
from chitu_diffusion.parallel.vae import VaeParallelPlacement
from chitu_diffusion.serve.config import EPEServeConfig, HotSwitchPoolConfig
from chitu_diffusion.serve.protocol import ImageGenerateRequest


class _FakeParallel:
    rank = 0
    local_rank = 0
    world_size = 1
    allowed_widths = (1,)


def test_llada_scheduler_accepts_stage_pool_options() -> None:
    pool = HotSwitchPoolConfig(
        min_resize_gain_ms=1.0,
        resize_hysteresis_ms=2.0,
        resize_control_cost_ms=3.0,
    )
    module = LLaDAImageEpeModule(
        _FakeParallel(), **scheduling_options_from_pool(pool)
    )
    assert module.min_resize_gain_ms == 1.0
    assert module.resize_hysteresis_ms == 2.0
    assert module.resize_control_cost_ms == 3.0


def _make_executor() -> LLaDAImageDecoderExecutor:
    transformer = torch.nn.Linear(4, 4, bias=False)
    transformer.config = SimpleNamespace(in_channels=128)
    pipeline = SimpleNamespace(
        parallel_context=_FakeParallel(),
        transformer=transformer,
    )
    return LLaDAImageDecoderExecutor(
        pipeline,
        EpeSchedulingModule(_FakeParallel()),
        default_width=512,
        default_height=512,
        default_num_steps=20,
        vae_placement=VaeParallelPlacement(sharded=False),
    )


def test_http_request_schema_is_text_only() -> None:
    request = ImageGenerateRequest(prompt="a red fox")

    assert request.generation_mode == "text"

    for generation_mode in ("vq", "editing"):
        with pytest.raises(ValidationError, match="generation_mode"):
            ImageGenerateRequest(
                prompt="a red fox",
                generation_mode=generation_mode,
            )


@pytest.mark.parametrize(
    "field",
    ("image", "image_path", "image_url", "image_base64"),
)
def test_http_request_schema_rejects_image_inputs(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        ImageGenerateRequest.model_validate(
            {
                "prompt": "a red fox",
                "generation_mode": "text",
                field: "not-accepted",
            }
        )


def test_executor_request_payload_round_trips_text() -> None:
    executor = _make_executor()
    request = LLaDAImageRequest(
        prompt="a red fox",
        width=512,
        height=512,
    )

    payload = executor.serialize_request(request)
    restored = executor.deserialize_request(payload)

    assert payload["generation_mode"] == "text"
    assert payload["image"] is None
    assert payload["extra_inputs"] == {}
    assert restored.generation_mode == "text"


@pytest.mark.parametrize("generation_mode", ("vq", "editing"))
def test_offline_executor_payload_round_trips_non_text_modes(
    generation_mode: str,
) -> None:
    executor = _make_executor()
    source_image = Image.new("RGB", (32, 32))
    request = LLaDAImageRequest(
        prompt="a red fox",
        generation_mode=generation_mode,
        image=source_image if generation_mode == "editing" else None,
        width=512,
        height=512,
    )

    restored = executor.deserialize_request(executor.serialize_request(request))

    assert restored.generation_mode == generation_mode
    assert restored.image is (source_image if generation_mode == "editing" else None)


def test_facade_serving_uses_exact_vae_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[EPEServeConfig] = []
    monkeypatch.setattr(
        DiffusersEPEPipeline,
        "serve",
        lambda self, config: calls.append(config),
    )
    pipeline = object.__new__(LLaDAImagePipeline)

    pipeline.serve()
    explicit = EPEServeConfig(parallel_vae=True)
    pipeline.serve(explicit)

    assert calls[0].parallel_vae is False
    assert calls[0].default_num_steps == 50
    assert calls[1] is explicit


def test_configured_executor_defaults_normalize_to_offline_values() -> None:
    executor = _make_executor()

    normalized = executor.normalize_request(ImageGenerateRequest(prompt="a red fox"))
    explicit = executor.normalize_request(
        ImageGenerateRequest(
            prompt="a red fox",
            num_steps=8,
            guidance_scale=5.0,
        )
    )

    assert normalized.num_inference_steps == 20
    assert normalized.guidance_scale == 4.5
    assert executor.default_num_steps == 20
    assert explicit.num_inference_steps == 8
    assert explicit.guidance_scale == 5.0


def test_mapping_zero_num_steps_is_not_defaulted() -> None:
    executor = _make_executor()

    with pytest.raises(ValueError, match="num_steps must be positive"):
        executor.normalize_request({"prompt": "a red fox", "num_steps": 0})


def test_facade_resolves_serving_defaults_and_parallel_vae_opt_in() -> None:
    seed_executor = _make_executor()
    seed_executor.pipeline.transformer.epe = seed_executor.scheduling_module
    facade = LLaDAImagePipeline(seed_executor.pipeline, model_path="unused")

    default_backend = facade._create_backend(EPEServeConfig(parallel_vae=False))
    opt_in_backend = facade._create_backend(EPEServeConfig(parallel_vae=True))
    serial_backend = facade._create_backend(EPEServeConfig(vae_parallel_degree=1))
    normalized = default_backend.normalize_request(
        ImageGenerateRequest(prompt="a red fox")
    )

    assert EPEServeConfig().parallel_vae is True
    assert default_backend.default_num_steps == 50
    assert normalized.num_inference_steps == 50
    assert default_backend.parallel_vae is False
    assert opt_in_backend.parallel_vae is True
    assert serial_backend.parallel_vae is False


def test_direct_executor_defaults_to_exact_vae() -> None:
    seed = _make_executor()
    executor = LLaDAImageDecoderExecutor(seed.pipeline, seed.scheduling_module)
    assert executor.parallel_vae is False
