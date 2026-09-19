from __future__ import annotations

from types import SimpleNamespace

import pytest

from chitu_diffusion import FppConfig, WanPipeline, WanRequest
from chitu_diffusion.flexcache import CacheConfig
from chitu_diffusion.models.wan.executor import WanVideoDecoderExecutor
from chitu_diffusion.models.wan.pipeline import EpeWanPipeline


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"pipeline_parallel_degree": 0}, "positive integer"),
        ({"pipeline_parallel_degree": 3}, "world size"),
        ({"pipeline_parallel_degree": 2, "cfg_parallel": True}, "world size"),
        ({"pipeline_parallel_degree": 2, "tensor_parallel_degree": 2}, "TP=1"),
        (
            {"pipeline_parallel_degree": 2, "attention_mode": "ulysses"},
            "AGKV",
        ),
        ({"pipeline_parallel_degree": 2, "agkv_transport": "fast_agkv"}, "Fast CP"),
        ({"fpp_config": FppConfig()}, "world size"),
        ({"context_parallel_degree": 0}, "positive integer"),
        (
            {
                "pipeline_parallel_degree": 2,
                "fpp_config": FppConfig(reference_stages=2),
            },
            "single process",
        ),
        (
            {
                "pipeline_parallel_degree": 1,
                "context_parallel_degree": 2,
                "fpp_config": FppConfig(schedule="step"),
            },
            "step FPP",
        ),
    ],
)
def test_unsupported_fpp_setups_fail_before_loading(monkeypatch, options, message):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match=message):
        EpeWanPipeline.from_pretrained("/does/not/exist", **options)


def test_fpp_serving_rejected_without_closing_generate_pipeline():
    facade = WanPipeline(SimpleNamespace(fpp_enabled=True), model_path="unused")
    with pytest.raises(NotImplementedError, match="generate"):
        facade.serve()
    assert not facade._closed


def test_fpp_flexcache_combination_fails_before_request_preparation():
    backend = object.__new__(WanVideoDecoderExecutor)
    backend.pipeline = SimpleNamespace(fpp_enabled=True)
    request = WanRequest(prompt="test", cache=CacheConfig(strategy="teacache"))
    with pytest.raises(ValueError, match="FlexCache"):
        backend.prepare_request(request)
