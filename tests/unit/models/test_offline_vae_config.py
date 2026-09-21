from __future__ import annotations

from types import SimpleNamespace

import pytest

from chitu_diffusion.models.wan.api import WanPipeline
from chitu_diffusion.models.zimage.api import ZImagePipeline
from chitu_diffusion.serve.config import EPEServeConfig


@pytest.mark.parametrize("facade_class", [WanPipeline, ZImagePipeline])
@pytest.mark.parametrize("enabled,halo", [(False, 0), (False, 32), (True, 16)])
def test_pretrained_vae_options_reach_each_offline_backend(
    monkeypatch, facade_class, enabled, halo
) -> None:
    parallel = SimpleNamespace(rank=0, local_rank=0, world_size=1, allowed_widths=(1,))

    def load(path, **kwargs):
        pipeline = SimpleNamespace(
            parallel_context=parallel,
            transformer=SimpleNamespace(epe=SimpleNamespace(cfg_parallel=True)),
            parallel_vae=kwargs.pop("parallel_vae", facade_class is WanPipeline),
            vae_parallel_halo=kwargs.pop("vae_parallel_halo", 8),
            set_progress_bar_config=lambda **_: None,
        )
        pipeline.to = lambda device: pipeline
        return pipeline

    monkeypatch.setattr(facade_class.pipeline_class, "from_pretrained", load)
    facade = facade_class.from_pretrained(
        "unused", device="cpu", parallel_vae=enabled, vae_parallel_halo=halo
    )
    # generate() rebuilds its backend for every request.
    for _ in range(2):
        backend = facade._create_backend()
        assert backend.parallel_vae is enabled
        assert backend.vae_parallel_halo == halo

    backend = facade._create_backend(
        EPEServeConfig(parallel_vae=not enabled, vae_parallel_halo=7)
    )
    assert backend.parallel_vae is (not enabled)
    assert backend.vae_parallel_halo == 7
    backend = facade._create_backend(EPEServeConfig(vae_parallel_degree=1))
    assert not backend.parallel_vae


@pytest.mark.parametrize("facade_class", [WanPipeline, ZImagePipeline])
def test_negative_halo_fails_before_model_loading(facade_class) -> None:
    with pytest.raises(ValueError, match="halo must be non-negative"):
        facade_class.from_pretrained("does-not-exist", vae_parallel_halo=-1)
