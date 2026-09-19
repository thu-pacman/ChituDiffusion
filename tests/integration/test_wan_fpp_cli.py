"""FPP must preserve the current CLI's effective VAE and timing metadata."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("fpp", [False, True])
def test_wan_metadata_keeps_effective_vae_and_resolved_solver(monkeypatch, tmp_path, fpp):
    from chitu_diffusion.commands.generate import wan

    output = tmp_path / "video.mp4"
    loaded, closed = {}, []
    stats = {"schedule": "stream", "patch_steps": 46} if fpp else None
    pipeline = SimpleNamespace(
        parallel_context=SimpleNamespace(rank=0, world_size=4),
        last_cache_stats=None,
        last_fpp_stats=stats,
        last_vae_stats={"parallel_vae": False, "vae_decode_mode": "leader"},
        generate=lambda request: SimpleNamespace(frames=[np.zeros((1, 2, 2, 3))]),
        close=lambda: closed.append(True),
    )

    def load(path, **kwargs):
        loaded.update(kwargs)
        return pipeline

    monkeypatch.setattr(wan.WanPipeline, "from_pretrained", load)
    monkeypatch.setattr(wan, "export_to_video", lambda *args, **kwargs: None)
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(wan, "generation_timestamp", lambda: next(ticks))
    args = ["wan", "--model-path", "unused", "--output", str(output), "--save-frames", "--parallel-vae"]
    if fpp:
        args += ["--pipeline-parallel-degree", "4", "--fpp-patches", "7"]
    monkeypatch.setattr(sys, "argv", args)
    wan.main()

    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["sample_solver"] == loaded["sample_solver"] == ("unipc" if fpp else "euler")
    assert metadata["implementation"] == ("fpp" if fpp else "epe")
    assert metadata["fpp"] == stats
    assert metadata["fpp_config"] == (vars(loaded["fpp_config"]) if fpp else None)
    assert metadata["generate_seconds"] == metadata["elapsed_s"] == 2.5
    assert metadata["requested_parallel_vae"] is True
    assert metadata["parallel_vae"] is False
    assert metadata["vae_decode_mode"] == "leader"
    assert output.with_suffix(".npz").exists()
    assert closed == [True]
