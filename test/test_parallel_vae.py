from __future__ import annotations

from types import SimpleNamespace

import torch

from chitu_diffusion.parallel.vae import parallel_tiled_vae_decode


def _decode(tensor: torch.Tensor) -> torch.Tensor:
    return tensor + 1


def test_single_rank_lane_decodes_on_nonzero_global_rank(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    lane = SimpleNamespace(group_size=1, rank_in_group=0, gpu_group=None)
    latents = torch.zeros((1, 1, 2, 2))

    output = parallel_tiled_vae_decode(
        latents,
        _decode,
        latent_split_dim=2,
        pixel_split_dim=2,
        scale=1,
        group=lane,
    )

    assert output is not None
    torch.testing.assert_close(output, torch.ones_like(latents))


def test_disabled_parallel_decode_only_runs_on_lane_leader(monkeypatch) -> None:
    monkeypatch.setenv("CHITU_VAE_PARALLEL_DECODE", "0")
    lane = SimpleNamespace(group_size=2, rank_in_group=1, gpu_group=None)
    latents = torch.zeros((1, 1, 2, 2))

    output = parallel_tiled_vae_decode(
        latents,
        _decode,
        latent_split_dim=2,
        pixel_split_dim=2,
        scale=1,
        group=lane,
    )

    assert output is None
