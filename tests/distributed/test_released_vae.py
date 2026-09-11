"""Dense comparisons using real release code without checkpoint weights.

Source downloads are optional for routine CI; the validation report pins their
revisions and gives the commands to populate refs/ for this acceptance suite.
"""

from __future__ import annotations

import importlib
import sys
import types
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from chitu_diffusion.parallel.vae import parallel_vae_decode

ROOT = Path(__file__).resolve().parents[2]
HUNYUAN = ROOT / "refs/HunyuanImage-3.0"
MINIMAX = ROOT / "refs/MiniMax-H3/FL2VA/video_vae"


def release_module(root, name):
    package_name = "_test_vae_" + root.name.replace("-", "_").replace(".", "_")
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module(package_name + "." + name)


def make_hunyuan():
    source = release_module(HUNYUAN, "autoencoder_kl_3d")
    return source.AutoencoderKLConv3D(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(32, 32),
        layers_per_block=1,
        ffactor_spatial=2,
        ffactor_temporal=2,
        sample_size=16,
        sample_tsize=4,
        only_decoder=True,
    ).eval()


class DirectVAE(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def decode(self, x, return_dict=False):
        return (self.decoder(x),)


def make_minimax(causal, registers):
    release_module(MINIMAX, "minimax_h3_video_vae")._ensure_vae_parallel_state()
    source = release_module(MINIMAX, "vae_vit")
    decoder = source.ViT3DDecoder(
        patch_size=2,
        patch_size_t=2,
        in_channels=4,
        num_layers=2,
        heads=2,
        dim_head=16,
        rope_dim_ratio=0.75,
        norm_type="rms_norm",
        qk_norm_type="rms_norm",
        ffn_use_gated=True,
        t_causal=causal,
        num_register_tokens=registers,
    ).eval()
    with torch.no_grad():
        for block in decoder.transformer_blocks:
            block.scale1.fill_(0.5)
            block.scale2.fill_(0.5)
    return DirectVAE(decoder).eval()


def _worker(rank, world, rendezvous, family, cuda):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    topology = SimpleNamespace(
        rank_in_lane=rank,
        width=world,
        is_leader=rank == 0,
        process_group=dist.group.WORLD,
    )
    try:
        cases = (
            [(False, 0)] if family == "hunyuan" else [(False, 0), (False, 4), (True, 2)]
        )
        for causal, registers in cases:
            torch.manual_seed(31)
            vae = (
                make_hunyuan()
                if family == "hunyuan"
                else make_minimax(causal, registers)
            ).to(device)
            keys = tuple(vae.state_dict())
            for frames, height in ((1, 8), (3, 9)):
                latent = torch.randn(2, 4, frames, height, 6, device=device)
                dist.broadcast(latent, src=0)
                with (
                    torch.inference_mode(),
                    torch.autocast(device.type, dtype=torch.float16, enabled=cuda),
                ):
                    expected = vae.decode(latent, return_dict=False)[0]
                    for _ in range(2):
                        actual = parallel_vae_decode(vae, latent, topology=topology)
                        assert (
                            vae._chitu_vae_decode_stats["vae_decode_mode"]
                            == "layerwise"
                        )
                        if rank == 0:
                            torch.testing.assert_close(
                                actual,
                                expected,
                                atol=3e-3 if cuda else 3e-5,
                                rtol=3e-3 if cuda else 4e-4,
                            )
                        else:
                            assert actual is None
                    torch.testing.assert_close(
                        vae.decode(latent, return_dict=False)[0],
                        expected,
                        atol=0,
                        rtol=0,
                    )
                    assert tuple(vae.state_dict()) == keys
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("family", ("hunyuan", "minimax"))
@pytest.mark.parametrize("world", (2, 4))
def test_released_spatial_vae_matches_dense(tmp_path, family, world):
    root = HUNYUAN if family == "hunyuan" else MINIMAX
    if not root.is_dir():
        pytest.skip(
            "published VAE source not available; see docs/validation/vae-parallel.md"
        )
    mp.spawn(
        _worker,
        args=(world, f"file://{tmp_path / 'rdzv'}", family, False),
        nprocs=world,
        join=True,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("family", ("hunyuan", "minimax"))
def test_released_spatial_vae_cuda(tmp_path, family):
    root = HUNYUAN if family == "hunyuan" else MINIMAX
    if torch.cuda.device_count() < 2 or not root.is_dir():
        pytest.skip("requires two GPUs and published VAE source")
    mp.spawn(
        _worker,
        args=(2, f"file://{tmp_path / 'rdzv'}", family, True),
        nprocs=2,
        join=True,
    )


def _temporal_worker(rank, rendezvous):
    from chitu_diffusion.models.minimax_h3.video_vae import MiniMaxH3VideoVAE

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(17)
        entry = release_module(MINIMAX, "minimax_h3_video_vae")
        entry._ensure_vae_parallel_state()
        source = release_module(MINIMAX, "klvae")
        backend = source.AutoencoderKLLegacy(
            ch=32,
            embed_dim=24,
            z_channels=24,
            use_3d_conv=True,
            use_vit_decoder=True,
            num_res_blocks=1,
            ch_mult=[1, 1],
            space_down=[2, 1],
            space_up=[2, 1],
            time_down=[2, 1],
            causal_decoder=False,
            clip_length=5,
            token_drop=1,
            isolated_first_frame=True,
            decoder_tiling=True,
            vit_decoder_kwargs=dict(
                heads=2, dim_head=16, rope_dim_ratio=0.75, num_layers=2
            ),
        ).eval()
        with torch.no_grad():
            for block in backend.decoder.transformer_blocks:
                block.scale1.fill_(0.5)
                block.scale2.fill_(0.5)
        wrapper = MiniMaxH3VideoVAE(
            entry.MiniMaxH3VideoVAE(backend),
            [0.1] * 24,
            [1.2] * 24,
            require_tiled_decoder=True,
        )
        topology = SimpleNamespace(
            rank_in_lane=rank,
            width=2,
            is_leader=rank == 0,
            process_group=dist.group.WORLD,
        )
        for frames in (1, 3, 7, 11):
            latent = torch.randn(1, 24, frames, 9, 6)
            expected = wrapper.decode(latent)
            for _ in range(2):
                actual = wrapper.decode_parallel(latent, topology=topology)
                assert backend._chitu_vae_decode_stats["vae_decode_mode"] == "layerwise"
                if rank == 0:
                    torch.testing.assert_close(actual, expected, rtol=4e-4, atol=3e-5)
                else:
                    assert actual is None
                assert backend.decoder_tiling is True
            serial = wrapper.decode_parallel(latent, topology=topology, enabled=False)
            if rank == 0:
                torch.testing.assert_close(serial, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_minimax_release_temporal_wrapper_preserves_dense_semantics(tmp_path):
    if not MINIMAX.is_dir():
        pytest.skip("requires published MiniMax VAE source")
    mp.spawn(
        _temporal_worker, args=(f"file://{tmp_path / 'rdzv'}",), nprocs=2, join=True
    )
