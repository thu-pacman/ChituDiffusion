from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import AutoencoderKL

from chitu_diffusion.models.zimage.executor import ZImageImageDecoderExecutor
from chitu_diffusion.models.zimage.pipeline import EpeZImagePipeline
from chitu_diffusion.parallel.vae import VaeParallelPlacement, parallel_tiled_vae_decode


def _decode_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    topology = SimpleNamespace(
        rank_in_lane=rank, width=2, is_leader=rank == 0, process_group=dist.group.WORLD
    )
    try:
        for attention in (False, True):
            torch.manual_seed(7)
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D",),
                up_block_types=("UpDecoderBlock2D",),
                block_out_channels=(8,),
                latent_channels=2,
                norm_num_groups=4,
                mid_block_add_attention=attention,
                scaling_factor=1.0,
                shift_factor=0.0,
            ).eval()
            parallel = SimpleNamespace(
                rank=rank, local_rank=rank, world_size=2, allowed_widths=(1, 2)
            )
            pipeline = SimpleNamespace(
                vae=vae,
                vae_scale_factor=1,
                parallel_vae=True,
                vae_parallel_halo=2,
                parallel_context=parallel,
                maybe_free_model_hooks=lambda: None,
                transformer=SimpleNamespace(epe=SimpleNamespace(cfg_parallel=True)),
            )
            latents = torch.randn(1, 2, 32, 16)
            state = SimpleNamespace(latents=latents, complete=True)
            with torch.inference_mode():
                expected = vae.decode(latents, return_dict=False)[0]
                unsafe = parallel_tiled_vae_decode(
                    latents,
                    lambda tile: vae.decode(tile, return_dict=False)[0],
                    topology=topology,
                    latent_split_dim=2,
                    pixel_split_dim=2,
                    scale=1,
                    halo=2,
                )
                if rank == 0:
                    assert (unsafe - expected).abs().max().item() > 1e-3
                executor = ZImageImageDecoderExecutor(
                    pipeline, vae_placement=VaeParallelPlacement(sharded=True)
                )
                assert executor.parallel_vae
                with patch.object(vae, "decode", wraps=vae.decode) as decode:
                    actual = EpeZImagePipeline.decode_request(
                        pipeline, state, topology=topology, parallel_vae=True
                    )
                    assert decode.call_count == 1
                if rank == 0:
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

                    def jump(image):
                        return (image[:, :, 16] - image[:, :, 15]).abs().mean()

                    torch.testing.assert_close(
                        jump(actual), jump(expected), rtol=2e-4, atol=2e-5
                    )
                else:
                    assert actual is None

        # A local convolution still reconstructs the full decode with enough halo.
        conv = torch.nn.Conv2d(2, 3, (3, 1), padding=(1, 0)).eval()
        with torch.inference_mode():
            actual = parallel_tiled_vae_decode(
                latents,
                conv,
                topology=topology,
                latent_split_dim=2,
                pixel_split_dim=2,
                scale=1,
                halo=1,
            )
            if rank == 0:
                torch.testing.assert_close(actual, conv(latents))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_global_vae_decodes_exactly_on_the_leader_across_ranks(tmp_path) -> None:
    mp.spawn(
        _decode_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )
