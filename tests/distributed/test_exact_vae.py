from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import (
    AutoencoderKL,
    AutoencoderKLFlux2,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
)

from chitu_diffusion.parallel.vae import parallel_vae_decode
from chitu_diffusion.parallel.vae.exact import _DecodeContext, _group_norm


def _vae(family):
    if family in {"kl", "flux2"}:
        cls = AutoencoderKL if family == "kl" else AutoencoderKLFlux2
        return cls(
            down_block_types=("DownEncoderBlock2D",) * 2,
            up_block_types=("UpDecoderBlock2D",) * 2,
            block_out_channels=(8, 8),
            layers_per_block=1,
            latent_channels=2,
            norm_num_groups=4,
        ).eval()
    cls = AutoencoderKLWan if family == "wan" else AutoencoderKLQwenImage
    return cls(
        base_dim=8,
        z_dim=2,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[True],
        latents_mean=[0, 0],
        latents_std=[1, 1],
    ).eval()


def _worker(rank, world, subgroup, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    try:
        ranks = [1, 3] if subgroup else list(range(world))
        group = dist.new_group(ranks) if subgroup else dist.group.WORLD
        if rank not in ranks:
            return
        local_rank = ranks.index(rank)
        topology = SimpleNamespace(
            rank_in_lane=local_rank,
            width=len(ranks),
            is_leader=local_rank == 0,
            process_group=group,
        )
        for family in ("kl", "flux2", "wan", "qwen"):
            torch.manual_seed(17)
            vae = _vae(family)
            keys = tuple(vae.state_dict())
            for rows in (8, 9):
                shape = (
                    (2, 2, rows, 6) if family in {"kl", "flux2"} else (1, 2, 3, rows, 6)
                )
                latent = torch.randn(shape)
                with torch.inference_mode():
                    expected = vae.decode(latent, return_dict=False)[0]
                    for _ in range(2):
                        actual = parallel_vae_decode(vae, latent, topology=topology)
                        assert (
                            vae._chitu_vae_decode_stats["vae_decode_mode"]
                            == "layerwise"
                        )
                        if topology.is_leader:
                            torch.testing.assert_close(
                                actual, expected, rtol=4e-4, atol=3e-5
                            )
                        else:
                            assert actual is None
                    # Original forwards and video cache accounting still work outside a decode.
                    torch.testing.assert_close(
                        vae.decode(latent, return_dict=False)[0],
                        expected,
                        rtol=0,
                        atol=0,
                    )
                    serial = parallel_vae_decode(
                        vae, latent, topology=topology, enabled=False
                    )
                    if topology.is_leader:
                        torch.testing.assert_close(serial, expected, rtol=0, atol=0)
                    else:
                        assert serial is None
                    assert tuple(vae.state_dict()) == keys

            def fail_decode(_):
                raise RuntimeError("injected decode failure")

            with pytest.raises(RuntimeError, match="injected decode failure"):
                parallel_vae_decode(
                    vae, latent, topology=topology, decode_fn=fail_decode
                )
            with torch.inference_mode():
                torch.testing.assert_close(
                    vae.decode(latent, return_dict=False)[0], expected, rtol=0, atol=0
                )
            vae.enable_tiling()
            with pytest.raises(ValueError, match="disable VAE tiling"):
                parallel_vae_decode(vae, latent, topology=topology)
            vae.disable_tiling()
            # A canvas too short for two nonempty bands follows the dense path.
            latent = latent[..., :1, :]
            result = parallel_vae_decode(vae, latent, topology=topology)
            assert (result is not None) == topology.is_leader
            assert vae._chitu_vae_decode_stats["vae_decode_mode"] == "leader"

        # Replacing the attention processor after adaptation must not silently
        # keep executing the previously supported attention implementation.
        from diffusers.models.attention_processor import AttnProcessor

        vae = _vae("kl")
        latent = torch.randn(1, 2, 8, 6)
        parallel_vae_decode(vae, latent, topology=topology)
        vae.set_attn_processor(AttnProcessor())
        with pytest.warns(UserWarning, match="standard SDPA self-attention"):
            actual = parallel_vae_decode(vae, latent, topology=topology)
        assert (actual is not None) == topology.is_leader
        assert vae._chitu_vae_decode_stats["vae_decode_mode"] == "leader"

        base, remainder = divmod(9, len(ranks))
        norm_rows = tuple(base + int(i < remainder) for i in range(len(ranks)))
        context = _DecodeContext(group, local_rank, tuple(ranks), norm_rows)
        for dtype, offset, atol in (
            (torch.float64, 1e5, 1e-9),
            (torch.float32, 0, 2e-6),
        ):
            norm = torch.nn.GroupNorm(4, 8).to(dtype).eval()
            x = torch.randn(2, 8, 9, 7, dtype=dtype) + offset
            # All ranks must normalize the same full tensor, including RNG state.
            dist.broadcast(x, src=ranks[0], group=group)
            start = sum(context.rows[:local_rank])
            shard = x.narrow(-2, start, context.rows[local_rank])
            with torch.inference_mode():
                actual = _group_norm(norm, shard, context)
                expected = norm(x).narrow(-2, start, context.rows[local_rank])
                torch.testing.assert_close(actual, expected, rtol=atol, atol=atol)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,subgroup", ((2, False), (4, False), (4, True)))
def test_static_parallel_vae_matches_dense_in_world_and_subgroup(
    tmp_path, world, subgroup
):
    mp.spawn(
        _worker,
        args=(world, subgroup, f"file://{tmp_path / 'rdzv'}"),
        nprocs=world,
        join=True,
    )


def _cuda_norm_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
        device_id=torch.device("cuda", rank),
    )
    try:
        context = _DecodeContext(dist.group.WORLD, rank, (0, 1), (5, 4))
        torch.manual_seed(17)
        for dtype, offset, atol in (
            (torch.float64, 1e5, 1e-8),
            (torch.float32, 0, 3e-6),
            (torch.float16, 0, 2e-3),
            (torch.bfloat16, 0, 2e-2),
        ):
            for affine in (False, True):
                norm = (
                    torch.nn.GroupNorm(4, 8, affine=affine).cuda(rank).to(dtype).eval()
                )
                if affine:
                    with torch.no_grad():
                        norm.weight.copy_(torch.randn_like(norm.weight))
                        norm.bias.copy_(torch.randn_like(norm.bias))
                for shape in ((2, 8, 9, 7), (2, 8, 3, 9, 7)):
                    x = torch.randn(shape, device=rank, dtype=dtype) + offset
                    dist.broadcast(x, src=0)
                    start = sum(context.rows[:rank])
                    shard = x.narrow(-2, start, context.rows[rank])
                    with torch.inference_mode():
                        actual = _group_norm(norm, shard, context)
                        expected = norm(x).narrow(-2, start, context.rows[rank])
                        torch.testing.assert_close(
                            actual, expected, rtol=atol, atol=atol
                        )
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_global_group_norm_fused_affine(tmp_path):
    mp.spawn(
        _cuda_norm_worker,
        args=(f"file://{tmp_path / 'cuda-rdzv'}",),
        nprocs=2,
        join=True,
    )
