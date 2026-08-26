from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from chitu_diffusion.parallel.cp import (
    EpeParallelContext,
    ImageContextParallelAttention,
    ImageSelfAttention,
    cfg_parallel_rank_groups,
    resolve_context_parallel_config,
)
from chitu_diffusion.parallel.vae import (
    create_vae_parallel_group,
    parallel_spatial_vae_decode,
    parallel_tiled_vae_decode,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)


def test_cfg_parallel_rank_groups_pair_matching_cp_shards() -> None:
    cp_groups, cfg_pairs = cfg_parallel_rank_groups((4, 5, 6, 7))

    assert cp_groups == ((4, 5), (6, 7))
    assert cfg_pairs == ((4, 6), (5, 7))

    with pytest.raises(ValueError, match="even lane width"):
        cfg_parallel_rank_groups((0, 1, 2))


def test_context_parallel_config_defaults_to_agkv() -> None:
    assert resolve_context_parallel_config() == ("agkv", None)
    assert resolve_context_parallel_config("ulysses") == ("ulysses", None)
    assert resolve_context_parallel_config("ULYSSES", 4) == ("ulysses", 4)

    with pytest.raises(ValueError, match="agkv, ulysses"):
        resolve_context_parallel_config("unknown")
    with pytest.raises(ValueError, match="positive"):
        resolve_context_parallel_config("ulysses", 0)
    with pytest.raises(ValueError, match="only valid"):
        resolve_context_parallel_config("agkv", 2)


def _assert_lane_equivalence(
    parallel: EpeParallelContext,
    *,
    lane: tuple[int, ...],
    seed: int,
    heads: int = 3,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    batch, local_tokens, text_tokens, head_dim = 1, 2, 3, 4
    full_tokens = local_tokens * len(lane)
    image_query = torch.randn(batch, full_tokens, heads, head_dim, generator=generator)
    image_key = torch.randn(batch, full_tokens, heads, head_dim, generator=generator)
    image_value = torch.randn(batch, full_tokens, heads, head_dim, generator=generator)
    text_query = torch.randn(batch, text_tokens, heads, head_dim, generator=generator)
    text_key = torch.randn(batch, text_tokens, heads, head_dim, generator=generator)
    text_value = torch.randn(batch, text_tokens, heads, head_dim, generator=generator)
    rank_in_lane = lane.index(parallel.rank)
    start = rank_in_lane * local_tokens
    stop = start + local_tokens

    with parallel.activate(lane):
        joint_output, local_image_output = ImageContextParallelAttention("ulysses")(
            image_query[:, start:stop],
            image_key[:, start:stop],
            image_value[:, start:stop],
            text_query,
            text_key,
            text_value,
            lane_process_group=parallel.active.process_group,
            ulysses_topology=parallel.active_ulysses,
        )
    reference = _reference_attention(
        torch.cat([image_query, text_query], dim=1),
        torch.cat([image_key, text_key], dim=1),
        torch.cat([image_value, text_value], dim=1),
    )
    torch.testing.assert_close(
        local_image_output,
        reference[:, start:stop],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        joint_output,
        reference[:, full_tokens:],
        rtol=1e-5,
        atol=1e-6,
    )


def _assert_self_attention_equivalence(
    parallel: EpeParallelContext,
    *,
    lane: tuple[int, ...],
    seed: int,
    heads: int = 3,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    batch, local_tokens, head_dim = 2, 3, 4
    full_tokens = local_tokens * len(lane)
    query = torch.randn(batch, full_tokens, heads, head_dim, generator=generator)
    key = torch.randn(batch, full_tokens, heads, head_dim, generator=generator)
    value = torch.randn(batch, full_tokens, heads, head_dim, generator=generator)
    rank_in_lane = lane.index(parallel.rank)
    start = rank_in_lane * local_tokens
    stop = start + local_tokens

    with parallel.activate(lane):
        output = ImageSelfAttention("ulysses")(
            query[:, start:stop],
            key[:, start:stop],
            value[:, start:stop],
            lane_process_group=parallel.active.process_group,
            ulysses_topology=parallel.active_ulysses,
        )
        agkv_output = ImageSelfAttention("agkv")(
            query[:, start:stop],
            key[:, start:stop],
            value[:, start:stop],
            lane_process_group=parallel.active.process_group,
        )
    reference = _reference_attention(query, key, value)
    torch.testing.assert_close(
        output,
        reference[:, start:stop],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        agkv_output,
        reference[:, start:stop],
        rtol=1e-5,
        atol=1e-6,
    )


def _ulysses_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    parallel = EpeParallelContext.from_torchrun(
        allowed_widths=(1, 2, 4),
        owns_process_group=False,
        ulysses_degree=4,
    )
    assert parallel.active_ulysses.degree == 4
    _assert_lane_equivalence(parallel, lane=(0, 1, 2, 3), seed=31)
    _assert_self_attention_equivalence(parallel, lane=(0, 1, 2, 3), seed=37)
    _assert_lane_equivalence(parallel, lane=(0, 1, 2, 3), seed=41, heads=6)
    _assert_self_attention_equivalence(
        parallel,
        lane=(0, 1, 2, 3),
        seed=43,
        heads=6,
    )

    lane = (0, 1) if rank < 2 else (2, 3)
    with parallel.activate(lane):
        assert parallel.active_ulysses.degree == 2
    _assert_lane_equivalence(parallel, lane=lane, seed=47)
    _assert_self_attention_equivalence(parallel, lane=lane, seed=53)
    dist.barrier()
    parallel.close()
    dist.destroy_process_group()


def _vae_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    parallel = EpeParallelContext.from_torchrun(
        allowed_widths=(1, 2),
        owns_process_group=False,
    )
    latents = torch.arange(1 * 2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
    with parallel.activate((0, 1)) as topology:
        decoded = parallel_tiled_vae_decode(
            latents,
            lambda value: value.repeat_interleave(2, dim=2),
            topology=topology,
            latent_split_dim=2,
            pixel_split_dim=2,
            scale=2,
            halo=1,
        )
    if rank == 0:
        expected = latents.repeat_interleave(2, dim=2)
        assert decoded is not None
        torch.testing.assert_close(decoded, expected)
    else:
        assert decoded is None
    dist.barrier()
    parallel.close()
    dist.destroy_process_group()


def _spatial_vae_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    parallel = EpeParallelContext.from_torchrun(
        allowed_widths=(1, world_size),
        owns_process_group=False,
    )
    latents = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    with parallel.activate(tuple(range(world_size))) as topology:
        decoded = parallel_spatial_vae_decode(
            latents,
            lambda tile: tile.repeat_interleave(2, -2).repeat_interleave(2, -1),
            topology=topology,
            scale=2,
            tile_size=4,
            overlap_min=2,
        )
    if rank == 0:
        expected = latents.repeat_interleave(2, -2).repeat_interleave(2, -1)
        assert decoded is not None
        torch.testing.assert_close(decoded, expected)
    else:
        assert decoded is None
    dist.barrier()
    parallel.close()
    dist.destroy_process_group()


def _independent_vaep_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    topology = create_vae_parallel_group(2)
    assert topology.ranks == (0, 1)
    assert topology.is_member == (rank < 2)
    if topology.is_member:
        latents = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        decoded = parallel_spatial_vae_decode(
            latents,
            lambda tile: tile.repeat_interleave(2, -2).repeat_interleave(2, -1),
            topology=topology,
            scale=2,
            tile_size=4,
            overlap_min=2,
        )
        assert (decoded is not None) == topology.is_leader
    dist.barrier()
    topology.close()
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_static_ulysses_lanes_match_full_attention() -> None:
    mp.spawn(_ulysses_worker, args=(4, _free_port()), nprocs=4, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_dynamic_lane_parallel_vae_gathers_only_on_leader() -> None:
    mp.spawn(_vae_worker, args=(2, _free_port()), nprocs=2, join=True)


@pytest.mark.parametrize("world_size", (2, 4))
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_spatial_vae_tiles_are_gathered_only_on_leader(world_size: int) -> None:
    mp.spawn(
        _spatial_vae_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_vaep_degree_is_independent_from_stage_world_size() -> None:
    mp.spawn(
        _independent_vaep_worker,
        args=(4, _free_port()),
        nprocs=4,
        join=True,
    )
