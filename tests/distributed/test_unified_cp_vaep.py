"""Terminal decode placement, exercised against real process groups.

The stage config lets CP and VAEP be chosen independently, so these workers
check the two placements a config can produce: decode that follows the denoise
lane, and a fixed group that spans ranks the lane may not contain.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from chitu_diffusion.parallel.cp import EpeParallelContext
from chitu_diffusion.parallel.vae import (
    create_vae_parallel_placement,
    parallel_spatial_vae_decode,
)

WORLD_SIZE = 4


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _latents(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, 8, 8, generator=generator)


def _upsample(tile: torch.Tensor) -> torch.Tensor:
    return tile.repeat_interleave(2, -2).repeat_interleave(2, -1)


def _decode(
    latents: torch.Tensor, topology, *, sharded: bool = True
) -> torch.Tensor | None:
    """Stand in for a pipeline's terminal decode with the placement it was given."""

    if not sharded:
        return _upsample(latents) if topology.is_leader else None
    return parallel_spatial_vae_decode(
        latents,
        _upsample,
        topology=topology,
        scale=2,
        tile_size=8,
        overlap_min=2,
    )


def _lane_of(rank: int, width: int) -> tuple[int, int]:
    start = (rank // width) * width
    return tuple(range(start, start + width))


def _lane_coupled_worker(rank: int, world_size: int, port: int, width: int) -> None:
    _init(rank, world_size, port)
    parallel = EpeParallelContext.from_torchrun(
        allowed_widths=(1, 2, world_size),
        owns_process_group=False,
    )
    placement = create_vae_parallel_placement(None, halo=4)
    lane = _lane_of(rank, width)
    try:
        # Every lane decodes its own request, which is why an elastic stage may
        # not pin decode to a group that outlives one lane.
        for request in range(3):
            latents = _latents(seed=lane[0] * 100 + request)
            with parallel.activate(lane) as topology:
                decoded = _decode(latents, placement.resolve(topology))
            if rank == lane[0]:
                assert decoded is not None
                torch.testing.assert_close(decoded, _upsample(latents))
            else:
                assert decoded is None
        dist.barrier()
    finally:
        placement.close()
        parallel.close()
        dist.destroy_process_group()


def _stage_group_worker(rank: int, world_size: int, port: int, degree: int) -> None:
    _init(rank, world_size, port)
    parallel = EpeParallelContext.from_torchrun(
        allowed_widths=(1, world_size),
        owns_process_group=False,
    )
    placement = create_vae_parallel_placement(degree, halo=4)
    lane = tuple(range(world_size))
    # Degree 1 keeps the whole canvas on the leader instead of building a group.
    decodes = rank < degree if placement.is_stage_scoped else True
    try:
        assert placement.degree == degree
        assert placement.is_stage_scoped == (degree > 1)
        for request in range(3):
            latents = _latents(seed=request)
            with parallel.activate(lane) as topology:
                resolved = placement.resolve(topology)
                assert (resolved is not None) == decodes
                decoded = (
                    None
                    if resolved is None
                    else _decode(latents, resolved, sharded=placement.sharded)
                )
            if rank == 0:
                assert decoded is not None
                torch.testing.assert_close(decoded, _upsample(latents))
            else:
                assert decoded is None
        # A rank that sat out every decode must still be in step with the group.
        dist.barrier()
    finally:
        placement.close()
        parallel.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
@pytest.mark.parametrize("width", (1, 2, 4))
def test_lane_coupled_decode_returns_to_each_denoise_lane_leader(width: int) -> None:
    mp.spawn(
        _lane_coupled_worker,
        args=(WORLD_SIZE, _free_port(), width),
        nprocs=WORLD_SIZE,
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
@pytest.mark.parametrize("degree", (1, 2, 4))
def test_a_fixed_decode_group_returns_only_to_the_stage_leader(degree: int) -> None:
    mp.spawn(
        _stage_group_worker,
        args=(WORLD_SIZE, _free_port(), degree),
        nprocs=WORLD_SIZE,
        join=True,
    )
