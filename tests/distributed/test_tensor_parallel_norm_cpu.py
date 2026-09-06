"""The reducing RMS norm against the dense one it has to reproduce.

This is the sharp test for the cross-shard reduction. At whole-model level a
missing reduction is surprisingly quiet -- the norm feeds attention, attention
feeds a residual stream, and a wrong Q/K scale moves the final output by less
than accumulated float noise on a small model. Here the norm's own output is
compared, where dropping the reduction is wrong by order one.
"""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 2
NORMALIZED_SIZE = 48


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reference() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    # A spread of magnitudes across the dimension, so the shards disagree about
    # the mean square and a rank that normalizes over its own slice cannot
    # accidentally land on the right scale.
    inputs = torch.randn(2, 5, NORMALIZED_SIZE, generator=generator)
    inputs *= torch.linspace(0.1, 4.0, NORMALIZED_SIZE)
    weight = torch.rand(NORMALIZED_SIZE, generator=generator) + 0.5
    dense = torch.nn.RMSNorm(NORMALIZED_SIZE, eps=1e-6, elementwise_affine=True)
    with torch.no_grad():
        dense.weight.copy_(weight)
    with torch.no_grad():
        return inputs, weight, dense(inputs)


def _worker(rank: int, port: int, results) -> None:
    from chitu_diffusion.parallel.tp import (
        TensorParallelRMSNorm,
        TensorParallelTopology,
    )
    from chitu_diffusion.parallel.tp.topology import set_tensor_parallel_topology

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
    )
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    set_tensor_parallel_topology(
        TensorParallelTopology(
            ranks=tuple(range(WORLD_SIZE)),
            rank=rank,
            rank_in_group=rank,
            degree=WORLD_SIZE,
            process_group=None,
        )
    )

    inputs, weight, _ = _reference()
    norm = TensorParallelRMSNorm(NORMALIZED_SIZE, 1e-6)
    span = slice(rank * norm.size_per_partition, (rank + 1) * norm.size_per_partition)
    with torch.no_grad():
        norm.weight.copy_(weight[span])
        results[rank] = norm(inputs[..., span])
    dist.barrier()
    dist.destroy_process_group()


def test_each_shard_reproduces_its_slice_of_the_dense_norm() -> None:
    inputs, weight, expected = _reference()
    width = NORMALIZED_SIZE // WORLD_SIZE

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(_free_port(), results), nprocs=WORLD_SIZE, join=True)

    for rank in range(WORLD_SIZE):
        span = slice(rank * width, (rank + 1) * width)
        torch.testing.assert_close(
            results[rank], expected[..., span], rtol=1e-6, atol=1e-6
        )

    # Guard the guard: a norm that reduced over its own shard instead of the
    # whole dimension has to miss by far more than the tolerance above, or the
    # assertions would hold for a norm that never reduced at all.
    slice_inputs = inputs[..., :width]
    shard_only = (
        slice_inputs
        * torch.rsqrt(
            slice_inputs.pow(2).sum(-1, keepdim=True) / NORMALIZED_SIZE + 1e-6
        )
        * weight[:width]
    )
    assert (shard_only - expected[..., :width]).abs().max() > 0.1
