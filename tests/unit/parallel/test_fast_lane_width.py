"""Which lane a fast transport binds to when CFG parallelism halves the plane.

A fast transport owns an NVSHMEM runtime and a process may host only one, so
the context builds it for exactly one lane width per rank. CFG parallelism
attends over half a tensor-parallel plane, so that width is not always the
plane.
"""

from __future__ import annotations

import contextlib
import os
import socket
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from chitu_diffusion.parallel.cp.context import EpeParallelContext


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _StubTransport:
    def __init__(self, name: str) -> None:
        self.name = name

    def close(self) -> None:
        return None


def _record_lane_widths(recorded: dict[int, str]):
    """Return factory doubles that log the transport chosen per lane width."""

    def factory(value, *, process_group, device, fast_eligible):
        width = (
            1
            if process_group is None
            else len(dist.get_process_group_ranks(process_group))
        )
        name = "fast" if fast_eligible else "nccl"
        recorded[width] = name
        assert (value != "torch") == fast_eligible
        return _StubTransport(name)

    return factory


def _cfg_lane_worker(rank: int, world_size: int, port: int, queue) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    recorded: dict[int, str] = {}
    factory = _record_lane_widths(recorded)
    with (
        patch(
            "chitu_diffusion.parallel.cp.context.create_ulysses_transport",
            factory,
        ),
        patch(
            "chitu_diffusion.parallel.cp.context.create_agkv_transport",
            factory,
        ),
    ):
        context = EpeParallelContext.from_torchrun(
            allowed_widths=(1, 2, 4),
            ulysses_degree=2,
            owns_process_group=False,
            ulysses_transport="fast",
            agkv_transport="fast",
            fast_lane_width=2,
        )
    queue.put((rank, recorded))
    dist.barrier()
    context.close()
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_cfg_halved_lane_owns_the_fast_transport() -> None:
    queue = mp.get_context("spawn").Queue()
    mp.spawn(_cfg_lane_worker, args=(4, _free_port(), queue), nprocs=4, join=True)
    reports = dict(queue.get() for _ in range(4))
    assert len(reports) == 4
    for recorded in reports.values():
        # The CFG branch attends over a two-rank lane, so that lane is fast and
        # the full plane it was cut from falls back to NCCL.
        assert recorded == {1: "nccl", 2: "fast", 4: "nccl"}


def test_fast_lane_width_must_divide_the_context_parallel_world(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(ValueError, match="fast_lane_width must be a positive divisor"):
        EpeParallelContext.from_torchrun(
            init_process_group=False,
            owns_process_group=False,
            fast_lane_width=3,
        )
