from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Iterator

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class TensorParallelTopology:
    ranks: tuple[int, ...]
    rank: int
    rank_in_group: int
    degree: int
    process_group: object | None


_LOCK = RLock()
_TOPOLOGY = TensorParallelTopology(
    ranks=(0,),
    rank=0,
    rank_in_group=0,
    degree=1,
    process_group=None,
)


def set_tensor_parallel_topology(topology: TensorParallelTopology) -> None:
    global _TOPOLOGY
    with _LOCK:
        _TOPOLOGY = topology


def adopt_external_tensor_parallel_group(
    *,
    ranks: tuple[int, ...],
    rank: int,
    process_group: object | None,
) -> TensorParallelTopology:
    """Bind Chitu TP kernels to a process group owned by another runtime.

    This function never creates or destroys a process group. It is intended for
    hosts such as SGLang that initialize distributed topology before loading a
    Chitu model adapter.
    """

    normalized = tuple(int(value) for value in ranks)
    global_rank = int(rank)
    if not normalized or global_rank not in normalized:
        raise ValueError("external TP ranks must contain the current rank")
    if len(normalized) > 1 and process_group is None:
        raise ValueError("multi-rank external TP requires a process group")
    topology = TensorParallelTopology(
        ranks=normalized,
        rank=global_rank,
        rank_in_group=normalized.index(global_rank),
        degree=len(normalized),
        process_group=process_group,
    )
    set_tensor_parallel_topology(topology)
    return topology


@contextmanager
def use_tensor_parallel_topology(
    topology: TensorParallelTopology,
) -> Iterator[TensorParallelTopology]:
    """Temporarily use an externally owned TP topology."""

    previous = get_tp_topology()
    set_tensor_parallel_topology(topology)
    try:
        yield topology
    finally:
        set_tensor_parallel_topology(previous)


def reset_tensor_parallel_topology() -> None:
    set_tensor_parallel_topology(
        TensorParallelTopology(
            ranks=(0,),
            rank=0,
            rank_in_group=0,
            degree=1,
            process_group=None,
        )
    )


def get_tp_topology() -> TensorParallelTopology:
    with _LOCK:
        return _TOPOLOGY


def get_tp_world_size() -> int:
    return get_tp_topology().degree


def get_tp_rank() -> int:
    return get_tp_topology().rank_in_group


def get_tp_group() -> object | None:
    return get_tp_topology().process_group


def tensor_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    topology = get_tp_topology()
    if topology.degree > 1:
        dist.all_reduce(tensor, group=topology.process_group)
    return tensor


def tensor_parallel_all_gather_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    topology = get_tp_topology()
    if topology.degree == 1:
        return tensor
    pieces = [tensor.new_empty(tensor.shape) for _ in range(topology.degree)]
    dist.all_gather(pieces, tensor.contiguous(), group=topology.process_group)
    return torch.cat(pieces, dim=-1).contiguous()

