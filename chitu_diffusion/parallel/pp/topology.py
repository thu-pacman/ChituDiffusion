from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class PipelineTopology:
    """Pipeline peers are always global ranks, including for nonzero subgroups."""

    ranks: tuple[int, ...]
    rank: int
    process_group: object | None = None

    def __post_init__(self) -> None:
        if (
            not self.ranks
            or len(set(self.ranks)) != len(self.ranks)
            or any(type(rank) is not int or rank < 0 for rank in self.ranks)
            or self.rank not in self.ranks
        ):
            raise ValueError(
                "pipeline ranks must be unique, nonnegative, and include this rank"
            )

    @property
    def degree(self) -> int:
        return len(self.ranks)

    @property
    def stage(self) -> int:
        return self.ranks.index(self.rank)

    @property
    def first(self) -> bool:
        return self.stage == 0

    @property
    def last(self) -> bool:
        return self.stage == self.degree - 1

    @property
    def previous_rank(self) -> int:
        if self.first:
            raise ValueError("the first pipeline stage has no predecessor")
        return self.ranks[self.stage - 1]

    @property
    def next_rank(self) -> int:
        if self.last:
            raise ValueError("the last pipeline stage has no successor")
        return self.ranks[self.stage + 1]


class PipelineTransport:
    """Ordered P2P with bounded live send buffers and an explicit final drain.

    Receives are posted with the *actual* patch length. Sends retain the tensor
    until completion, so overlapping compute cannot recycle its CUDA storage.
    No tags are needed (NCCL does not support them); every peer traverses the
    same patch/CFG order and drains before starting another forward.
    """

    def __init__(self, topology: PipelineTopology) -> None:
        self.topology = topology
        self._pending: deque[tuple[object, torch.Tensor]] = deque()

    def receive(
        self, shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype, device=device)
        work = dist.irecv(
            tensor, src=self.topology.previous_rank, group=self.topology.process_group
        )
        work.wait()
        return tensor

    def send(self, tensor: torch.Tensor) -> None:
        if len(self._pending) >= 2:
            work, buffer = self._pending.popleft()
            work.wait()
            del buffer
        buffer = tensor.contiguous()
        work = dist.isend(
            buffer, dst=self.topology.next_rank, group=self.topology.process_group
        )
        self._pending.append((work, buffer))

    def drain(self) -> None:
        while self._pending:
            work, buffer = self._pending.popleft()
            work.wait()
            del buffer
