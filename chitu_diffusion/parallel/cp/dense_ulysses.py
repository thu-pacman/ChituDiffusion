"""Ulysses context parallelism for dense, possibly uneven sequences.

The packed helpers in :mod:`chitu_diffusion.parallel.cp.packed_usp` assume THD
layouts with equal shards. This module exchanges dense
``[batch, heads, sequence, head_dim]`` tensors without requiring the sequence
length to divide the context-parallel degree. Attention sees the full sequence
after the head exchange, so callers retain their full-query mask unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .ulysses_transport import UlyssesTransport


def balanced_sequence_lengths(total_length: int, degree: int) -> tuple[int, ...]:
    """Split a sequence into contiguous shards that differ by at most one token."""

    if total_length < 0:
        raise ValueError("total_length must be non-negative")
    if degree < 1:
        raise ValueError("degree must be positive")
    quotient, remainder = divmod(total_length, degree)
    return tuple(quotient + int(index < remainder) for index in range(degree))


@dataclass(frozen=True, slots=True)
class SequencePartition:
    """Contiguous, possibly uneven sequence ownership for one lane rank."""

    lengths: tuple[int, ...]
    rank: int

    def __post_init__(self) -> None:
        if not self.lengths:
            raise ValueError("a sequence partition needs at least one shard")
        if any(length < 0 for length in self.lengths):
            raise ValueError("shard lengths must be non-negative")
        if not 0 <= self.rank < len(self.lengths):
            raise ValueError("rank must index one of the shards")

    @classmethod
    def balanced(
        cls, total_length: int, *, degree: int, rank: int
    ) -> "SequencePartition":
        return cls(
            lengths=balanced_sequence_lengths(total_length, degree),
            rank=rank,
        )

    @property
    def degree(self) -> int:
        return len(self.lengths)

    @property
    def total_length(self) -> int:
        return sum(self.lengths)

    @property
    def offset(self) -> int:
        return sum(self.lengths[: self.rank])

    @property
    def local_length(self) -> int:
        return self.lengths[self.rank]

    def shard(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Select this rank's contiguous slice along ``dim``."""

        if tensor.shape[dim] != self.total_length:
            raise ValueError(
                f"tensor extent {tensor.shape[dim]} along dim {dim} does not match "
                f"partitioned length {self.total_length}"
            )
        return tensor.narrow(dim, self.offset, self.local_length)

    def gather(
        self,
        tensor: torch.Tensor,
        dim: int,
        *,
        process_group: object | None,
    ) -> torch.Tensor:
        """Restore the full sequence from uneven shards in rank order."""

        if tensor.shape[dim] != self.local_length:
            raise ValueError(
                f"tensor extent {tensor.shape[dim]} along dim {dim} does not match "
                f"local length {self.local_length}"
            )
        if self.degree == 1:
            return tensor
        if process_group is None:
            raise ValueError("gathering an uneven partition requires a process group")
        padded_length = max(self.lengths)
        padded = tensor
        if self.local_length < padded_length:
            padding = list(tensor.shape)
            padding[dim] = padded_length - self.local_length
            padded = torch.cat([tensor, tensor.new_zeros(padding)], dim=dim)
        padded = padded.contiguous()
        pieces = [torch.empty_like(padded) for _ in range(self.degree)]
        dist.all_gather(pieces, padded, group=process_group)
        return torch.cat(
            [
                piece.narrow(dim, 0, length)
                for piece, length in zip(pieces, self.lengths, strict=True)
                if length
            ],
            dim=dim,
        ).contiguous()


def _uneven_all_to_all(
    payload: torch.Tensor,
    *,
    input_splits: list[int],
    output_splits: list[int],
    process_group: object | None,
) -> torch.Tensor:
    received = payload.new_empty(sum(output_splits))
    dist.all_to_all_single(
        received,
        payload,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=process_group,
    )
    return received


def heads_to_sequence(
    tensor: torch.Tensor,
    *,
    partition: SequencePartition,
    process_group: object | None,
    transport: UlyssesTransport | None = None,
) -> torch.Tensor:
    """``[B, H, S_local, D]`` to ``[B, H / degree, S_total, D]``."""

    if tensor.ndim != 4:
        raise ValueError("dense Ulysses expects [batch, heads, sequence, head_dim]")
    degree = partition.degree
    if degree == 1:
        return tensor
    batch, heads, local_length, head_dim = tensor.shape
    if local_length != partition.local_length:
        raise ValueError(
            f"sequence extent {local_length} does not match local shard "
            f"{partition.local_length}"
        )
    if heads % degree:
        raise ValueError(
            f"head count {heads} must be divisible by context-parallel degree {degree}"
        )
    if (
        transport is not None
        and transport.name == "fast_ulysses"
    ):
        # Give every rank the same dense extent, then remove the at-most-one
        # padding token from each rank-major shard after the transpose.
        padded_length = max(partition.lengths)
        if local_length < padded_length:
            tensor = torch.cat(
                (
                    tensor,
                    tensor.new_zeros(
                        batch, heads, padded_length - local_length, head_dim
                    ),
                ),
                dim=2,
            )
        transport.reset()
        padded = transport.all_to_all(
            tensor.transpose(1, 2).contiguous(), 2, 1
        )
        shards = padded.split(padded_length, dim=1)
        return torch.cat(
            tuple(
                shard[:, :length]
                for shard, length in zip(shards, partition.lengths, strict=True)
            ),
            dim=1,
        ).transpose(1, 2).contiguous()
    local_heads = heads // degree
    unit = batch * local_heads * head_dim
    payload = (
        tensor.reshape(batch, degree, local_heads, local_length, head_dim)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
        .reshape(-1)
    )
    received = _uneven_all_to_all(
        payload,
        input_splits=[unit * local_length] * degree,
        output_splits=[unit * length for length in partition.lengths],
        process_group=process_group,
    )
    offset = 0
    shards = []
    for length in partition.lengths:
        size = unit * length
        if length:
            shards.append(
                received[offset : offset + size].reshape(
                    batch, local_heads, length, head_dim
                )
            )
        offset += size
    if not shards:
        return tensor.new_empty((batch, local_heads, 0, head_dim))
    return torch.cat(shards, dim=2).contiguous()


def sequence_to_heads(
    tensor: torch.Tensor,
    *,
    partition: SequencePartition,
    process_group: object | None,
    transport: UlyssesTransport | None = None,
) -> torch.Tensor:
    """``[B, H_local, S_total, D]`` to ``[B, H_local * degree, S_local, D]``."""

    if tensor.ndim != 4:
        raise ValueError("dense Ulysses expects [batch, heads, sequence, head_dim]")
    degree = partition.degree
    if degree == 1:
        return tensor
    batch, local_heads, total_length, head_dim = tensor.shape
    if total_length != partition.total_length:
        raise ValueError(
            f"sequence extent {total_length} does not match partitioned length "
            f"{partition.total_length}"
        )
    if (
        transport is not None
        and transport.name == "fast_ulysses"
    ):
        padded_length = max(partition.lengths)
        padded_shards = tuple(
            torch.cat(
                (
                    shard,
                    shard.new_zeros(
                        batch, local_heads, padded_length - length, head_dim
                    ),
                ),
                dim=2,
            )
            if length < padded_length
            else shard
            for shard, length in zip(
                tensor.split(list(partition.lengths), dim=2),
                partition.lengths,
                strict=True,
            )
        )
        padded = torch.cat(padded_shards, dim=2).transpose(1, 2).contiguous()
        local = transport.all_to_all(padded, 1, 2).transpose(1, 2)
        return local[:, :, : partition.local_length].contiguous()
    unit = batch * local_heads * head_dim
    shards = [
        shard.contiguous().reshape(-1)
        for shard in tensor.split(list(partition.lengths), dim=2)
        if shard.shape[2]
    ]
    payload = (
        torch.cat(shards).contiguous() if shards else tensor.new_empty(0)
    )
    received = _uneven_all_to_all(
        payload,
        input_splits=[unit * length for length in partition.lengths],
        output_splits=[unit * partition.local_length] * degree,
        process_group=process_group,
    )
    local_length = partition.local_length
    return torch.cat(
        [
            piece.reshape(batch, local_heads, local_length, head_dim)
            for piece in received.split([unit * local_length] * degree)
        ],
        dim=1,
    ).contiguous()


__all__ = [
    "SequencePartition",
    "balanced_sequence_lengths",
    "heads_to_sequence",
    "sequence_to_heads",
]
