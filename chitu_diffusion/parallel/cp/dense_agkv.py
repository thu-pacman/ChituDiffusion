"""AGKV transport for dense, possibly uneven sequence partitions."""

from __future__ import annotations

import torch

from .agkv_transport import AgkvTransport
from .dense_ulysses import SequencePartition


def gather_uneven_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    partition: SequencePartition,
    transport: AgkvTransport,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ``[B, H, S_local, D]`` K/V and remove per-rank padding.

    AGKV transports operate on equal sequence-major inputs. Dense multimodal
    sequences commonly differ by one token across ranks, so this adapter pads
    every local shard to the largest partition, performs one paired gather, and
    restores the exact rank-ordered sequence.
    """

    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError("dense AGKV expects matching [batch, heads, sequence, dim]")
    if key.shape[2] != partition.local_length:
        raise ValueError(
            f"K/V sequence length {key.shape[2]} does not match local partition "
            f"{partition.local_length}"
        )
    if partition.degree == 1:
        return key, value

    padded_length = max(partition.lengths)
    sequence_major = []
    for tensor in (key, value):
        tensor = tensor.transpose(1, 2)
        if partition.local_length < padded_length:
            padding = tensor.new_zeros(
                tensor.shape[0],
                padded_length - partition.local_length,
                tensor.shape[2],
                tensor.shape[3],
            )
            tensor = torch.cat([tensor, padding], dim=1)
        sequence_major.append(tensor.contiguous())

    full_key, full_value = transport.all_gather_kv(
        sequence_major[0], sequence_major[1]
    )
    expected = partition.degree * padded_length
    if full_key.shape[1] != expected or full_value.shape != full_key.shape:
        raise ValueError(
            "AGKV transport returned an incompatible gathered shape: "
            f"key={tuple(full_key.shape)}, value={tuple(full_value.shape)}, "
            f"expected sequence length {expected}"
        )

    gathered = []
    for tensor in (full_key, full_value):
        pieces = tensor.split(padded_length, dim=1)
        tensor = torch.cat(
            [
                piece.narrow(1, 0, length)
                for piece, length in zip(pieces, partition.lengths, strict=True)
                if length
            ],
            dim=1,
        )
        gathered.append(tensor.transpose(1, 2).contiguous())
    return gathered[0], gathered[1]


__all__ = ["gather_uneven_kv"]
