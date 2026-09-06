import pytest
import torch

from chitu_diffusion.parallel.cp import (
    SequencePartition,
    balanced_sequence_lengths,
    heads_to_sequence,
    sequence_to_heads,
)


class _FakeFastUlysses:
    name = "fast_ulysses"

    def __init__(self) -> None:
        self.calls = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def all_to_all(self, tensor, scatter_dim, gather_dim):
        self.calls.append((tensor.shape, scatter_dim, gather_dim))
        if (scatter_dim, gather_dim) == (2, 1):
            batch, sequence, heads, head_dim = tensor.shape
            return tensor.reshape(batch, sequence, 2, heads // 2, head_dim).permute(
                0, 2, 1, 3, 4
            ).reshape(batch, 2 * sequence, heads // 2, head_dim)
        batch, sequence, heads, head_dim = tensor.shape
        return tensor.reshape(batch, 2, sequence // 2, heads, head_dim).permute(
            0, 2, 1, 3, 4
        ).reshape(
            batch, sequence // 2, 2 * heads, head_dim
        )


def test_balanced_lengths_spread_the_remainder_over_leading_ranks():
    assert balanced_sequence_lengths(9284, 4) == (2321, 2321, 2321, 2321)
    assert balanced_sequence_lengths(10, 4) == (3, 3, 2, 2)
    assert balanced_sequence_lengths(3, 4) == (1, 1, 1, 0)
    assert balanced_sequence_lengths(0, 4) == (0, 0, 0, 0)


def test_partition_offsets_tile_the_sequence_without_gaps():
    lengths = balanced_sequence_lengths(10, 4)
    offsets = [
        SequencePartition(lengths=lengths, rank=rank).offset for rank in range(4)
    ]
    assert offsets == [0, 3, 6, 8]
    assert sum(lengths) == 10


def test_shard_selects_this_ranks_contiguous_rows():
    values = torch.arange(10).reshape(1, 10, 1)
    partition = SequencePartition(lengths=(3, 3, 2, 2), rank=2)
    assert partition.shard(values, 1).flatten().tolist() == [6, 7]


def test_shard_rejects_a_mismatched_extent():
    partition = SequencePartition(lengths=(2, 2), rank=0)
    with pytest.raises(ValueError, match="does not match"):
        partition.shard(torch.zeros(1, 5, 1), 1)


def test_single_rank_exchanges_are_identity():
    partition = SequencePartition(lengths=(6,), rank=0)
    tensor = torch.randn(1, 4, 6, 8)
    torch.testing.assert_close(
        heads_to_sequence(tensor, partition=partition, process_group=None), tensor
    )
    torch.testing.assert_close(
        sequence_to_heads(tensor, partition=partition, process_group=None), tensor
    )


def test_head_exchange_requires_divisible_heads():
    partition = SequencePartition(lengths=(2, 2), rank=0)
    with pytest.raises(ValueError, match="divisible"):
        heads_to_sequence(
            torch.zeros(1, 3, 2, 4), partition=partition, process_group=object()
        )


def test_head_exchange_requires_four_dimensional_states():
    partition = SequencePartition(lengths=(2, 2), rank=0)
    with pytest.raises(ValueError, match="batch, heads, sequence, head_dim"):
        heads_to_sequence(
            torch.zeros(1, 4, 2), partition=partition, process_group=object()
        )


def test_sequence_exchange_requires_the_full_sequence():
    partition = SequencePartition(lengths=(3, 3), rank=0)
    with pytest.raises(ValueError, match="does not match partitioned length"):
        sequence_to_heads(
            torch.zeros(1, 2, 3, 4), partition=partition, process_group=object()
        )


def test_equal_dense_exchange_uses_fast_ulysses_transport():
    partition = SequencePartition(lengths=(2, 2), rank=0)
    transport = _FakeFastUlysses()
    local = torch.arange(32).reshape(1, 4, 2, 4)

    global_sequence = heads_to_sequence(
        local,
        partition=partition,
        process_group=object(),
        transport=transport,
    )
    restored = sequence_to_heads(
        global_sequence,
        partition=partition,
        process_group=object(),
        transport=transport,
    )

    assert transport.resets == 1
    assert transport.calls == [
        (torch.Size([1, 2, 4, 4]), 2, 1),
        (torch.Size([1, 4, 2, 4]), 1, 2),
    ]
    torch.testing.assert_close(restored, local)


def test_uneven_dense_exchange_pads_for_fast_ulysses():
    partition = SequencePartition(lengths=(3, 2), rank=0)
    transport = _FakeFastUlysses()

    global_sequence = heads_to_sequence(
        torch.zeros(1, 4, 3, 4),
        partition=partition,
        process_group=object(),
        transport=transport,
    )
    restored = sequence_to_heads(
        global_sequence,
        partition=partition,
        process_group=object(),
        transport=transport,
    )

    assert global_sequence.shape == (1, 2, 5, 4)
    assert restored.shape == (1, 4, 3, 4)
    assert transport.calls == [
        (torch.Size([1, 3, 4, 4]), 2, 1),
        (torch.Size([1, 6, 2, 4]), 1, 2),
    ]
