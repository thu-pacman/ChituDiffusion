import pytest
import torch

from chitu_diffusion.parallel.cp import SequencePartition, gather_uneven_kv


class _Gather:
    name = "fake"

    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.key = key
        self.value = value
        self.inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def all_gather_kv(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.inputs = key, value
        return self.key, self.value

    def close(self) -> None:
        return None


def test_uneven_agkv_pads_transport_inputs_and_trims_rank_outputs() -> None:
    partition = SequencePartition(lengths=(3, 2), rank=1)
    local_key = torch.tensor([[[[10.0], [11.0]], [[20.0], [21.0]]]])
    local_value = local_key + 100
    rank0 = torch.tensor(
        [[[[0.0], [1.0]], [[2.0], [3.0]], [[4.0], [5.0]]]]
    )
    rank1_padded = torch.tensor(
        [[[[10.0], [20.0]], [[11.0], [21.0]], [[0.0], [0.0]]]]
    )
    gathered_key = torch.cat([rank0, rank1_padded], dim=1)
    transport = _Gather(gathered_key, gathered_key + 100)

    key, value = gather_uneven_kv(
        local_key,
        local_value,
        partition=partition,
        transport=transport,
    )

    assert transport.inputs is not None
    sent_key, sent_value = transport.inputs
    assert sent_key.shape == sent_value.shape == (1, 3, 2, 1)
    torch.testing.assert_close(sent_key[:, :2], local_key.transpose(1, 2))
    assert bool((sent_key[:, 2] == 0).all())
    assert key.shape == value.shape == (1, 2, 5, 1)
    torch.testing.assert_close(
        key,
        torch.tensor(
            [[[[0.0], [2.0], [4.0], [10.0], [11.0]], [[1.0], [3.0], [5.0], [20.0], [21.0]]]]
        ),
    )
    torch.testing.assert_close(value, key + 100)


def test_single_rank_agkv_returns_inputs_without_transport() -> None:
    partition = SequencePartition(lengths=(2,), rank=0)
    key = torch.randn(1, 2, 2, 4)
    value = torch.randn_like(key)
    transport = _Gather(torch.empty(0), torch.empty(0))

    result = gather_uneven_kv(
        key, value, partition=partition, transport=transport
    )

    assert result[0] is key and result[1] is value
    assert transport.inputs is None


def test_uneven_agkv_rejects_a_mismatched_local_sequence() -> None:
    partition = SequencePartition(lengths=(3, 2), rank=1)
    key = torch.randn(1, 2, 3, 4)

    with pytest.raises(ValueError, match="local partition"):
        gather_uneven_kv(
            key,
            key,
            partition=partition,
            transport=_Gather(torch.empty(0), torch.empty(0)),
        )
