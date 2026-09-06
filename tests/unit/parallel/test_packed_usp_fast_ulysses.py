from __future__ import annotations

import torch

from chitu_diffusion.parallel.cp.packed_usp import (
    all_gather_packed_kv,
    all_to_all_packed_output,
    all_to_all_packed_qkv,
)
from chitu_diffusion.parallel.cp.topology import UspTopology


class _FakeFastUlysses:
    name = "fast_ulysses"

    def __init__(self) -> None:
        self.reset_count = 0
        self.qkv_shapes: tuple[torch.Size, ...] | None = None
        self.output_call: tuple[torch.Size, int, int] | None = None

    def reset(self) -> None:
        self.reset_count += 1

    def all_to_all_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.qkv_shapes = (query.shape, key.shape, value.shape)
        return query + 1, key + 2, value + 3

    def all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor:
        self.output_call = (tensor.shape, scatter_dim, gather_dim)
        return tensor + 4


class _FakeFastAgkv:
    name = "fast_agkv"

    def __init__(self) -> None:
        self.shapes = None

    def all_gather_kv(self, key, value):
        self.shapes = (key.shape, value.shape)
        return torch.cat((key, key + 10), dim=1), torch.cat(
            (value, value + 20), dim=1
        )


def _topology(
    transport: _FakeFastUlysses,
    agkv_transport: _FakeFastAgkv | None = None,
) -> UspTopology:
    return UspTopology(
        lane_ranks=(0, 1),
        ulysses_ranks=(0, 1),
        ring_ranks=(0,),
        ulysses_rank=0,
        ring_rank=0,
        ulysses_degree=2,
        ring_degree=1,
        ulysses_process_group=object(),
        ring_process_group=None,
        ulysses_transport=transport,
        agkv_transport=agkv_transport,
    )


def test_packed_qkv_uses_fast_ulysses_transport() -> None:
    transport = _FakeFastUlysses()
    topology = _topology(transport)
    query = torch.zeros(4, 4, 8)
    key = torch.ones_like(query)
    value = torch.full_like(query, 2)

    q_out, k_out, v_out = all_to_all_packed_qkv(
        query,
        key,
        value,
        topology=topology,
    )

    assert transport.reset_count == 1
    assert transport.qkv_shapes == (
        torch.Size([1, 4, 4, 8]),
        torch.Size([1, 4, 4, 8]),
        torch.Size([1, 4, 4, 8]),
    )
    torch.testing.assert_close(q_out, query + 1)
    torch.testing.assert_close(k_out, key + 2)
    torch.testing.assert_close(v_out, value + 3)


def test_packed_output_uses_fast_ulysses_transport() -> None:
    transport = _FakeFastUlysses()
    topology = _topology(transport)
    output = torch.zeros(8, 2, 8)

    redistributed = all_to_all_packed_output(output, topology=topology)

    assert transport.output_call == (torch.Size([1, 8, 2, 8]), 1, 2)
    torch.testing.assert_close(redistributed, output + 4)


def test_packed_kv_uses_fast_agkv_transport() -> None:
    ulysses = _FakeFastUlysses()
    agkv = _FakeFastAgkv()
    topology = _topology(ulysses, agkv)
    key = torch.zeros(4, 2, 8)
    value = torch.ones_like(key)

    gathered_key, gathered_value = all_gather_packed_kv(
        key, value, topology=topology
    )

    assert agkv.shapes == (
        torch.Size([1, 4, 2, 8]),
        torch.Size([1, 4, 2, 8]),
    )
    assert gathered_key.shape == (8, 2, 8)
    assert gathered_value.shape == (8, 2, 8)
    torch.testing.assert_close(gathered_key[4:], key + 10)
    torch.testing.assert_close(gathered_value[4:], value + 20)
