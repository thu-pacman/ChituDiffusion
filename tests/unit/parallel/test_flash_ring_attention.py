from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from chitu_diffusion.parallel.fast_cp.experimental.ring import (
    FastRingAttention,
    FastRingKVTransport,
    RingKVPhase,
    merge_flash_partials_fp32,
)


def _partial_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    return_lse: bool,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert not causal
    assert return_lse
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * scale
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, value.float())
    return output.to(query.dtype), lse


@dataclass
class _MockSession:
    shards: list[tuple[torch.Tensor, torch.Tensor]]
    rank: int
    acquired: list[int]
    released: list[int]
    closed: bool = False
    started_after: int | None = None

    @property
    def phase_count(self) -> int:
        return len(self.shards)

    def start(self) -> None:
        self.started_after = len(self.acquired)

    def acquire(self, phase_index: int) -> RingKVPhase:
        self.acquired.append(phase_index)
        source_rank = (self.rank - phase_index) % self.phase_count
        key, value = self.shards[source_rank]
        return RingKVPhase(phase_index, source_rank, key, value)

    def release(self, phase: RingKVPhase) -> None:
        self.released.append(phase.index)

    def close(self) -> None:
        self.closed = True


class _MockTransport:
    def __init__(
        self,
        shards: list[tuple[torch.Tensor, torch.Tensor]],
        rank: int = 0,
    ) -> None:
        self.rank = rank
        self.world_size = len(shards)
        self.shards = shards
        self.session: _MockSession | None = None

    def begin(self, key: torch.Tensor, value: torch.Tensor) -> _MockSession:
        local_key, local_value = self.shards[self.rank]
        assert torch.equal(key, local_key)
        assert torch.equal(value, local_value)
        self.session = _MockSession(self.shards, self.rank, [], [])
        return self.session


class _FakeCEState:
    def __init__(self, shards: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.shards = shards

    def phase_kv(self, phase: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.shards[phase]


class _FakeCERuntime:
    def __init__(self, shards: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.state = _FakeCEState(shards)
        self.calls: list[tuple[str, int]] = []

    def begin_ring_ce_transport(self, key, value, **kwargs):
        self.calls.append(("begin", kwargs["epoch"]))
        return self.state

    def wait_ring_ce_ready(self, state, *, phase: int):
        self.calls.append(("wait", phase))
        return state.phase_kv(phase)

    def forward_ring_ce(self, state, *, phase: int):
        self.calls.append(("forward", phase))

    def publish_ring_ce_consumed(self, state, *, phase: int):
        self.calls.append(("consumed", phase))


def _reference(
    query: torch.Tensor,
    shards: list[tuple[torch.Tensor, torch.Tensor]],
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.cat([shard[0] for shard in shards], dim=1)
    value = torch.cat([shard[1] for shard in shards], dim=1)
    return _partial_attention(
        query,
        key,
        value,
        causal=False,
        return_lse=True,
        softmax_scale=softmax_scale,
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_ring_matches_global_attention_cpu(
    world_size: int,
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(1000 + world_size)
    query = torch.randn(2, 5, 3, 16, generator=generator).to(dtype)
    shards = [
        (
            torch.randn(2, rank % 3 + 1, 3, 16, generator=generator).to(dtype),
            torch.randn(2, rank % 3 + 1, 3, 16, generator=generator).to(dtype),
        )
        for rank in range(world_size)
    ]
    transport = _MockTransport(shards, rank=world_size // 2)
    attention = FastRingAttention(transport, flash_attn_fwd=_partial_attention)

    output, lse = attention(
        query,
        *shards[transport.rank],
        return_lse=True,
    )
    reference_output, reference_lse = _reference(query, shards)

    torch.testing.assert_close(
        output.float(), reference_output.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(lse, reference_lse, atol=2e-5, rtol=2e-5)
    assert output.dtype == dtype
    assert lse.dtype == torch.float32
    assert transport.session is not None
    assert transport.session.acquired == list(range(world_size))
    assert transport.session.released == list(range(world_size))
    assert transport.session.closed
    # The traversal is submitted once phase zero is queued and never earlier.
    assert transport.session.started_after == 1


def test_custom_softmax_scale_is_applied_to_every_phase() -> None:
    query = torch.randn(1, 3, 2, 8).to(torch.float16)
    shards = [
        (torch.randn(1, 2, 2, 8).half(), torch.randn(1, 2, 2, 8).half())
        for _ in range(2)
    ]
    transport = _MockTransport(shards)
    attention = FastRingAttention(transport, flash_attn_fwd=_partial_attention)

    output = attention(query, *shards[0], softmax_scale=0.125)
    reference, _ = _reference(query, shards, softmax_scale=0.125)

    torch.testing.assert_close(output.float(), reference.float(), atol=2e-2, rtol=2e-2)


def test_merge_initializes_and_updates_fp32_hbm_state() -> None:
    output_a = torch.tensor([[[[1.0, 3.0]]]], dtype=torch.float16)
    output_b = torch.tensor([[[[5.0, 7.0]]]], dtype=torch.float16)
    lse_a = torch.tensor([[[0.0]]])
    lse_b = torch.tensor([[[0.0]]])

    running_output, running_lse = merge_flash_partials_fp32(None, None, output_a, lse_a)
    assert running_output.dtype == torch.float32
    assert running_lse.dtype == torch.float32

    merged_output, merged_lse = merge_flash_partials_fp32(
        running_output, running_lse, output_b, lse_b
    )
    torch.testing.assert_close(
        merged_output, torch.tensor([[[[3.0, 5.0]]]]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        merged_lse, torch.tensor([[[torch.log(torch.tensor(2.0))]]])
    )


def test_rejects_causal_and_unsupported_cp() -> None:
    with pytest.raises(ValueError, match="CP2, CP4, or CP8"):
        FastRingAttention(_MockTransport([(torch.empty(0), torch.empty(0))]))

    shards = [
        (torch.randn(1, 2, 1, 8).half(), torch.randn(1, 2, 1, 8).half())
        for _ in range(2)
    ]
    attention = FastRingAttention(
        _MockTransport(shards), flash_attn_fwd=_partial_attention
    )
    with pytest.raises(NotImplementedError, match="non-causal"):
        attention(shards[0][0], *shards[0], causal=True)


def test_fast_ce_adapter_overlaps_forward_and_gates_slot_reuse() -> None:
    shards = [
        (torch.full((1,), rank), torch.full((1,), rank + 10)) for rank in range(4)
    ]
    runtime = _FakeCERuntime(shards)
    transport = FastRingKVTransport(runtime, rank=0, world_size=4)
    session = transport.begin(*shards[0])

    # Nothing is submitted until phase zero has been queued by the compute side.
    assert runtime.calls == []
    for phase in range(4):
        lease = session.acquire(phase)
        assert lease.source_rank == (-phase) % 4
        if phase == 0:
            session.start()
        session.release(lease)
    session.close()

    assert runtime.calls == [
        ("begin", 1),
        ("wait", 1),
        ("consumed", 1),
        ("wait", 2),
        ("consumed", 2),
        ("wait", 3),
        ("consumed", 3),
    ]


def test_fast_ce_adapter_drains_before_epoch_wrap(monkeypatch) -> None:
    shards = [
        (torch.full((1,), rank), torch.full((1,), rank + 10)) for rank in range(2)
    ]
    runtime = _FakeCERuntime(shards)
    transport = FastRingKVTransport(runtime, rank=0, world_size=2)
    transport._epoch = transport._max_epoch
    synchronized: list[torch.device] = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronized.append(torch.device(device)),
    )

    transport.begin(*shards[0]).start()

    assert synchronized == [shards[0][0].device]
    assert runtime.calls == [("begin", 1)]
