from __future__ import annotations

import pytest
import torch

from chitu_diffusion.parallel.fast_cp._runtime import (
    RingCETransportState,
    ring_ce_slot,
    ring_ce_ticket,
)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ring_ce_tickets_are_monotonic_across_epochs(world_size: int) -> None:
    tickets = [
        ring_ce_ticket(world_size, epoch, phase)
        for epoch in range(1, 4)
        for phase in range(1, world_size)
    ]
    assert tickets == list(range(1, 3 * (world_size - 1) + 1))
    assert [
        ring_ce_slot(world_size, epoch, phase)
        for epoch in range(1, 4)
        for phase in range(1, world_size)
    ] == [ticket & 1 for ticket in tickets]


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ring_ce_slot_reuse_predecessor_is_ticket_minus_two(
    world_size: int,
) -> None:
    tickets = [
        ring_ce_ticket(world_size, epoch, phase)
        for epoch in range(1, 4)
        for phase in range(1, world_size)
    ]
    for ticket in tickets[2:]:
        assert (ticket & 1) == ((ticket - 2) & 1)


def test_ring_ce_phase_zero_preserves_original_kv_identity() -> None:
    local_key = torch.empty(1)
    local_value = torch.empty(1)
    landing_key = torch.arange(2)
    landing_value = torch.arange(2) + 10
    state = RingCETransportState(
        local_key=local_key,
        local_value=local_value,
        landing_key=landing_key,
        landing_value=landing_value,
        arrivals=torch.empty(2, dtype=torch.int32),
        acks=torch.empty(2, dtype=torch.int32),
        handles=(0, 0, 0, 0, 0),
        launch_event=None,  # type: ignore[arg-type]
        key_tag="ring_k",
        value_tag="ring_v",
        state_tag="ring_state",
        epoch=1,
        world_size=4,
        device_index=0,
        slot_kv=(
            (landing_key[0], landing_value[0]),
            (landing_key[1], landing_value[1]),
        ),
    )

    key0, value0 = state.phase_kv(0)
    assert key0 is local_key
    assert value0 is local_value
    assert state.phase_kv(1) == (landing_key[1], landing_value[1])
    assert state.phase_kv(2) == (landing_key[0], landing_value[0])


@pytest.mark.parametrize(
    ("world_size", "epoch", "phase"),
    [(1, 1, 1), (3, 1, 1), (4, 0, 1), (4, 1, 0), (4, 1, 4)],
)
def test_ring_ce_ticket_rejects_invalid_protocol_coordinates(
    world_size: int,
    epoch: int,
    phase: int,
) -> None:
    with pytest.raises(ValueError, match="Flash Ring CE"):
        ring_ce_ticket(world_size, epoch, phase)
