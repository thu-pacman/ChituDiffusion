from __future__ import annotations

import pytest

from chitu_diffusion.epe.scheduling.cost import MeasuredStepCostModel
from chitu_diffusion.parallel.tp.topology import (
    adopt_external_tensor_parallel_group,
    get_tp_topology,
    reset_tensor_parallel_topology,
)


def test_external_tp_group_is_adopted_without_group_creation() -> None:
    marker = object()
    topology = adopt_external_tensor_parallel_group(
        ranks=(4, 5, 6, 7),
        rank=6,
        process_group=marker,
    )
    try:
        assert topology.rank_in_group == 2
        assert topology.degree == 4
        assert get_tp_topology().process_group is marker
    finally:
        reset_tensor_parallel_topology()


def test_external_tp_group_rejects_missing_rank() -> None:
    with pytest.raises(ValueError, match="contain"):
        adopt_external_tensor_parallel_group(
            ranks=(0, 1),
            rank=2,
            process_group=object(),
        )


def test_terminal_cost_separates_lane_and_cpu_tail() -> None:
    model = MeasuredStepCostModel()
    model.initialize(
        [
            {
                "sequence_length": 1024,
                "width": 2,
                "latency_ms": 10.0,
                "terminal_gpu_ms": 20.0,
                "terminal_d2h_ms": 5.0,
                "terminal_cpu_ms": 75.0,
            }
        ]
    )
    assert model.predict_terminal_ms(sequence_length=1024, width=2) == 25.0
    assert model.predict_completion_ms(sequence_length=1024, width=2) == 100.0
