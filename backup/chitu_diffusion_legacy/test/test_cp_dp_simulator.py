"""Regression tests for the CP/DP offline simulator.

These tests are intentionally torch-free. They guard the event model invariants
that matter for policy comparisons: a request cannot execute two denoise steps
at the same time, and aggregate simulated GPU busy time cannot exceed capacity.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SIM_DIR = os.path.join(ROOT, "experiments", "cp_dp_hot_switch")
for path in (ROOT, SIM_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from simulate import RequestSpec, SimulationConfig, StepLatencyProfile, simulate  # noqa: E402


def _heterogeneous_profiles() -> dict[str, StepLatencyProfile]:
    return {
        "long": StepLatencyProfile(
            name="long",
            dp_1gpu_step_ms=[100.0],
            cp_2gpu_step_ms=[80.0],
            seq_len=4096,
            resolution="long",
        ),
        "short": StepLatencyProfile(
            name="short",
            dp_1gpu_step_ms=[10.0],
            cp_2gpu_step_ms=[8.0],
            seq_len=1024,
            resolution="short",
        ),
    }


def test_static_dp_slots_do_not_overlap_steps_or_exceed_capacity():
    requests = [
        RequestSpec("long_a", 0.0, 3, "long"),
        RequestSpec("short_b", 0.0, 3, "short"),
        RequestSpec("short_c", 0.0, 3, "short"),
    ]
    result = simulate(
        requests,
        _heterogeneous_profiles(),
        "static_dp",
        SimulationConfig(total_gpus=2),
    )

    assert result.metrics["num_requests"] == 3.0
    assert result.metrics["gpu_busy_ratio"] <= 1.0 + 1e-9

    by_request = defaultdict(list)
    for event in result.trace:
        if event.event == "step":
            by_request[event.request_id].append((event.start_ms, event.end_ms, event.step_index))

    for request_id, spans in by_request.items():
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            assert current[0] >= previous[1], (request_id, previous, current)


if __name__ == "__main__":
    test_static_dp_slots_do_not_overlap_steps_or_exceed_capacity()
    print("all cp_dp simulator tests passed")
