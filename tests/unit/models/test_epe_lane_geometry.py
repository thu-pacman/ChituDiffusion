from __future__ import annotations

from chitu_diffusion.epe.scheduling.cost import MeasuredStepCostModel
from chitu_diffusion.epe.scheduling.policy import EpeSchedulingPolicy
from chitu_diffusion.epe.scheduling.types import RequestProfile, SchedulableRequest
from chitu_diffusion.parallel.cp.context import EpeParallelContext
from chitu_diffusion.parallel.tp.topology import TensorParallelTopology
from chitu_diffusion.serve.config import StageServiceConfig


def _request(request_id: str) -> SchedulableRequest:
    return SchedulableRequest(
        request_id=request_id,
        submitted_at=0.0,
        deadline_at=None,
        priority=0,
        profile=RequestProfile(
            total_steps=2,
            completed_steps=0,
            image_tokens=1024,
        ),
    )


def test_tp_aware_policy_expands_cp_lanes_without_splitting_tp_groups() -> None:
    costs = MeasuredStepCostModel()
    costs.initialize(
        [
            {"image_tokens": 1024, "width": 1, "latency_ms": 10.0},
            {"image_tokens": 1024, "width": 2, "latency_ms": 100.0},
        ]
    )
    policy = EpeSchedulingPolicy(
        world_size=2,
        tp_degree=4,
        allowed_lane_widths=(1, 2),
        cost_model=costs,
    )

    plans = policy.plan_at(
        [_request("a"), _request("b")],
        pulse_steps=1,
        now_ms=0.0,
    )

    lanes = {tuple(plan.lane_ranks or ()) for plan in plans}
    assert lanes == {(0, 2, 4, 6), (1, 3, 5, 7)}
    assert all(plan.metadata["cp_width"] == 1 for plan in plans)
    assert policy.predict_step_ms(_request("probe"), 4) == 10.0


def test_tp_one_lane_geometry_degenerates_to_existing_rank_layout() -> None:
    costs = MeasuredStepCostModel()
    costs.initialize(
        [
            {"image_tokens": 1024, "width": 1, "latency_ms": 10.0},
            {"image_tokens": 1024, "width": 2, "latency_ms": 5.0},
        ]
    )
    policy = EpeSchedulingPolicy(
        world_size=2,
        allowed_lane_widths=(1, 2),
        cost_model=costs,
    )

    plan = policy.plan_at([_request("a")], pulse_steps=1, now_ms=0.0)[0]

    assert plan.lane_ranks == (0, 1)
    assert plan.metadata["cp_width"] == 2


def test_parallel_context_selects_the_callers_plane_local_lane() -> None:
    topology = TensorParallelTopology(
        ranks=(0, 2, 4, 6),
        rank=4,
        rank_in_group=2,
        degree=4,
        process_group=None,
    )
    context = EpeParallelContext(
        rank=4,
        world_size=8,
        local_rank=4,
        allowed_widths=(1, 2),
        groups={(4,): None, (5,): None, (4, 5): None},
        ulysses_topologies={},
        usp_topologies={},
        agkv_transports={},
        cfg_topologies={},
        tensor_parallel_topology=topology,
        initial_lane_ranks=(4, 5),
        owned_ulysses_transports=(),
        owned_agkv_transports=(),
        owned_groups=(),
        owns_world=False,
    )
    try:
        assert context.cp_world_size == 2
        assert context.tp_plane == 2
        assert context.cp_rank == 0
        assert context.plane_lane((0, 2, 4, 6)) == (4,)
        assert context.plane_lane(tuple(range(8))) == (4, 5)
    finally:
        context.close()


def test_parallel_context_uses_default_ulysses_degree_when_unset(
    monkeypatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    context = EpeParallelContext.from_torchrun(
        allowed_widths=(1,),
        ulysses_degree=None,
    )
    try:
        assert context.world_size == 1
        assert context.allowed_widths == (1,)
    finally:
        context.close()


def test_stage_config_validates_cp_times_tp_against_gpu_count() -> None:
    config = StageServiceConfig.from_mapping(
        {
            "name": "h3",
            "factory": "minimax-h3",
            "gpu": list(range(8)),
            "parallelism": {
                "cp": {"world_size": 2, "attention_mode": "ulysses"},
                "model": {"tensor_parallel_degree": 4},
                "scheduler": {"allowed_lane_widths": [1, 2]},
            },
            "factory_args": {"model_path": "/models/MiniMax-H3"},
        }
    )

    assert config.parallelism.cp.world_size == 2
    assert config.parallelism.model.tensor_parallel_degree == 4
    assert config.parallelism.scheduler.allowed_lane_widths == (1, 2)
