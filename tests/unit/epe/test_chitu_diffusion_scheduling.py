from __future__ import annotations

import threading

import pytest

from chitu_diffusion.epe import (
    EpeSchedulingPolicy,
    FullWorldLaneWorkerPool,
    LaneConstraints,
    LaneLease,
    LaneReport,
    LaneTask,
    LaneWorker,
    LaneWorkItem,
    LaneWorkResult,
    MeasuredStepCostModel,
    MeasuredTransferCostModel,
    PulseCoordinator,
    PulseLaneBroker,
    RequestProfile,
    RuntimeCostCalibrator,
    SchedulableRequest,
    SingletonLaneWorkerPool,
    StepOutcome,
)
from chitu_diffusion.serve import HotSwitchPoolConfig
from chitu_diffusion.serve.diffusion_runtime import EpeDiffusionServiceRuntime


def test_singleton_lane_pool_pulls_next_work_without_a_pulse() -> None:
    pending = [LaneWorkItem("first", {}), LaneWorkItem("second", {})]
    completed: list[str] = []
    lock = threading.Lock()

    def claim(rank: int) -> LaneWorkItem | None:
        assert rank == 0
        with lock:
            return None if not pending else pending.pop(0)

    def execute(item: LaneWorkItem, rank: int) -> LaneWorkResult:
        assert rank == 0
        return LaneWorkResult(item.request_id, output=item.request_id.upper())

    def complete(result: LaneWorkResult) -> None:
        completed.append(result.request_id)

    SingletonLaneWorkerPool(
        rank=0,
        world_size=1,
        control_group=lambda _peer: None,
        claim=claim,
        execute=execute,
        complete=complete,
        should_stop=lambda: len(completed) == 2,
        idle_sleep_s=0.001,
    ).run()

    assert completed == ["first", "second"]


def test_full_world_lane_pool_pulls_next_work_without_a_pulse() -> None:
    pending = [LaneWorkItem("first", {}), LaneWorkItem("second", {})]
    completed: list[str] = []

    FullWorldLaneWorkerPool(
        rank=0,
        world_size=1,
        control_group=None,
        claim=lambda _rank: None if not pending else pending.pop(0),
        execute=lambda item, ranks: LaneWorkResult(
            item.request_id,
            output=ranks,
        ),
        complete=lambda result: completed.append(result.request_id),
        should_stop=lambda: len(completed) == 2,
        idle_sleep_s=0.001,
    ).run()

    assert completed == ["first", "second"]


def _measured_cost_model() -> MeasuredStepCostModel:
    model = MeasuredStepCostModel()
    model.initialize(
        [
            {"image_tokens": 1024, "width": 1, "latency_ms": 90},
            {"image_tokens": 1024, "width": 2, "latency_ms": 100},
            {"image_tokens": 1024, "width": 4, "latency_ms": 110},
            {"image_tokens": 4096, "width": 1, "latency_ms": 1000},
            {"image_tokens": 4096, "width": 2, "latency_ms": 520},
            {"image_tokens": 4096, "width": 4, "latency_ms": 300},
            {"image_tokens": 9216, "width": 1, "latency_ms": 3200},
            {"image_tokens": 9216, "width": 2, "latency_ms": 1500},
            {"image_tokens": 9216, "width": 4, "latency_ms": 780},
        ]
    )
    return model


def _schedulable(
    request_id: str,
    *,
    image_tokens: int,
    total_steps: int = 50,
    completed_steps: int = 0,
    deadline_at: float | None = None,
    priority: int = 0,
    admitted: bool = True,
    current_ranks: tuple[int, ...] | None = None,
) -> SchedulableRequest:
    return SchedulableRequest(
        request_id=request_id,
        submitted_at=0.0,
        deadline_at=deadline_at,
        priority=priority,
        profile=RequestProfile(
            total_steps=total_steps,
            completed_steps=completed_steps,
            image_tokens=image_tokens,
        ),
        admitted=admitted,
        current_ranks=current_ranks,
    )


def test_measured_cost_model_accepts_existing_image_tokens_rows() -> None:
    model = _measured_cost_model()

    assert model.predict_step_ms(sequence_length=4096, width=2) == 520
    assert model.snapshot()["source"] == "startup_warmup"
    with pytest.raises(RuntimeError, match="warmup"):
        MeasuredStepCostModel().predict_step_ms(sequence_length=4096, width=2)


def test_measured_cost_model_tracks_terminal_and_transfer_costs() -> None:
    model = MeasuredStepCostModel()
    model.initialize(
        [
            {
                "image_tokens": 1024,
                "width": 2,
                "latency_ms": 40,
                "terminal_ms": 65,
                "vae_latency_ms": 60,
                "d2h_latency_ms": 5,
            }
        ]
    )
    transfers = MeasuredTransferCostModel()
    transfers.initialize(
        [{"bytes": 4096, "source": 0, "destination": 1, "latency_ms": 3}]
    )

    assert model.predict_terminal_ms(sequence_length=1024, width=2) == 65
    assert transfers.predict_transfer_ms(bytes=8192, source=0, destination=1) == 6


def test_terminal_cost_lets_other_lane_fill_finalize_tail() -> None:
    model = MeasuredStepCostModel()
    model.initialize(
        [
            {
                "image_tokens": 1024,
                "width": 1,
                "latency_ms": 10,
                "terminal_ms": 50,
            },
            {
                "image_tokens": 4096,
                "width": 1,
                "latency_ms": 20,
                "terminal_ms": 0,
            },
        ]
    )
    policy = EpeSchedulingPolicy(
        world_size=2,
        allowed_lane_widths=(1, 2),
        cost_model=model,
        strategy="static_dp",
    )

    plans = policy.plan_at(
        [
            _schedulable(
                "finishing",
                image_tokens=1024,
                total_steps=10,
                completed_steps=9,
            ),
            _schedulable("running", image_tokens=4096, total_steps=20),
        ],
        pulse_steps=5,
        now_ms=0,
    )
    by_id = {plan.request_id: plan for plan in plans}

    assert by_id["finishing"].steps == 1
    assert by_id["finishing"].metadata["predicted_terminal_ms"] == 50
    assert by_id["running"].steps == 3
    assert max(plan.metadata["predicted_phase_ms"] for plan in plans) == 60


def test_epe_policy_uses_sequence_cost_and_balanced_k() -> None:
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        switch_allowed_until_step=20,
    )

    plans = policy.plan(
        [
            _schedulable("short", image_tokens=1024, total_steps=100),
            _schedulable("long", image_tokens=9216, total_steps=100),
        ],
        pulse_steps=4,
    )
    by_id = {plan.request_id: plan for plan in plans}

    assert set(by_id) == {"short", "long"}
    assert by_id["short"].steps > by_id["long"].steps
    assert (
        by_id["short"].metadata["predicted_step_ms"]
        < (by_id["long"].metadata["predicted_step_ms"])
    )
    assert set(by_id["short"].lane_ranks).isdisjoint(by_id["long"].lane_ranks)


def test_epe_normalizes_throughput_across_mixed_sequence_lengths() -> None:
    model = MeasuredStepCostModel()
    model.initialize(
        [
            {"image_tokens": 1024, "width": 1, "latency_ms": 241},
            {"image_tokens": 1024, "width": 2, "latency_ms": 166},
            {"image_tokens": 1024, "width": 4, "latency_ms": 109},
            {"image_tokens": 16384, "width": 1, "latency_ms": 5311},
            {"image_tokens": 16384, "width": 2, "latency_ms": 2790},
            {"image_tokens": 16384, "width": 4, "latency_ms": 1491},
        ]
    )
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=model,
        switch_allowed_until_step=20,
    )

    plans = policy.plan_at(
        [
            _schedulable("small", image_tokens=1024, total_steps=12),
            _schedulable("large", image_tokens=16384, total_steps=12),
        ],
        pulse_steps=5,
        now_ms=0.0,
    )

    # Raw steps/s would spend all four GPUs on the small request. The
    # efficiency-aware score keeps both mixed-size requests moving instead.
    assert {plan.request_id: len(plan.lane_ranks) for plan in plans} == {
        "small": 2,
        "large": 2,
    }


def test_epe_widens_only_when_remaining_compute_repays_migration() -> None:
    costs = MeasuredStepCostModel()
    costs.initialize(
        [
            {"image_tokens": 1024, "width": 1, "latency_ms": 10},
            {"image_tokens": 1024, "width": 2, "latency_ms": 6},
            {"image_tokens": 1024, "width": 4, "latency_ms": 4},
        ]
    )
    transfers = MeasuredTransferCostModel()
    transfers.initialize(
        [
            {"bytes": 100, "source": 0, "destination": 2, "latency_ms": 15},
            {"bytes": 100, "source": 0, "destination": 3, "latency_ms": 15},
        ]
    )

    def plan(remaining_steps: int):
        policy = EpeSchedulingPolicy(
            world_size=4,
            allowed_lane_widths=(1, 2, 4),
            cost_model=costs,
            transfer_cost_model=transfers,
            switch_allowed_until_step=6,
        )
        request = SchedulableRequest(
            request_id="tail",
            submitted_at=0.0,
            deadline_at=None,
            priority=0,
            profile=RequestProfile(
                total_steps=remaining_steps + 5,
                completed_steps=5,
                image_tokens=1024,
                attributes={"state_bytes": 100},
            ),
            current_ranks=(0, 1),
        )
        return policy.plan_at([request], pulse_steps=2, now_ms=0.0)[0]

    short = plan(10)
    assert short.metadata["cp_width"] == 2

    long = plan(50)
    assert long.metadata["cp_width"] == 4
    assert long.metadata["predicted_transfer_ms"] == 30
    assert long.metadata["predicted_resize_gain_ms"] == 70


def test_epe_keeps_lane_after_switch_cutoff() -> None:
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        switch_allowed_until_step=4,
    )

    plan = policy.plan_at(
        [
            _schedulable(
                "running",
                image_tokens=9216,
                completed_steps=5,
                current_ranks=(0, 1),
            )
        ],
        pulse_steps=2,
        now_ms=0.0,
    )[0]

    assert plan.lane_ranks == (0, 1)
    assert plan.metadata["switched"] is False


def test_epe_keeps_lane_at_switch_cutoff_boundary() -> None:
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        switch_allowed_until_step=4,
    )

    plan = policy.plan_at(
        [
            _schedulable(
                "running",
                image_tokens=9216,
                completed_steps=4,
                current_ranks=(0, 1),
            )
        ],
        pulse_steps=2,
        now_ms=0.0,
    )[0]

    assert plan.lane_ranks == (0, 1)
    assert plan.metadata["switched"] is False


def test_epe_does_not_relocate_an_existing_same_width_lane() -> None:
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(2,),
        cost_model=_measured_cost_model(),
        switch_allowed_until_step=50,
    )

    plan = policy.plan_at(
        [
            _schedulable(
                "running",
                image_tokens=4096,
                completed_steps=1,
                current_ranks=(2, 3),
            )
        ],
        pulse_steps=2,
        now_ms=0.0,
    )[0]

    assert plan.lane_ranks == (2, 3)
    assert plan.metadata["predicted_transfer_ms"] == 0


def test_epe_resize_hysteresis_requires_net_widening_gain() -> None:
    costs = MeasuredStepCostModel()
    costs.initialize(
        [
            {"image_tokens": 1024, "width": 1, "latency_ms": 10},
            {"image_tokens": 1024, "width": 2, "latency_ms": 6},
            {"image_tokens": 1024, "width": 4, "latency_ms": 4},
        ]
    )
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=costs,
        switch_allowed_until_step=50,
        min_resize_gain_ms=100.0,
        resize_hysteresis_ms=1.0,
        resize_control_cost_ms=10.0,
    )

    plan = policy.plan_at(
        [
            _schedulable(
                "tail",
                image_tokens=1024,
                total_steps=25,
                completed_steps=5,
                current_ranks=(0, 1),
            )
        ],
        pulse_steps=2,
        now_ms=0.0,
    )[0]

    # Gross compute saving is 40ms and cannot repay the control cost plus
    # configured anti-churn threshold.
    assert plan.lane_ranks == (0, 1)
    assert plan.metadata["switched"] is False


def test_epe_narrowing_must_repay_control_cost() -> None:
    costs = MeasuredStepCostModel()
    costs.initialize(
        [
            {"image_tokens": 1024, "width": 2, "latency_ms": 5},
            {"image_tokens": 1024, "width": 4, "latency_ms": 4},
        ]
    )
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(2, 4),
        cost_model=costs,
        switch_allowed_until_step=50,
        resize_control_cost_ms=1_000.0,
    )

    plans = policy.plan_at(
        [
            _schedulable(
                "running",
                image_tokens=1024,
                total_steps=20,
                current_ranks=(0, 1, 2, 3),
            ),
            _schedulable(
                "pending",
                image_tokens=1024,
                total_steps=20,
                admitted=False,
            ),
        ],
        pulse_steps=2,
        now_ms=0.0,
    )

    assert [(plan.request_id, plan.lane_ranks) for plan in plans] == [
        ("running", (0, 1, 2, 3))
    ]


def test_runtime_extracts_actual_request_priority() -> None:
    class Request:
        priority = 7

    assert EpeDiffusionServiceRuntime._request_priority(Request()) == 7
    assert EpeDiffusionServiceRuntime._request_priority({"priority": 9}) == 9


def test_dense_epe_layout_search_is_bounded() -> None:
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        switch_allowed_until_step=20,
        max_active_requests=4,
        max_layout_candidates=64,
    )

    plans = policy.plan_at(
        [
            _schedulable(
                f"request-{index}",
                image_tokens=(1024, 4096, 9216)[index % 3],
                total_steps=50,
                admitted=False,
            )
            for index in range(12)
        ],
        pulse_steps=5,
        now_ms=0.0,
    )

    assert plans
    assert policy.last_plan_stats["queue_size"] == 12
    assert policy.last_plan_stats["layout_request_count"] == 8
    assert 1 <= policy.last_plan_stats["candidate_count"] <= 64


def test_online_calibration_is_exact_keyed_and_tolerates_small_error() -> None:
    model = MeasuredStepCostModel()
    model.initialize(
        [
            {"image_tokens": 1024, "width": 1, "latency_ms": 100},
            {"image_tokens": 4096, "width": 1, "latency_ms": 100},
        ]
    )
    calibrator = RuntimeCostCalibrator(
        model,
        enabled=True,
        warmup_skip=0,
        tolerance=0.05,
    )

    calibrator.observe(
        image_tokens=1024,
        width=1,
        batch_size=1,
        cfg_conditions=1,
        measured_step_ms=50,
    )
    assert calibrator.factor(1024, 1, 1, 1) == pytest.approx(0.5)
    assert calibrator.factor(4096, 1, 1, 1) == pytest.approx(1.0)

    calibrator.observe(
        image_tokens=4096,
        width=1,
        batch_size=1,
        cfg_conditions=1,
        measured_step_ms=103,
    )
    assert calibrator.factor(4096, 1, 1, 1) == pytest.approx(1.0)


def test_hot_switch_pool_requires_three_warmup_steps() -> None:
    with pytest.raises(ValueError, match="warmup_steps must be >= 3"):
        HotSwitchPoolConfig(warmup_steps=2)


def test_epe_uses_fair_lane_share_for_future_queue_prediction() -> None:
    model = MeasuredStepCostModel()
    model.initialize(
        [
            {"image_tokens": 1024, "width": 1, "latency_ms": 90},
            {"image_tokens": 1024, "width": 2, "latency_ms": 60},
            {"image_tokens": 1024, "width": 4, "latency_ms": 40},
        ]
    )
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=model,
    )
    request = policy._convert(_schedulable("small", image_tokens=1024))

    assert (
        policy._future_lane_width(
            request,
            free_ranks=4,
            waiting_count=1,
            start_ms=0.0,
        )
        == 4
    )
    assert (
        policy._future_lane_width(
            request,
            free_ranks=4,
            waiting_count=2,
            start_ms=0.0,
        )
        == 2
    )


def test_epe_pulse_stops_when_the_nearest_request_completes() -> None:
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1,),
        cost_model=_measured_cost_model(),
        balanced_k=True,
    )
    plans = policy.plan(
        [
            _schedulable(
                "short-tail",
                image_tokens=1024,
                total_steps=12,
                completed_steps=11,
            ),
            _schedulable("long-tail", image_tokens=4096),
        ],
        pulse_steps=5,
    )
    by_id = {plan.request_id: plan for plan in plans}

    assert by_id["short-tail"].steps == 1
    assert by_id["long-tail"].steps == 1


def test_deadline_guard_reserves_slack_from_throughput_work() -> None:
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        deadline_guard_ms=1_000.0,
    )
    urgent = SchedulableRequest(
        request_id="urgent",
        submitted_at=0.0,
        deadline_at=5.0,
        priority=0,
        profile=RequestProfile(
            total_steps=12,
            completed_steps=0,
            image_tokens=4096,
        ),
    )

    plans = policy.plan_at(
        [urgent, _schedulable("normal", image_tokens=1024, total_steps=12)],
        pulse_steps=5,
        now_ms=0.0,
    )

    assert len(plans) == 1
    assert plans[0].request_id == "urgent"
    assert plans[0].lane_ranks == (0, 1, 2, 3)


def test_slo_predictor_sees_pending_queue_beyond_admission_limit() -> None:
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        max_active_requests=2,
    )

    plans = policy.plan_at(
        [
            _schedulable("short", image_tokens=1024, total_steps=12),
            _schedulable("long", image_tokens=9216, total_steps=12),
            _schedulable(
                "urgent-pending",
                image_tokens=4096,
                total_steps=12,
                deadline_at=5.0,
                admitted=False,
            ),
        ],
        pulse_steps=5,
        now_ms=0.0,
    )

    # Both active-state slots are occupied, so the pending request cannot be
    # admitted yet. The planner runs the short active request first so the
    # urgent request can enter before its predicted SLO.
    assert [(plan.request_id, plan.lane_ranks) for plan in plans] == [("short", (0,))]


def test_hot_switch_pool_parses_default_request_slo() -> None:
    pool = HotSwitchPoolConfig.from_mapping(
        {
            "allowed_lane_widths": [1, 2, 4],
            "default_deadline_ms": 25_000,
        },
        world_size=4,
    )

    assert pool.default_deadline_ms == 25_000
    with pytest.raises(ValueError, match="default_deadline_ms"):
        HotSwitchPoolConfig.from_mapping(
            {
                "allowed_lane_widths": [1, 2, 4],
                "default_deadline_ms": 0,
            },
            world_size=4,
        )


def test_hot_switch_pool_parses_resize_controls() -> None:
    pool = HotSwitchPoolConfig.from_mapping(
        {
            "allowed_lane_widths": [1, 2, 4],
            "min_resize_gain_ms": 25,
            "resize_hysteresis_ms": 10,
            "resize_control_cost_ms": 3,
        },
        world_size=4,
    )

    assert pool.min_resize_gain_ms == 25
    assert pool.resize_hysteresis_ms == 10
    assert pool.resize_control_cost_ms == 3
    with pytest.raises(ValueError, match="resize cost"):
        HotSwitchPoolConfig.from_mapping(
            {
                "allowed_lane_widths": [1, 2, 4],
                "min_resize_gain_ms": -1,
            },
            world_size=4,
        )


@pytest.mark.parametrize(
    ("strategy", "expected_widths"),
    [
        ("static_dp", (1,)),
        ("elastic", (1, 2, 4)),
        ("static_cp", (4,)),
    ],
)
def test_lane_constraints_make_static_strategies_elastic_extremes(
    strategy, expected_widths
) -> None:
    constraints = LaneConstraints.create(
        strategy,
        world_size=4,
        elastic_widths=(1, 2, 4),
    )

    assert constraints.allowed_widths == expected_widths
    assert constraints.pin_running_lanes is (strategy == "static_dp")


def test_static_dp_and_cp_use_the_shared_cost_aware_planner() -> None:
    requests = [_schedulable(str(index), image_tokens=4096) for index in range(4)]
    static_dp = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        strategy="static_dp",
    )
    static_cp = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        strategy="static_cp",
    )

    dp_plans = static_dp.plan(requests, pulse_steps=5)
    cp_plans = static_cp.plan(requests, pulse_steps=5)

    assert len(dp_plans) == 4
    assert all(len(plan.lane_ranks) == 1 for plan in dp_plans)
    assert len(cp_plans) == 1
    assert cp_plans[0].lane_ranks == (0, 1, 2, 3)


def test_pulse_leases_cover_world_and_require_all_lane_reports() -> None:
    now_ms = 10_000.0
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        strategy="elastic",
    )
    coordinator = PulseCoordinator(
        world_size=4,
        policy=policy,
        pulse_steps=5,
        clock_ms=lambda: now_ms,
    )
    plan = coordinator.open(
        [
            _schedulable("short", image_tokens=1024),
            _schedulable("long", image_tokens=9216),
        ]
    )

    assert plan.epoch == 0
    assert plan.ranks == (0, 1, 2, 3)
    assert plan.deadline_ms > now_ms
    with pytest.raises(RuntimeError, match="reports"):
        coordinator.open([])

    for lease in plan.leases:
        coordinator.report(
            LaneReport(
                epoch=plan.epoch,
                ranks=lease.ranks,
                request_id=lease.request_id,
                current_step=(None if lease.request_id is None else lease.min_steps),
                state_owner_rank=(None if lease.request_id is None else lease.ranks[0]),
            )
        )

    assert coordinator.ready_to_replan
    assert len(coordinator.reports()) == len(plan.leases)
    assert coordinator.open([]).epoch == 1


def test_lane_can_pull_slo_ordered_work_before_pulse_deadline() -> None:
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        strategy="static_dp",
    )
    coordinator = PulseCoordinator(
        world_size=4,
        policy=policy,
        pulse_steps=5,
        clock_ms=lambda: 1_000.0,
    )
    plan = coordinator.open([_schedulable("running", image_tokens=9216)])
    lease = next(lease for lease in plan.leases if lease.request_id == "running")
    urgent = SchedulableRequest(
        request_id="urgent",
        submitted_at=2.0,
        deadline_at=3.0,
        priority=0,
        profile=RequestProfile(
            total_steps=10,
            completed_steps=0,
            image_tokens=1024,
        ),
    )
    normal = _schedulable("normal", image_tokens=1024)

    selected = coordinator.choose_pull_request(
        lease,
        [normal, urgent],
        now_ms=1_100.0,
    )

    assert selected is urgent
    assert (
        coordinator.choose_pull_request(
            lease,
            [normal],
            now_ms=plan.deadline_ms,
        )
        is None
    )


def test_tp2_cp1_lease_uses_cp_width_for_pull_cost() -> None:
    policy = EpeSchedulingPolicy(
        world_size=4,
        tp_degree=2,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        strategy="elastic",
    )
    coordinator = PulseCoordinator(
        world_size=8,
        policy=policy,
        pulse_steps=5,
        clock_ms=lambda: 1_000.0,
    )
    plan = coordinator.open(
        [
            _schedulable(f"running-{index}", image_tokens=1024)
            for index in range(4)
        ]
    )
    lease = next(lease for lease in plan.leases if lease.request_id == "running-0")

    assert lease.width == 2
    assert lease.scheduling_width == 1
    assert (
        coordinator.choose_pull_request(
            lease,
            [_schedulable("pending", image_tokens=1024)],
            now_ms=plan.opened_at_ms,
        )
        is not None
    )


def test_pulse_lane_broker_pulls_work_without_waiting_for_other_lanes() -> None:
    now_ms = 1_000.0
    policy = EpeSchedulingPolicy(
        world_size=4,
        allowed_lane_widths=(1, 2, 4),
        cost_model=_measured_cost_model(),
        strategy="static_dp",
        clock_ms=lambda: now_ms,
    )
    coordinator = PulseCoordinator(
        world_size=4,
        policy=policy,
        pulse_steps=5,
        clock_ms=lambda: now_ms,
    )
    requests = {
        f"req{index}": _schedulable(f"req{index}", image_tokens=1024, total_steps=5)
        for index in range(5)
    }
    status = {request_id: "pending" for request_id in requests}
    placements: dict[str, tuple[int, ...]] = {}
    completed: list[str] = []
    observed: list[tuple[str, tuple[int, ...], float]] = []

    def schedulable(include_running: bool):
        output = []
        for request_id, request in requests.items():
            if status[request_id] == "pending" or (
                include_running and status[request_id] == "running"
            ):
                output.append(
                    SchedulableRequest(
                        request_id=request.request_id,
                        submitted_at=request.submitted_at,
                        deadline_at=request.deadline_at,
                        priority=request.priority,
                        profile=request.profile,
                        current_ranks=placements.get(request_id),
                    )
                )
        return output

    def reserve(request_id: str, ranks: tuple[int, ...]) -> None:
        status[request_id] = "running"
        placements[request_id] = ranks

    def complete(result: LaneWorkResult) -> None:
        status[result.request_id] = "completed"
        completed.append(result.request_id)

    broker = PulseLaneBroker(
        world_size=4,
        coordinator=coordinator,
        schedulable=schedulable,
        reserve=reserve,
        payload=lambda request_id: {"request_id": request_id},
        update_progress=lambda request_id, _step, ranks: placements.__setitem__(
            request_id, ranks
        ),
        retire=lambda _request_id: None,
        complete=complete,
        observe=lambda request, ranks, step_ms: observed.append(
            (request.request_id, ranks, step_ms)
        ),
        should_stop=lambda: False,
        clock_ms=lambda: now_ms,
    )
    responses: dict[int, dict] = {}

    def open_rank(rank: int) -> None:
        responses[rank] = broker.exchange(
            rank,
            {"kind": "pulse_report", "epoch": -1, "sync": 0, "report": None},
        )

    threads = [threading.Thread(target=open_rank, args=(rank,)) for rank in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    dispatch = responses[0]
    assert all(response == dispatch for response in responses.values())
    rank0_lease = next(lease for lease in dispatch["leases"] if 0 in lease["ranks"])
    first_request = rank0_lease["request_id"]
    response = broker.exchange(
        0,
        {
            "kind": "lane_progress",
            "epoch": dispatch["epoch"],
            "sync": 0,
            "ranks": rank0_lease["ranks"],
            "request_id": first_request,
            "current_step": 5,
            "result": LaneWorkResult(first_request, output=b"done"),
            "observed_step_ms": 12.5,
            "error": None,
        },
    )

    assert completed == [first_request]
    assert observed == [(first_request, tuple(rank0_lease["ranks"]), 12.5)]
    assert response["action"] == "run"
    assert response["request_id"] == "req4"


def test_pulse_lane_broker_retires_completion_in_global_report() -> None:
    now_ms = 1_000.0
    policy = EpeSchedulingPolicy(
        world_size=1,
        allowed_lane_widths=(1,),
        cost_model=_measured_cost_model(),
        strategy="static_dp",
        clock_ms=lambda: now_ms,
    )
    coordinator = PulseCoordinator(
        world_size=1,
        policy=policy,
        pulse_steps=1,
        clock_ms=lambda: now_ms,
    )
    request = _schedulable("finishing", image_tokens=1024, total_steps=1)
    status = "pending"
    completed = []
    observed = []

    def schedulable(_include_running: bool):
        return [request] if status in {"pending", "running"} else []

    def reserve(_request_id: str, _ranks: tuple[int, ...]) -> None:
        nonlocal status
        status = "running"

    def complete(result: LaneWorkResult) -> None:
        nonlocal status
        status = "completed"
        completed.append(result.request_id)

    def retire(_request_id: str) -> None:
        nonlocal status
        status = "retired"

    broker = PulseLaneBroker(
        world_size=1,
        coordinator=coordinator,
        schedulable=schedulable,
        reserve=reserve,
        payload=lambda request_id: {"request_id": request_id},
        update_progress=lambda _request_id, _step, _ranks: None,
        retire=retire,
        complete=complete,
        observe=lambda item, ranks, step_ms: observed.append(
            (item.request_id, ranks, step_ms)
        ),
        should_stop=lambda: False,
        clock_ms=lambda: now_ms,
    )
    dispatch = broker.exchange(
        0,
        {"kind": "pulse_report", "epoch": -1, "sync": 0, "report": None},
    )
    result = LaneWorkResult("finishing", output=b"done")

    response = broker.exchange(
        0,
        {
            "kind": "pulse_report",
            "epoch": dispatch["epoch"],
            "sync": 1,
            "report": {
                "request_id": None,
                "current_step": None,
                "state_owner_rank": None,
                "completed_request_ids": ["finishing"],
                "observed_step_ms": 12.5,
                "error": None,
                "finished_request_id": "finishing",
            },
        },
    )

    assert response == {
        "action": "idle",
        "retired_request_ids": ["finishing"],
    }
    assert completed == []
    assert observed == [("finishing", (0,), 12.5)]

    result_response = broker.exchange(
        0,
        {
            "kind": "lane_result",
            "epoch": dispatch["epoch"],
            "sync": 1,
            "ranks": [0],
            "result": result,
        },
    )
    assert result_response == {
        "action": "result_ack",
        "request_id": "finishing",
    }
    assert completed == ["finishing"]


class _FakeLaneHooks:
    def __init__(self, clock, remaining, pending=()):
        self.clock = clock
        self.remaining = dict(remaining)
        self.pending = list(pending)
        self.completed = []

    def execute_step(self, task, ranks):
        assert ranks
        self.remaining[task.request_id] -= 1
        self.clock[0] += task.predicted_step_ms
        return StepOutcome(
            current_step=task.current_step + 1,
            completed=self.remaining[task.request_id] == 0,
            observed_step_ms=task.predicted_step_ms,
        )

    def complete_request(self, task):
        self.completed.append(task.request_id)

    def pull_request(self, lease, completed_request_ids):
        del lease, completed_request_ids
        return None if not self.pending else self.pending.pop(0)


def test_lane_worker_completes_and_pulls_without_waiting_for_pulse() -> None:
    clock = [0.0]
    hooks = _FakeLaneHooks(
        clock,
        remaining={"first": 2, "next": 3},
        pending=[LaneTask("next", current_step=0, predicted_step_ms=10.0)],
    )
    lease = LaneLease(
        epoch=3,
        ranks=(0,),
        request_id="first",
        min_steps=2,
        deadline_ms=100.0,
        predicted_step_ms=10.0,
    )

    report = LaneWorker(clock_ms=lambda: clock[0]).run_lease(
        lease,
        LaneTask("first", current_step=0, predicted_step_ms=10.0),
        hooks,
    )

    assert report.completed_request_ids == ("first", "next")
    assert report.request_id is None
    assert hooks.completed == ["first", "next"]
    assert clock[0] == 50.0


def test_lane_worker_runs_extra_steps_but_respects_lease_deadline() -> None:
    clock = [0.0]
    hooks = _FakeLaneHooks(clock, remaining={"running": 100})
    lease = LaneLease(
        epoch=4,
        ranks=(2, 3),
        request_id="running",
        min_steps=2,
        deadline_ms=35.0,
        predicted_step_ms=10.0,
    )

    report = LaneWorker(clock_ms=lambda: clock[0]).run_lease(
        lease,
        LaneTask("running", current_step=0, predicted_step_ms=10.0),
        hooks,
    )

    assert report.request_id == "running"
    assert report.current_step == 3
    assert report.state_owner_rank == 2
    assert clock[0] == 30.0
