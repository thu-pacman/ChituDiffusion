from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import pytest

from chitu_diffusers import (
    AdapterRegistry,
    DiffusersEngine,
    DiffusersModelAdapter,
    DiffusionRequest,
    EngineConfig,
    EpeSchedulingPolicy,
    FullWorldLaneWorkerPool,
    HotSwitchPoolConfig,
    LaneConstraints,
    LaneLease,
    LaneReport,
    LaneTask,
    LaneWorker,
    MeasuredStepCostModel,
    MeasuredTransferCostModel,
    LaneWorkItem,
    LaneWorkResult,
    OptimizationChain,
    PipelineCapabilities,
    PulseCoordinator,
    PulseLaneBroker,
    RequestProfile,
    RequestStatus,
    SchedulableRequest,
    StepPlan,
    StepOutcome,
    SingletonLaneWorkerPool,
)


class FakePipeline:
    def __init__(self) -> None:
        self.events: list[str] = []


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


@dataclass
class FakeState:
    request_id: str
    total_steps: int
    step_index: int = 0
    value: int = 0


class FakeAdapter(DiffusersModelAdapter):
    name = "fake"
    capabilities = PipelineCapabilities(flexcache=True)

    @classmethod
    def supports(cls, pipeline: Any) -> bool:
        return isinstance(pipeline, FakePipeline)

    def prepare_request(
        self, pipeline: FakePipeline, request: DiffusionRequest
    ) -> FakeState:
        pipeline.events.append(f"prepare:{request.request_id}")
        if request.inputs.get("fail_prepare"):
            raise ValueError("prepare failed")
        return FakeState(request.request_id, request.num_inference_steps)

    def prepare_step(
        self, pipeline: FakePipeline, state: FakeState, plan: StepPlan
    ) -> int:
        del plan
        pipeline.events.append(f"step:{state.request_id}:{state.step_index}")
        return state.step_index

    def model_forward(
        self,
        pipeline: FakePipeline,
        state: FakeState,
        model_inputs: int,
        plan: StepPlan,
    ) -> int:
        del plan
        pipeline.events.append(f"model:{state.request_id}:{model_inputs}")
        if state.request_id == "fail-model":
            raise RuntimeError("model failed")
        return 1

    def process_model_output(
        self,
        pipeline: FakePipeline,
        state: FakeState,
        model_inputs: int,
        model_output: int,
        plan: StepPlan,
    ) -> int:
        del pipeline, state, model_inputs, plan
        return model_output

    def scheduler_step(
        self,
        pipeline: FakePipeline,
        state: FakeState,
        prediction: int,
        plan: StepPlan,
    ) -> None:
        del pipeline, plan
        state.value += prediction
        state.step_index += 1

    def is_complete(self, state: FakeState) -> bool:
        return state.step_index >= state.total_steps

    def profile(self, state: FakeState) -> RequestProfile:
        return RequestProfile(
            total_steps=state.total_steps,
            completed_steps=state.step_index,
            shape_key=(state.total_steps,),
        )

    def finalize_request(
        self,
        pipeline: FakePipeline,
        state: FakeState,
        request: DiffusionRequest,
    ) -> dict[str, int | str]:
        pipeline.events.append(f"finalize:{request.request_id}")
        return {"request_id": request.request_id, "value": state.value}

    def abort_request(self, pipeline: FakePipeline, state: FakeState) -> None:
        pipeline.events.append(f"abort:{state.request_id}")


class FiveStepPolicy:
    def plan(self, requests, *, pulse_steps):
        assert pulse_steps == 5
        return [StepPlan(request_id=requests[0].request_id, steps=5)]


class CapturingPolicy:
    def __init__(self) -> None:
        self.requests = ()

    def plan(self, requests, *, pulse_steps):
        self.requests = tuple(requests)
        return [StepPlan(request_id=requests[0].request_id)]


class AddTenOptimization:
    def __init__(self) -> None:
        self.calls = 0
        self.ended: list[str] = []

    def execute_model(self, context, call_next):
        self.calls += 1
        return call_next() + 10

    def process_prediction(self, context, prediction):
        return prediction

    def after_step(self, context, prediction) -> None:
        return None

    def on_request_end(self, request_id: str) -> None:
        self.ended.append(request_id)


def make_engine(**kwargs) -> tuple[FakePipeline, DiffusersEngine]:
    pipeline = FakePipeline()
    engine = DiffusersEngine.from_pipeline(
        pipeline,
        adapter=FakeAdapter(),
        config=kwargs.pop(
            "config",
            EngineConfig(max_pending_requests=8, max_inflight_requests=2),
        ),
        **kwargs,
    )
    return pipeline, engine


def test_requests_own_state_and_interleave_by_pulse() -> None:
    pipeline, engine = make_engine()
    engine.submit(DiffusionRequest({}, request_id="a", num_inference_steps=2))
    engine.submit(DiffusionRequest({}, request_id="b", num_inference_steps=3))

    assert engine.pulse() == 2
    assert engine.status("a").steps_executed == 1
    assert engine.status("b").steps_executed == 1
    assert engine.pulse() == 2
    assert engine.status("a").status is RequestStatus.COMPLETED
    assert engine.status("b").status is RequestStatus.RUNNING
    assert engine.pulse() == 1

    assert engine.result("a").output == {"request_id": "a", "value": 2}
    assert engine.result("b").output == {"request_id": "b", "value": 3}
    assert [event for event in pipeline.events if event.startswith("model:")] == [
        "model:a:0",
        "model:b:0",
        "model:a:1",
        "model:b:1",
        "model:b:2",
    ]


def test_policy_controls_consecutive_steps_but_stops_at_completion() -> None:
    _, engine = make_engine(policy=FiveStepPolicy())
    request = DiffusionRequest({}, request_id="short", num_inference_steps=2)

    engine.submit(request)
    assert engine.pulse() == 2
    assert engine.result("short").status is RequestStatus.COMPLETED


def test_engine_default_slo_is_applied_and_request_deadline_overrides_it() -> None:
    policy = CapturingPolicy()
    _, engine = make_engine(
        config=EngineConfig(
            max_pending_requests=4,
            max_inflight_requests=2,
            default_deadline_ms=2_500,
        ),
        policy=policy,
    )
    engine.submit(DiffusionRequest({}, request_id="default", num_inference_steps=2))
    engine.submit(
        DiffusionRequest(
            {},
            request_id="override",
            num_inference_steps=2,
            deadline_ms=1_000,
        )
    )

    engine.pulse()
    by_id = {request.request_id: request for request in policy.requests}

    assert by_id["default"].deadline_at - by_id[
        "default"
    ].submitted_at == pytest.approx(2.5)
    assert by_id["override"].deadline_at - by_id[
        "override"
    ].submitted_at == pytest.approx(1.0)


def test_optimization_wraps_only_model_forward_and_cleans_request() -> None:
    optimization = AddTenOptimization()
    _, engine = make_engine(
        optimizations=OptimizationChain([optimization]),
    )

    result = engine.generate(
        DiffusionRequest({}, request_id="cached", num_inference_steps=2)
    )

    assert result.output["value"] == 22
    assert optimization.calls == 2
    assert optimization.ended == ["cached"]


def test_pending_and_running_cancel_finish_at_safe_boundaries() -> None:
    pipeline, engine = make_engine(
        config=EngineConfig(max_pending_requests=4, max_inflight_requests=1)
    )
    engine.submit(DiffusionRequest({}, request_id="running", num_inference_steps=3))
    engine.submit(DiffusionRequest({}, request_id="pending", num_inference_steps=3))

    assert engine.pulse() == 1
    assert engine.cancel("pending")
    assert engine.status("pending").status is RequestStatus.CANCELLED
    assert engine.cancel("running")
    assert engine.status("running").status is RequestStatus.RUNNING
    assert engine.pulse() == 0
    assert engine.status("running").status is RequestStatus.CANCELLED
    assert "abort:running" in pipeline.events

    assert {result.request_id for result in engine.poll()} == {"pending", "running"}
    assert engine.poll() == []


def test_running_cancel_reuses_slot_on_the_next_pulse() -> None:
    _, engine = make_engine(
        config=EngineConfig(max_pending_requests=4, max_inflight_requests=1)
    )
    engine.submit(DiffusionRequest({}, request_id="first", num_inference_steps=3))
    engine.submit(DiffusionRequest({}, request_id="next", num_inference_steps=1))
    assert engine.pulse() == 1

    assert engine.cancel("first")
    assert engine.pulse() == 1

    assert engine.status("first").status is RequestStatus.CANCELLED
    assert engine.status("next").status is RequestStatus.COMPLETED


def test_request_failure_is_isolated_and_structured() -> None:
    _, engine = make_engine()
    engine.submit(DiffusionRequest({}, request_id="fail-model"))
    engine.submit(DiffusionRequest({}, request_id="healthy", num_inference_steps=1))

    assert engine.pulse() == 1
    failed = engine.result("fail-model")
    healthy = engine.result("healthy")
    assert failed.status is RequestStatus.FAILED
    assert failed.error.type == "RuntimeError"
    assert failed.error.message == "model failed"
    assert healthy.status is RequestStatus.COMPLETED


def test_registry_requires_exactly_one_matching_adapter() -> None:
    registry = AdapterRegistry([FakeAdapter])
    assert isinstance(registry.resolve(FakePipeline()), FakeAdapter)

    with pytest.raises(LookupError, match="no chitu_diffusers adapter"):
        registry.resolve(object())
    registry.register(FakeAdapter)
    assert isinstance(registry.resolve(FakePipeline()), FakeAdapter)


def test_duplicate_id_and_queue_limit_fail_at_admission() -> None:
    _, engine = make_engine(
        config=EngineConfig(max_pending_requests=1, max_inflight_requests=1)
    )
    request = DiffusionRequest({}, request_id="same")
    engine.submit(request)

    with pytest.raises(KeyError, match="duplicate request_id"):
        engine.submit(request)
    with pytest.raises(OverflowError, match="request queue is full"):
        engine.submit(DiffusionRequest({}, request_id="other"))


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
    admitted: bool = True,
) -> SchedulableRequest:
    return SchedulableRequest(
        request_id=request_id,
        submitted_at=0.0,
        deadline_at=deadline_at,
        priority=0,
        profile=RequestProfile(
            total_steps=total_steps,
            completed_steps=completed_steps,
            image_tokens=image_tokens,
        ),
        admitted=admitted,
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
        complete=complete,
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
            "error": None,
        },
    )

    assert completed == [first_request]
    assert response["action"] == "run"
    assert response["request_id"] == "req4"


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
