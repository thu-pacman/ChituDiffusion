from __future__ import annotations

import json

from chitu_diffusion.epac.cost import (
    MeasuredStepCostModel,
    MeasuredTransferCostModel,
)
from chitu_diffusion.epac.trace_simulator import (
    EpeTraceSimulator,
    TraceRequest,
    apply_isolated_p95_deadlines,
    compare_epe_to_static_cp,
    generate_profile_trace,
    load_trace,
    load_warmup_models,
    measure_isolated_baseline,
    normalize_trace_load,
    run_slo_baseline_sweep,
    save_trace,
)


def _models():
    costs = MeasuredStepCostModel()
    rows = []
    for sequence_length, scale in ((1024, 1.0), (4096, 2.0), (16384, 5.0)):
        for width, latency in ((1, 10.0), (2, 7.0), (4, 6.0)):
            rows.append(
                {
                    "sequence_length": sequence_length,
                    "width": width,
                    "latency_ms": latency * scale,
                    "terminal_gpu_ms": 3.0 * scale,
                    "terminal_d2h_ms": 1.0,
                    "terminal_cpu_ms": 2.0,
                    "completion_tail_ms": 6.0 * scale,
                }
            )
    costs.initialize(rows)
    transfers = MeasuredTransferCostModel()
    transfers.initialize(
        [
            {
                "bytes": 1_000_000,
                "source": source,
                "destination": destination,
                "latency_ms": 0.5,
            }
            for source in range(8)
            for destination in range(8)
            if source != destination
        ]
    )
    return costs, transfers, rows


def _mixed_trace() -> tuple[TraceRequest, ...]:
    profiles = [
        {
            "profile_key": profile,
            "sequence_length": length,
            "num_steps": steps,
            "state_bytes": state_bytes,
            "metadata": {"adapter_value": profile},
        }
        for profile, length, state_bytes in (
            ("compact", 1024, 2_000_000),
            ("medium", 4096, 5_000_000),
            ("large", 16384, 8_000_000),
        )
        for steps in (4, 6, 8)
    ]
    return generate_profile_trace(
        request_count=12,
        arrival_rate_rps=1000.0,
        profiles=profiles,
        deadline_ms=500.0,
        seed=7,
    )


def test_variable_trace_contains_multiple_profiles_and_lengths() -> None:
    trace = _mixed_trace()

    assert len({request.profile_key for request in trace}) >= 2
    assert len({request.sequence_length for request in trace}) >= 2
    assert [request.arrival_ms for request in trace] == sorted(
        request.arrival_ms for request in trace
    )


def test_trace_round_trip_preserves_shape_metadata(tmp_path) -> None:
    path = tmp_path / "trace.json"
    trace = _mixed_trace()
    save_trace(path, trace)

    loaded = load_trace(path)

    assert loaded == trace
    assert json.loads(path.read_text())["requests"][0]["profile_key"]


def test_elastic_replay_uses_valid_tp2_cp_lanes_and_beats_static_cp() -> None:
    costs, transfers, _ = _models()
    result = compare_epe_to_static_cp(
        _mixed_trace(),
        cost_model=costs,
        transfer_cost_model=transfers,
        tp_degree=2,
        cp_world_size=4,
        allowed_lane_widths=(1, 2, 4),
        pulse_steps=2,
    )

    assert result["comparison"]["baseline"] == "tp2cp4"
    assert result["comparison"]["throughput_speedup"] > 1.0
    assert result["elastic"]["metrics"]["request_count"] == 12
    assert len(result["elastic"]["metrics"]["latency_by_profile_ms"]) >= 2
    for phase in result["elastic"]["timeline"]:
        assert phase["cp_width"] in {1, 2, 4}
        assert len(phase["physical_ranks"]) == phase["cp_width"] * 2
        assert len(set(phase["physical_ranks"])) == len(phase["physical_ranks"])


def test_cpu_completion_tail_does_not_extend_lane_occupancy() -> None:
    costs, transfers, _ = _models()
    request = TraceRequest(
        request_id="one",
        arrival_ms=0.0,
        sequence_length=1024,
        num_steps=2,
        profile_key="compact",
    )
    result = EpeTraceSimulator(
        cost_model=costs,
        transfer_cost_model=transfers,
        strategy="static_cp",
    ).run((request,))

    completed = result.requests[0]
    assert completed.completion_ms > completed.lane_complete_ms
    assert result.timeline[0].end_ms == completed.lane_complete_ms


def test_loads_runtime_warmup_report_schema(tmp_path) -> None:
    _, _, rows = _models()
    path = tmp_path / "epe_warmup.json"
    path.write_text(json.dumps({"rows": rows, "transfer_rows": []}))

    costs, transfers = load_warmup_models(path)

    assert costs.ready
    assert not transfers.ready
    assert costs.predict_step_ms(sequence_length=4096, width=2) == 14.0


def test_isolated_profile_p95_deadlines_and_normalized_load() -> None:
    costs, transfers, _ = _models()
    trace = _mixed_trace()
    baseline = measure_isolated_baseline(
        trace,
        cost_model=costs,
        transfer_cost_model=transfers,
    )

    loaded = normalize_trace_load(
        trace,
        normalized_load=0.8,
        static_capacity_rps=baseline["static_capacity_rps"],
    )
    deadline_trace = apply_isolated_p95_deadlines(
        loaded,
        baseline["isolated_p95_ms"],
        alpha=2.0,
    )

    span_s = (loaded[-1].arrival_ms - loaded[0].arrival_ms) / 1000.0
    assert (len(loaded) - 1) / span_s == (
        0.8 * baseline["static_capacity_rps"]
    )
    assert all(
        request.deadline_ms
        == 2.0 * baseline["isolated_p95_ms"][request.profile_key]
        for request in deadline_trace
    )


def test_slo_sweep_reports_matched_ablation_goodput_and_overheads() -> None:
    costs, transfers, _ = _models()
    sweep = run_slo_baseline_sweep(
        _mixed_trace(),
        cost_model=costs,
        transfer_cost_model=transfers,
        alphas=(2.0,),
        normalized_loads=(0.8,),
        planner_overhead_ms=0.25,
    )

    scenario = sweep["scenarios"][0]
    assert scenario["normalized_load"] == 0.8
    assert set(scenario["results"]) == {
        "static-cp-max",
        "elastic-no-resize",
        "elastic-resize",
    }
    for mode, result in scenario["results"].items():
        metrics = result["metrics"]
        assert result["ablation_mode"] == mode
        assert metrics["slo_goodput_rps"] is not None
        assert metrics["deadline_met_count"] + metrics["deadline_miss_count"] == 12
        assert set(metrics["overhead_ms"]) == {
            "planner_wall_ms",
            "planner_simulated_ms",
            "dispatch_ms",
            "migration_ms",
            "terminal_lane_ms",
            "pulse_barrier_idle_cp_ms",
        }
    assert (
        scenario["results"]["elastic-no-resize"]["metrics"]["switch_count"] == 0
    )


def test_h3_cli_generates_mixed_trace_and_comparison(tmp_path, monkeypatch) -> None:
    from script.simulate_h3_epe_trace import main

    _, _, rows = _models()
    report_path = tmp_path / "epe_warmup.json"
    trace_path = tmp_path / "trace.json"
    result_path = tmp_path / "result.json"
    report_path.write_text(json.dumps({"rows": rows, "transfer_rows": []}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "simulate_h3_epe_trace.py",
            "--warmup-report",
            str(report_path),
            "--result-out",
            str(result_path),
            "--trace-out",
            str(trace_path),
            "--requests",
            "20",
            "--arrival-rate",
            "10",
            "--shapes",
            "512x768@2,768x768@5,768x1024@8",
            "--steps",
            "4,6",
            "--seed",
            "7",
            "--alphas",
            "2",
            "--normalized-loads",
            "0.8",
        ],
    )

    assert main() == 0
    trace = load_trace(trace_path)
    result = json.loads(result_path.read_text())
    assert len(
        {
            (
                request.metadata["height"],
                request.metadata["width"],
            )
            for request in trace
        }
    ) == 3
    assert len({request.sequence_length for request in trace}) == 3
    assert result["comparison"]["baseline"] == "tp2cp4"
    assert len(result["slo_baseline"]["scenarios"]) == 1


def test_integrated_experiment_collects_fresh_deployment_warmup(
    tmp_path, monkeypatch
) -> None:
    from script.simulate_h3_epe_trace import _record_startup_warmup

    _, _, rows = _models()
    output_root = tmp_path / "service-output"
    config_path = tmp_path / "stage.yaml"
    config_path.write_text(
        f"gpu: [2, 3]\noutput_root: {output_root}\n",
        encoding="utf-8",
    )
    captured = {}

    class FakeProcess:
        pid = 999_999

        def poll(self):
            return None

        def wait(self, timeout):
            captured["wait_timeout"] = timeout
            return -15

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        output_root.mkdir(parents=True)
        (output_root / "epe_warmup.json").write_text(
            json.dumps({"rows": rows, "transfer_rows": []}),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr(
        "script.simulate_h3_epe_trace.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "script.simulate_h3_epe_trace.os.killpg",
        lambda pid, sig: captured.setdefault("signal", (pid, sig)),
    )

    report = _record_startup_warmup(
        config_path,
        service_python=tmp_path / "python",
        timeout_s=5,
        log_path=tmp_path / "warmup.log",
    )

    assert report == output_root / "epe_warmup.json"
    assert captured["environment"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert "--nproc-per-node=2" in captured["command"]
    assert captured["signal"][0] == FakeProcess.pid
