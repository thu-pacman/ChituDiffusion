import argparse
import json

from chitu_diffusers.benchmarks.benchmark_client import scale_trace_arrivals
from chitu_diffusers.benchmarks.flux1_3arm_summary import build_comparison
from chitu_diffusers.benchmarks.flux1_adaptive_trace import build_adaptive_trace
from chitu_diffusers.benchmarks.flux1_embedded_benchmark import _request_specs


def test_scale_trace_arrivals_preserves_order_and_scales_rate() -> None:
    trace = {
        "meta": {"rate_req_per_s": 3.0},
        "requests": [
            {"request_id": "later", "arrival_ms": 300.0},
            {"request_id": "first", "arrival_ms": 0.0},
        ],
    }

    requests, source_rate = scale_trace_arrivals(trace, 0.3)

    assert source_rate == 3.0
    assert [request["request_id"] for request in requests] == ["first", "later"]
    assert requests[1]["arrival_ms"] == 3000.0
    assert trace["requests"][0]["arrival_ms"] == 300.0


def test_flux1_benchmark_replays_trace_arrivals_and_deadlines(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "request_id": "later",
                        "arrival_ms": 120.0,
                        "prompt": "later prompt",
                        "width": 1024,
                        "height": 768,
                        "seed": 8,
                        "deadline_ms": 5000.0,
                    },
                    {
                        "request_id": "first",
                        "arrival_ms": 0.0,
                        "prompt": "first prompt",
                        "width": 512,
                        "height": 512,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(trace=str(trace_path))

    specs = _request_specs(args)

    assert [spec["request_id"] for spec in specs] == ["first", "later"]
    assert specs[0]["seed"] == 0
    assert specs[1]["arrival_ms"] == 120.0
    assert specs[1]["width"] == 1024
    assert specs[1]["height"] == 768
    assert specs[1]["deadline_ms"] == 5000.0


def test_flux1_adaptive_trace_uses_warmup_capacity_and_longest_cp1_slo() -> None:
    warmup = {
        "rows": [
            {
                "resolution": [512, 512],
                "width": 1,
                "latency_ms": 100.0,
                "terminal_ms": 20.0,
            },
            {
                "resolution": [512, 512],
                "width": 4,
                "latency_ms": 40.0,
                "terminal_ms": 10.0,
            },
            {
                "resolution": [1024, 1024],
                "width": 1,
                "latency_ms": 300.0,
                "terminal_ms": 60.0,
            },
            {
                "resolution": [1024, 1024],
                "width": 4,
                "latency_ms": 100.0,
                "terminal_ms": 30.0,
            },
        ]
    }

    trace = build_adaptive_trace(
        warmup,
        steps=10,
        requests_per_phase=4,
        world_size=4,
        slo_factor=1.2,
        seed=3,
    )

    meta = trace["meta"]
    # Longest cp1 request is 10 * 300 + 60 = 3060 ms.
    assert meta["slo"]["deadline_ms"] == 3672.0
    assert {request["deadline_ms"] for request in trace["requests"]} == {3672.0}
    assert len(trace["requests"]) == 12
    for phase_name in ("sparse", "crossover", "overload"):
        phase_resolutions = {
            (request["height"], request["width"])
            for request in trace["requests"]
            if request["phase"] == phase_name
        }
        assert phase_resolutions == {(512, 512), (1024, 1024)}
    assert [phase["name"] for phase in meta["phases"]] == [
        "sparse",
        "crossover",
        "overload",
    ]
    capacity = meta["capacity_estimate"]
    assert meta["phases"][0]["rate_req_per_s"] == 0.5 * min(
        capacity["static_dp_req_per_s"], capacity["static_cp_req_per_s"]
    )
    assert meta["phases"][2]["rate_req_per_s"] == 1.2 * max(
        capacity["static_dp_req_per_s"], capacity["static_cp_req_per_s"]
    )


def test_flux1_three_arm_summary_reports_phase_slo_and_cp_improvement() -> None:
    trace = {
        "meta": {
            "phases": [
                {"name": "sparse", "rate_req_per_s": 1.0},
                {"name": "crossover", "rate_req_per_s": 2.0},
                {"name": "overload", "rate_req_per_s": 3.0},
            ]
        },
        "requests": [
            {"request_id": name, "phase": name}
            for name in ("sparse", "crossover", "overload")
        ],
    }

    def metrics(throughput: float, scale: float) -> dict:
        return {
            "summary": {
                "throughput_req_per_s": throughput,
                "makespan_ms": 3000.0 / throughput,
                "mean_latency_ms": 1000.0 * scale,
                "p95_latency_ms": 1200.0 * scale,
                "mean_queue_delay_ms": 100.0 * scale,
                "p95_queue_delay_ms": 200.0 * scale,
                "slo_attainment_rate": 2 / 3,
            },
            "per_request": {
                name: {
                    "latency_ms": 1000.0 * scale,
                    "queue_delay_ms": 100.0 * scale,
                    "deadline_met": name != "overload",
                }
                for name in ("sparse", "crossover", "overload")
            },
        }

    comparison = build_comparison(
        trace,
        {
            "static_dp": metrics(2.0, 1.0),
            "static_cp": metrics(1.0, 2.0),
            "elastic": metrics(1.5, 1.2),
        },
    )

    assert comparison["elastic_improvement"]["vs_static_cp"]["throughput_pct"] == 50.0
    assert (
        comparison["results"]["elastic"]["phases"]["overload"][
            "slo_attainment_rate"
        ]
        == 0.0
    )
