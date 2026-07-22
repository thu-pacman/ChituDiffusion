from chitu_diffusers.benchmarks.benchmark_client import scale_trace_arrivals


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
