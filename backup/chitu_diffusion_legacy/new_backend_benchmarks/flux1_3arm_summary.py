from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


STRATEGIES = ("static_dp", "static_cp", "elastic")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _phase_summary(metrics: dict[str, Any], request_ids: list[str]) -> dict[str, Any]:
    rows = [metrics["per_request"][request_id] for request_id in request_ids]
    latencies = [float(row["latency_ms"]) for row in rows]
    queue_delays = [float(row["queue_delay_ms"]) for row in rows]
    slo_values = [bool(row["deadline_met"]) for row in rows]
    return {
        "num_requests": len(rows),
        "mean_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mean_queue_delay_ms": statistics.fmean(queue_delays),
        "slo_attainment_rate": sum(slo_values) / len(slo_values),
    }


def _improvement(new: float, baseline: float) -> float:
    return (new / baseline - 1.0) * 100.0


def build_comparison(
    trace: dict[str, Any], metrics_by_strategy: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    missing = set(STRATEGIES) - metrics_by_strategy.keys()
    if missing:
        raise ValueError(f"missing strategies: {sorted(missing)}")
    requests = trace["requests"]
    request_ids = {str(request["request_id"]) for request in requests}
    phases = {
        str(phase["name"]): [
            str(request["request_id"])
            for request in requests
            if request.get("phase") == phase["name"]
        ]
        for phase in trace["meta"]["phases"]
    }
    results = {}
    for strategy in STRATEGIES:
        metrics = metrics_by_strategy[strategy]
        actual_ids = set(metrics["per_request"])
        if actual_ids != request_ids:
            raise ValueError(
                f"{strategy} request IDs differ from trace: "
                f"missing={sorted(request_ids - actual_ids)}, "
                f"extra={sorted(actual_ids - request_ids)}"
            )
        results[strategy] = {
            "overall": metrics["summary"],
            "phases": {
                name: _phase_summary(metrics, phase_request_ids)
                for name, phase_request_ids in phases.items()
            },
        }

    elastic = results["elastic"]["overall"]
    static_cp = results["static_cp"]["overall"]
    static_dp = results["static_dp"]["overall"]
    return {
        "trace_meta": trace["meta"],
        "results": results,
        "elastic_improvement": {
            "vs_static_cp": {
                "throughput_pct": _improvement(
                    float(elastic["throughput_req_per_s"]),
                    float(static_cp["throughput_req_per_s"]),
                ),
                "makespan_reduction_pct": -_improvement(
                    float(elastic["makespan_ms"]), float(static_cp["makespan_ms"])
                ),
                "mean_latency_reduction_pct": -_improvement(
                    float(elastic["mean_latency_ms"]),
                    float(static_cp["mean_latency_ms"]),
                ),
                "p95_latency_reduction_pct": -_improvement(
                    float(elastic["p95_latency_ms"]),
                    float(static_cp["p95_latency_ms"]),
                ),
                "mean_queue_reduction_pct": -_improvement(
                    float(elastic["mean_queue_delay_ms"]),
                    float(static_cp["mean_queue_delay_ms"]),
                ),
                "p95_queue_reduction_pct": -_improvement(
                    float(elastic["p95_queue_delay_ms"]),
                    float(static_cp["p95_queue_delay_ms"]),
                ),
            },
            "vs_static_dp": {
                "throughput_pct": _improvement(
                    float(elastic["throughput_req_per_s"]),
                    float(static_dp["throughput_req_per_s"]),
                ),
                "makespan_reduction_pct": -_improvement(
                    float(elastic["makespan_ms"]), float(static_dp["makespan_ms"])
                ),
                "mean_latency_reduction_pct": -_improvement(
                    float(elastic["mean_latency_ms"]),
                    float(static_dp["mean_latency_ms"]),
                ),
            },
        },
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    meta = comparison["trace_meta"]
    capacity = meta["capacity_estimate"]
    lines = [
        "# Flux.1 Adaptive Three-Arm Benchmark",
        "",
        f"SLO: `{meta['slo']['deadline_ms'] / 1000.0:.3f}s` "
        f"(`{meta['slo']['factor']:.2f}x` longest measured cp1 latency).",
        "",
        f"Warmup capacity estimate: static-DP `{capacity['static_dp_req_per_s']:.5f}` "
        f"req/s, static-CP `{capacity['static_cp_req_per_s']:.5f}` req/s.",
        "",
        "| Strategy | Throughput (req/s) | Makespan (s) | Mean latency (s) | "
        "P95 latency (s) | SLO attainment |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in STRATEGIES:
        row = comparison["results"][strategy]["overall"]
        lines.append(
            f"| {strategy} | {row['throughput_req_per_s']:.5f} | "
            f"{row['makespan_ms'] / 1000.0:.3f} | "
            f"{row['mean_latency_ms'] / 1000.0:.3f} | "
            f"{row['p95_latency_ms'] / 1000.0:.3f} | "
            f"{row['slo_attainment_rate']:.1%} |"
        )
    improvement = comparison["elastic_improvement"]["vs_static_cp"]
    lines.extend(
        [
            "",
            "Elastic versus serial static-CP: "
            f"throughput `{improvement['throughput_pct']:+.2f}%`, "
            f"makespan reduction `{improvement['makespan_reduction_pct']:+.2f}%`, "
            f"mean-latency reduction `{improvement['mean_latency_reduction_pct']:+.2f}%`, "
            f"P95-latency reduction `{improvement['p95_latency_reduction_pct']:+.2f}%`, "
            f"mean-queue reduction `{improvement['mean_queue_reduction_pct']:+.2f}%`.",
            "",
            "| Phase | Offered rate (req/s) | Strategy | Mean latency (s) | "
            "P95 latency (s) | Mean queue (s) | SLO attainment |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase in meta["phases"]:
        name = str(phase["name"])
        for strategy in STRATEGIES:
            row = comparison["results"][strategy]["phases"][name]
            lines.append(
                f"| {name} | {phase['rate_req_per_s']:.5f} | {strategy} | "
                f"{row['mean_latency_ms'] / 1000.0:.3f} | "
                f"{row['p95_latency_ms'] / 1000.0:.3f} | "
                f"{row['mean_queue_delay_ms'] / 1000.0:.3f} | "
                f"{row['slo_attainment_rate']:.1%} |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Flux.1 three-arm run")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    metrics = {
        strategy: json.loads(
            (args.run_root / strategy / "metrics.json").read_text(encoding="utf-8")
        )
        for strategy in STRATEGIES
    }
    comparison = build_comparison(trace, metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(render_markdown(comparison), encoding="utf-8")
    print(args.output)
    print(markdown_path)


if __name__ == "__main__":
    main()
