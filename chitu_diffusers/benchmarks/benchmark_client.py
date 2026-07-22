from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


def scale_trace_arrivals(
    trace: dict[str, Any], target_rate_req_per_s: float | None
) -> tuple[list[dict[str, Any]], float | None]:
    requests = sorted(trace["requests"], key=lambda item: float(item["arrival_ms"]))
    if target_rate_req_per_s is None:
        return requests, trace.get("meta", {}).get("rate_req_per_s")
    if target_rate_req_per_s <= 0:
        raise ValueError("arrival rate must be positive")
    source_rate = trace.get("meta", {}).get("rate_req_per_s")
    if source_rate is None or float(source_rate) <= 0:
        raise ValueError("trace meta.rate_req_per_s is required for rate scaling")
    scale = float(source_rate) / target_rate_req_per_s
    return (
        [
            {**request, "arrival_ms": float(request["arrival_ms"]) * scale}
            for request in requests
        ],
        float(source_rate),
    )


def _json_request(
    endpoint: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"content-type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail}") from exc


def _bytes_request(
    endpoint: str,
    path: str,
    *,
    timeout_s: float = 120.0,
) -> bytes:
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}{path}",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} GET {path}: {detail}") from exc


def _wait_ready(endpoint: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            health = _json_request(endpoint, "GET", "/health", timeout_s=2.0)
            if health.get("state") == "running":
                return
        except (OSError, RuntimeError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"service did not become ready within {timeout_s}s")


def _wait_requests(
    endpoint: str,
    request_ids: list[str],
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, dict[str, Any]]:
    remaining = set(request_ids)
    statuses: dict[str, dict[str, Any]] = {}
    deadline = time.monotonic() + timeout_s
    while remaining and time.monotonic() < deadline:
        for request_id in list(remaining):
            status = _json_request(
                endpoint,
                "GET",
                f"/v1/image-decode/{request_id}",
            )
            if status["status"] in {"completed", "failed", "cancelled"}:
                statuses[request_id] = status
                remaining.remove(request_id)
        if remaining:
            time.sleep(poll_interval_s)
    if remaining:
        raise TimeoutError(
            f"requests did not finish within {timeout_s}s: {sorted(remaining)}"
        )
    return statuses


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _request_payload(entry: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "request_id": str(entry["request_id"]),
        "prompt": str(entry["prompt"]),
        "negative_prompt": entry.get("negative_prompt"),
        "width": int(entry["width"]),
        "height": int(entry["height"]),
        "seed": int(entry.get("seed", 0)),
        "num_steps": int(entry["num_steps"]),
        "guidance_scale": float(entry.get("guidance_scale", 5.0)),
    }
    if entry.get("deadline_ms") is not None:
        payload["deadline_ms"] = float(entry["deadline_ms"])
    return payload


def _warmup(endpoint: str, first: dict[str, Any], steps: int) -> None:
    payload = _request_payload(first)
    payload.update(request_id=f"benchmark-warmup-{time.time_ns()}", num_steps=steps)
    _json_request(endpoint, "POST", "/v1/image-decode", payload)
    status = _wait_requests(
        endpoint,
        [payload["request_id"]],
        timeout_s=900.0,
        poll_interval_s=0.1,
    )[payload["request_id"]]
    if status["status"] != "completed":
        raise RuntimeError(f"warmup failed: {status}")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    trace_path = Path(args.trace).expanduser().resolve()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    requests, source_arrival_rate = scale_trace_arrivals(trace, args.arrival_rate)
    if not requests:
        raise ValueError("trace has no requests")

    _wait_ready(args.endpoint, args.ready_timeout_s)
    if args.warmup_steps > 0:
        _warmup(args.endpoint, requests[0], args.warmup_steps)

    benchmark_start = time.monotonic()
    admissions: dict[str, dict[str, float]] = {}
    for entry in requests:
        arrival_ms = float(entry["arrival_ms"])
        wait_s = benchmark_start + arrival_ms / 1000.0 - time.monotonic()
        if wait_s > 0:
            time.sleep(wait_s)
        request_id = str(entry["request_id"])
        before_ms = (time.monotonic() - benchmark_start) * 1000.0
        _json_request(
            args.endpoint, "POST", "/v1/image-decode", _request_payload(entry)
        )
        after_ms = (time.monotonic() - benchmark_start) * 1000.0
        admissions[request_id] = {
            "trace_arrival_ms": arrival_ms,
            "submit_start_ms": before_ms,
            "submit_finish_ms": after_ms,
        }

    request_ids = [str(entry["request_id"]) for entry in requests]
    statuses = _wait_requests(
        args.endpoint,
        request_ids,
        timeout_s=args.timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    failures = {
        request_id: status
        for request_id, status in statuses.items()
        if status["status"] != "completed"
    }
    if failures:
        raise RuntimeError(f"benchmark requests failed: {failures}")

    saved_images: dict[str, dict[str, Any]] = {}
    if args.save_images_dir:
        images_dir = Path(args.save_images_dir).expanduser().resolve()
        images_dir.mkdir(parents=True, exist_ok=True)
        requests_by_id = {str(entry["request_id"]): entry for entry in requests}
        for request_id in request_ids:
            data = _bytes_request(
                args.endpoint,
                f"/v1/image-decode/{request_id}/image",
            )
            image_path = images_dir / f"{request_id}.png"
            image_path.write_bytes(data)
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
                actual_size = image.size
            entry = requests_by_id[request_id]
            expected_size = (int(entry["width"]), int(entry["height"]))
            if actual_size != expected_size:
                raise RuntimeError(
                    f"{request_id} image size {actual_size} != {expected_size}"
                )
            saved_images[request_id] = {
                "image_path": str(image_path),
                "image_bytes": len(data),
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "image_size": list(actual_size),
            }

    first_server_submit = min(
        float(status["submitted_at"]) for status in statuses.values()
    )
    per_request: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    queue_delays: list[float] = []
    admission_delays: list[float] = []
    finishes: list[float] = []
    trace_by_id = {str(entry["request_id"]): entry for entry in requests}
    slo_results: list[tuple[bool, float]] = []
    for request_id in request_ids:
        status = statuses[request_id]
        admission = admissions[request_id]
        trace_entry = trace_by_id[request_id]
        finish_ms = (float(status["completed_at"]) - first_server_submit) * 1000.0
        latency_ms = finish_ms - admission["trace_arrival_ms"]
        admission_delay_ms = (
            admission["submit_finish_ms"] - admission["trace_arrival_ms"]
        )
        queue_delay_ms = float(status["queue_delay_ms"])
        deadline_ms = trace_entry.get("deadline_ms")
        deadline_metrics: dict[str, Any] = {}
        if deadline_ms is not None:
            deadline_ms = float(deadline_ms)
            tardiness_ms = max(0.0, latency_ms - deadline_ms)
            deadline_met = tardiness_ms == 0.0
            slo_results.append((deadline_met, tardiness_ms))
            deadline_metrics = {
                "deadline_ms": deadline_ms,
                "absolute_deadline_ms": (admission["trace_arrival_ms"] + deadline_ms),
                "deadline_met": deadline_met,
                "tardiness_ms": tardiness_ms,
            }
        per_request[request_id] = {
            **admission,
            "finish_ms": finish_ms,
            "latency_ms": latency_ms,
            "queue_delay_ms": queue_delay_ms,
            "admission_delay_ms": admission_delay_ms,
            "server_latency_ms": float(status["latency_ms"]),
            **deadline_metrics,
            **saved_images.get(request_id, {}),
        }
        finishes.append(finish_ms)
        latencies.append(latency_ms)
        queue_delays.append(queue_delay_ms)
        admission_delays.append(admission_delay_ms)

    makespan_ms = max(finishes) - min(float(item["arrival_ms"]) for item in requests)
    summary = {
        "num_requests": len(requests),
        "num_completed": len(statuses),
        "makespan_ms": makespan_ms,
        "throughput_req_per_s": len(requests) / (makespan_ms / 1000.0),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p99_latency_ms": _percentile(latencies, 99),
        "max_latency_ms": max(latencies),
        "mean_queue_delay_ms": sum(queue_delays) / len(queue_delays),
        "p95_queue_delay_ms": _percentile(queue_delays, 95),
        "mean_admission_delay_ms": sum(admission_delays) / len(admission_delays),
        "num_slo_requests": len(slo_results),
        "num_slo_met": sum(met for met, _ in slo_results),
        "slo_attainment_rate": (
            sum(met for met, _ in slo_results) / len(slo_results)
            if slo_results
            else None
        ),
        "total_tardiness_ms": sum(value for _, value in slo_results),
        "max_tardiness_ms": max(
            (value for _, value in slo_results),
            default=0.0,
        ),
    }
    result = {
        "endpoint": args.endpoint,
        "trace": str(trace_path),
        "workload": {
            "source_arrival_rate_req_per_s": source_arrival_rate,
            "arrival_rate_req_per_s": args.arrival_rate or source_arrival_rate,
            "duration_ms": max(float(item["arrival_ms"]) for item in requests),
        },
        "summary": summary,
        "per_request": per_request,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("BENCHMARK_RESULT", json.dumps(summary, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a Poisson trace against the EPAC Z-Image service"
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--poll-interval-s", type=float, default=0.1)
    parser.add_argument("--save-images-dir")
    parser.add_argument(
        "--arrival-rate",
        type=float,
        help="Scale trace arrivals to this offered rate in requests/s",
    )
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
