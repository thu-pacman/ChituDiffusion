#!/usr/bin/env python3
"""Run static DP/SP serving baselines on existing CP/DP hot-switch traces.

This is a deliberately simple fixed-deployment simulator:

* ``dp`` uses one GPU per request, with ``total_gpus`` independent FIFO servers.
* ``sp2`` uses two GPUs per request, with ``total_gpus//2`` FIFO servers.
* ``sp4`` uses four GPUs per request, with ``total_gpus//4`` FIFO servers.

Each request occupies one server until its denoise phase completes. This is meant
as a baseline for serving metrics and SLO attainment, not as a hot-switch policy
simulator.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for path in (str(_HERE), str(_REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from chitu_diffusion.runtime.cost_model import CostModel, image_token_count  # noqa: E402
from run_experiment import load_profiles, load_scenario  # noqa: E402


@dataclass(frozen=True)
class BaselineRequest:
    request_id: str
    arrival_ms: float
    service_ms: float
    shape: str
    num_steps: int
    n_sample: int = 1


@dataclass(frozen=True)
class BaselineResult:
    trace: str
    policy: str
    total_gpus: int
    gpus_per_request: int
    metrics: dict[str, float]
    slo_attainment: dict[str, float]
    schedule: list[dict[str, Any]]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _fmt_s(ms: float) -> str:
    return f"{ms / 1000.0:.1f}"


def _slo_attainment(latencies_ms: list[float], slo_seconds: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {f"<= {s:g}s": 0.0 for s in slo_seconds}
    return {
        f"<= {s:g}s": sum(1 for value in latencies_ms if value <= s * 1000.0) / len(latencies_ms)
        for s in slo_seconds
    }


def simulate_fixed_servers(
    requests: list[BaselineRequest],
    *,
    trace_name: str,
    policy: str,
    total_gpus: int,
    gpus_per_request: int,
    slo_seconds: list[float],
) -> BaselineResult | None:
    if total_gpus < gpus_per_request:
        return None
    server_count = total_gpus // gpus_per_request
    if server_count <= 0:
        return None

    available_at = [0.0 for _ in range(server_count)]
    rows: list[dict[str, float | str]] = []
    busy_gpu_ms = 0.0

    for req in sorted(requests, key=lambda item: (item.arrival_ms, item.request_id)):
        server = min(range(server_count), key=lambda idx: (available_at[idx], idx))
        start_ms = max(req.arrival_ms, available_at[server])
        finish_ms = start_ms + req.service_ms
        available_at[server] = finish_ms
        busy_gpu_ms += req.service_ms * gpus_per_request
        gpu_start = server * gpus_per_request
        rows.append(
            {
                "request_id": req.request_id,
                "arrival_ms": req.arrival_ms,
                "start_ms": start_ms,
                "finish_ms": finish_ms,
                "latency_ms": finish_ms - req.arrival_ms,
                "queue_ms": start_ms - req.arrival_ms,
                "service_ms": req.service_ms,
                "shape": req.shape,
                "server": server,
                "gpu_start": gpu_start,
                "gpu_count": gpus_per_request,
                "gpu_end": gpu_start + gpus_per_request - 1,
            }
        )

    if not rows:
        makespan_ms = 0.0
        latencies: list[float] = []
        queues: list[float] = []
    else:
        first_arrival = min(float(row["arrival_ms"]) for row in rows)
        last_finish = max(float(row["finish_ms"]) for row in rows)
        makespan_ms = last_finish - first_arrival
        latencies = [float(row["latency_ms"]) for row in rows]
        queues = [float(row["queue_ms"]) for row in rows]

    metrics = {
        "num_requests": float(len(rows)),
        "server_count": float(server_count),
        "makespan_ms": makespan_ms,
        "throughput_req_per_s": len(rows) * 1000.0 / makespan_ms if makespan_ms > 0 else 0.0,
        "mean_latency_ms": mean(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "mean_queue_delay_ms": mean(queues) if queues else 0.0,
        "p95_queue_delay_ms": _percentile(queues, 0.95),
        "gpu_busy_ratio": busy_gpu_ms / (makespan_ms * total_gpus) if makespan_ms > 0 else 0.0,
    }
    return BaselineResult(
        trace=trace_name,
        policy=policy,
        total_gpus=total_gpus,
        gpus_per_request=gpus_per_request,
        metrics=metrics,
        slo_attainment=_slo_attainment(latencies, slo_seconds),
        schedule=rows,
    )


def abstract_requests(trace_path: Path, mode: str, profiles_path: Path) -> tuple[list[BaselineRequest], int]:
    profiles = load_profiles(profiles_path)
    requests, _switch, config = load_scenario(trace_path)
    reqs: list[BaselineRequest] = []
    for req in requests:
        profile = profiles[req.profile]
        service_ms = sum(profile.step_ms(mode, step) for step in range(req.num_steps))
        reqs.append(
            BaselineRequest(
                request_id=req.request_id,
                arrival_ms=req.arrival_ms,
                service_ms=service_ms,
                shape=profile.name,
                num_steps=req.num_steps,
            )
        )
    return reqs, config.total_gpus


def serve_requests(trace_path: Path, mode: str, cost_model: CostModel) -> list[BaselineRequest]:
    data = json.loads(trace_path.read_text())
    reqs: list[BaselineRequest] = []
    sp_degree = {"dp1": 1, "cp2": 2, "cp4": 4}[mode]
    for item in data.get("requests", []):
        width = int(item["width"])
        height = int(item["height"])
        steps = int(item["num_steps"])
        n_sample = int(item.get("n_sample", 1) or 1)
        img_tokens = image_token_count(width, height)
        service_ms = cost_model.predict_request_ms(
            img_tokens=img_tokens,
            num_steps=steps,
            batch=n_sample,
            sp_degree=sp_degree,
        )
        reqs.append(
            BaselineRequest(
                request_id=str(item["request_id"]),
                arrival_ms=float(item["arrival_ms"]),
                service_ms=service_ms,
                shape=f"{width}x{height}",
                num_steps=steps,
                n_sample=n_sample,
            )
        )
    return reqs


def run_baselines(
    *,
    abstract_trace_paths: list[Path],
    serve_trace_paths: list[Path],
    profiles_path: Path,
    cost_model_path: Path,
    serve_gpus: int,
    slo_seconds: list[float],
) -> list[BaselineResult]:
    results: list[BaselineResult] = []
    policies = [("dp", "dp1", 1), ("sp2", "cp2", 2), ("sp4", "cp4", 4)]

    for trace_path in abstract_trace_paths:
        for policy, mode, gpus_per_request in policies:
            reqs, total_gpus = abstract_requests(trace_path, mode, profiles_path)
            result = simulate_fixed_servers(
                reqs,
                trace_name=trace_path.stem,
                policy=policy,
                total_gpus=total_gpus,
                gpus_per_request=gpus_per_request,
                slo_seconds=slo_seconds,
            )
            if result is not None:
                results.append(result)

    cost_model = CostModel.load(cost_model_path)
    for trace_path in serve_trace_paths:
        for policy, mode, gpus_per_request in policies:
            reqs = serve_requests(trace_path, mode, cost_model)
            result = simulate_fixed_servers(
                reqs,
                trace_name=f"serve/{trace_path.stem}",
                policy=policy,
                total_gpus=serve_gpus,
                gpus_per_request=gpus_per_request,
                slo_seconds=slo_seconds,
            )
            if result is not None:
                results.append(result)
    return results


def render_markdown(results: list[BaselineResult], slo_seconds: list[float]) -> str:
    lines: list[str] = [
        "# Static DP/SP Baseline",
        "",
        "No trace currently carries explicit request deadlines, so SLO attainment is reported against fixed latency thresholds.",
        "Latencies model denoise time only; text encode, VAE decode, and image save are not included.",
        "",
        "| trace | policy | GPUs | groups | thrpt req/s | mean s | p95 s | p99 s | queue s | busy | "
        + " | ".join(f"SLO<={s:g}s" for s in slo_seconds)
        + " |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        + " | ".join("---:" for _ in slo_seconds)
        + " |",
    ]
    for result in results:
        m = result.metrics
        row = [
            result.trace,
            result.policy,
            str(result.total_gpus),
            str(int(m["server_count"])),
            f"{m['throughput_req_per_s']:.3f}",
            _fmt_s(m["mean_latency_ms"]),
            _fmt_s(m["p95_latency_ms"]),
            _fmt_s(m["p99_latency_ms"]),
            _fmt_s(m["mean_queue_delay_ms"]),
            f"{m['gpu_busy_ratio']:.3f}",
        ]
        row.extend(f"{result.slo_attainment[f'<= {s:g}s'] * 100:.0f}%" for s in slo_seconds)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _safe_name(text: str) -> str:
    text = text.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _nice_tick_seconds(max_seconds: float) -> float:
    if max_seconds <= 0:
        return 1.0
    candidates = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
    target = max_seconds / 8.0
    for value in candidates:
        if value >= target:
            return float(value)
    return float(candidates[-1])


def _shape_color(shape: str) -> str:
    palette = [
        "#4f7cac",
        "#e07a5f",
        "#59a14f",
        "#f2c14e",
        "#af7aa1",
        "#76b7b2",
        "#d37295",
        "#8cd17d",
    ]
    return palette[sum(ord(ch) for ch in shape) % len(palette)]


def render_timeline_svg(result: BaselineResult) -> str:
    m = result.metrics
    rows = result.schedule
    first_arrival = min((float(row["arrival_ms"]) for row in rows), default=0.0)
    makespan_ms = max(float(m["makespan_ms"]), 1.0)

    width = 1180
    left = 92
    right = 28
    top = 66
    bottom = 58
    lane_h = 28
    gap = 7
    request_gap = 38
    req_h = 13
    req_gap = 8
    req_top = top + result.total_gpus * lane_h + max(0, result.total_gpus - 1) * gap + request_gap
    plot_w = width - left - right
    request_area_h = len(rows) * (req_h + req_gap)
    height = req_top + request_area_h + bottom

    def x(ms: float) -> float:
        return left + ((ms - first_arrival) / makespan_ms) * plot_w

    def lane_y(gpu: int) -> float:
        return top + gpu * (lane_h + gap)

    max_s = makespan_ms / 1000.0
    tick_s = _nice_tick_seconds(max_s)
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#172033} .muted{fill:#667085} .axis{stroke:#98a2b3;stroke-width:1}",
        ".grid{stroke:#e4e7ec;stroke-width:1} .lane{fill:#f8fafc;stroke:#d0d5dd;stroke-width:1} .bar{stroke:#101828;stroke-width:.7}",
        ".queue{stroke:#98a2b3;stroke-width:3;stroke-linecap:round}.arrival{stroke:#344054;stroke-width:1.5}.finish{fill:#101828}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-size="18" font-weight="700">{html.escape(result.trace)} / {html.escape(result.policy)}</text>',
        (
            f'<text x="{left}" y="46" font-size="12" class="muted">'
            f'groups={int(m["server_count"])}  gpus/request={result.gpus_per_request}  '
            f'thrpt={m["throughput_req_per_s"]:.3f} req/s  mean={m["mean_latency_ms"]/1000:.1f}s  '
            f'p95={m["p95_latency_ms"]/1000:.1f}s  busy={m["gpu_busy_ratio"]:.3f}'
            "</text>"
        ),
    ]

    tick = 0.0
    while tick <= max_s + 1e-9:
        px = left + (tick / max_s) * plot_w if max_s > 0 else left
        parts.append(f'<line x1="{px:.1f}" y1="{top - 10}" x2="{px:.1f}" y2="{height - bottom + 8}" class="grid"/>')
        parts.append(f'<text x="{px:.1f}" y="{height - 20}" font-size="11" text-anchor="middle" class="muted">{tick:g}s</text>')
        tick += tick_s

    for gpu in range(result.total_gpus):
        y = lane_y(gpu)
        parts.append(f'<rect x="{left}" y="{y:.1f}" width="{plot_w}" height="{lane_h}" rx="3" class="lane"/>')
        parts.append(f'<text x="{left - 12}" y="{y + lane_h / 2 + 4:.1f}" font-size="12" text-anchor="end" class="muted">GPU{gpu}</text>')

    for row in rows:
        start = float(row["start_ms"])
        finish = float(row["finish_ms"])
        px = x(start)
        w = max(1.0, x(finish) - px)
        gpu_start = int(row["gpu_start"])
        gpu_count = int(row["gpu_count"])
        y = lane_y(gpu_start)
        h = gpu_count * lane_h + max(0, gpu_count - 1) * gap
        color = _shape_color(str(row["shape"]))
        title = (
            f'{row["request_id"]} shape={row["shape"]} '
            f'start={(start-first_arrival)/1000:.2f}s finish={(finish-first_arrival)/1000:.2f}s '
            f'queue={float(row["queue_ms"])/1000:.2f}s'
        )
        parts.append(
            f'<rect x="{px:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" '
            f'fill="{color}" fill-opacity="0.82" class="bar"><title>{html.escape(title)}</title></rect>'
        )
        if w >= 46:
            label = html.escape(str(row["request_id"]).replace("req", "r"))
            parts.append(
                f'<text x="{px + 4:.1f}" y="{y + min(h, lane_h) / 2 + 4:.1f}" '
                f'font-size="10" fill="#ffffff">{label}</text>'
            )

    request_title_y = req_top - 16
    parts.append(f'<text x="{left}" y="{request_title_y:.1f}" font-size="13" font-weight="700">Requests: arrival -> queue -> compute -> finish</text>')
    for idx, row in enumerate(sorted(rows, key=lambda item: (float(item["arrival_ms"]), str(item["request_id"])))):
        y = req_top + idx * (req_h + req_gap)
        mid_y = y + req_h / 2
        arr_x = x(float(row["arrival_ms"]))
        start_x = x(float(row["start_ms"]))
        finish_x = x(float(row["finish_ms"]))
        color = _shape_color(str(row["shape"]))
        label = html.escape(str(row["request_id"]))
        parts.append(f'<text x="{left - 12}" y="{mid_y + 4:.1f}" font-size="10" text-anchor="end" class="muted">{label}</text>')
        parts.append(f'<line x1="{arr_x:.1f}" y1="{mid_y:.1f}" x2="{start_x:.1f}" y2="{mid_y:.1f}" class="queue"/>')
        parts.append(
            f'<rect x="{start_x:.1f}" y="{y:.1f}" width="{max(1.0, finish_x - start_x):.1f}" height="{req_h}" '
            f'rx="3" fill="{color}" fill-opacity="0.86"><title>{html.escape(str(row["request_id"]))}: compute '
            f'{float(row["service_ms"])/1000:.2f}s, queue {float(row["queue_ms"])/1000:.2f}s, '
            f'latency {float(row["latency_ms"])/1000:.2f}s</title></rect>'
        )
        parts.append(f'<line x1="{arr_x:.1f}" y1="{y - 2:.1f}" x2="{arr_x:.1f}" y2="{y + req_h + 2:.1f}" class="arrival"/>')
        parts.append(f'<circle cx="{arr_x:.1f}" cy="{mid_y:.1f}" r="3.2" fill="#ffffff" stroke="#344054" stroke-width="1.4"><title>arrival</title></circle>')
        parts.append(f'<circle cx="{finish_x:.1f}" cy="{mid_y:.1f}" r="3.5" class="finish"><title>finish</title></circle>')

    legend_y = height - 36
    parts.append(f'<line x1="{left}" y1="{height - bottom + 8}" x2="{width - right}" y2="{height - bottom + 8}" class="axis"/>')
    parts.append(f'<circle cx="{left:.1f}" cy="{legend_y:.1f}" r="3.2" fill="#ffffff" stroke="#344054" stroke-width="1.4"/>')
    parts.append(f'<text x="{left + 10}" y="{legend_y + 4:.1f}" font-size="11" class="muted">arrival</text>')
    parts.append(f'<line x1="{left + 72}" y1="{legend_y:.1f}" x2="{left + 112}" y2="{legend_y:.1f}" class="queue"/>')
    parts.append(f'<text x="{left + 120}" y="{legend_y + 4:.1f}" font-size="11" class="muted">queue</text>')
    parts.append(f'<rect x="{left + 170}" y="{legend_y - 6:.1f}" width="34" height="12" rx="3" fill="#4f7cac" fill-opacity="0.86"/>')
    parts.append(f'<text x="{left + 212}" y="{legend_y + 4:.1f}" font-size="11" class="muted">compute</text>')
    parts.append(f'<circle cx="{left + 284}" cy="{legend_y:.1f}" r="3.5" class="finish"/>')
    parts.append(f'<text x="{left + 294}" y="{legend_y + 4:.1f}" font-size="11" class="muted">finish</text>')
    parts.append(f'<text x="{width - right}" y="{height - 20}" font-size="11" text-anchor="end" class="muted">time since first arrival</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_timeline_outputs(results: list[BaselineResult], output_dir: Path) -> None:
    timeline_dir = output_dir / "static_baseline_timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    index_lines = [
        "# Static Baseline Timelines",
        "",
        "Each SVG is a fixed-deployment execution timeline. Filled blocks are denoise compute; blank space is idle capacity or queueing.",
        "",
    ]
    for result in results:
        filename = f"{_safe_name(result.trace)}__{_safe_name(result.policy)}.svg"
        path = timeline_dir / filename
        path.write_text(render_timeline_svg(result))
        index_lines.extend(
            [
                f"## {result.trace} / {result.policy}",
                "",
                f"![{result.trace} {result.policy}](static_baseline_timelines/{filename})",
                "",
            ]
        )
    (output_dir / "static_baseline_timelines.md").write_text("\n".join(index_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=_HERE / "traces" / "profiles.json")
    parser.add_argument("--cost-model", type=Path, default=_HERE / "cost_model.json")
    parser.add_argument("--serve-gpus", type=int, default=4)
    parser.add_argument("--slo-seconds", type=float, nargs="*", default=[10.0, 30.0, 60.0, 120.0, 180.0])
    parser.add_argument("--output-dir", type=Path, default=_HERE / "out")
    parser.add_argument("--abstract-traces", type=Path, nargs="*")
    parser.add_argument("--serve-traces", type=Path, nargs="*")
    args = parser.parse_args()

    abstract_traces = (
        sorted(p for p in (_HERE / "traces").glob("*.json") if p.name != "profiles.json")
        if args.abstract_traces is None
        else args.abstract_traces
    )
    serve_traces = (
        sorted((_HERE / "traces" / "serve").glob("*.json"))
        if args.serve_traces is None
        else args.serve_traces
    )
    results = run_baselines(
        abstract_trace_paths=abstract_traces,
        serve_trace_paths=serve_traces,
        profiles_path=args.profiles,
        cost_model_path=args.cost_model,
        serve_gpus=args.serve_gpus,
        slo_seconds=args.slo_seconds,
    )
    markdown = render_markdown(results, args.slo_seconds)
    print(markdown)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "static_baseline.md").write_text(markdown)
    (args.output_dir / "static_baseline.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False) + "\n"
    )
    write_timeline_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
