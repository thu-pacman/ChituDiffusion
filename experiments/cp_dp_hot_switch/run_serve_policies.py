#!/usr/bin/env python3
"""Run static + dynamic CP/DP policies on realistic serve traces and visualize GPU usage.

This ties the scaffolding together: the per-hardware cost model
(``cost_models/*.json``) drives per-shape step latencies, the realistic serve
traces (``traces/serve/*.json``) provide arrivals/shapes, and the N-GPU pool
engine (``simulate.simulate_pool``) runs every policy -- the static DP/SP
baselines and the dynamic ``elastic_hot_switch`` / ``oracle`` -- head to head. For
each (trace, policy) it emits a metrics comparison plus a GPU-usage timeline SVG
that shows, per GPU lane, the denoise-step blocks and any CP<->DP hot switches.

Usage::

    python3 run_serve_policies.py --cost-model rtx4090 --output-dir out/serve_policies_4090
    python3 run_serve_policies.py --cost-model h20     --output-dir out/serve_policies_h20
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for path in (str(_HERE), str(_REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from simulate import (  # noqa: E402
    PoolResult,
    RequestSpec,
    SimulationConfig,
    StepLatencyProfile,
    SwitchCostModel,
    load_cost_model,
    profile_from_cost_model,
    simulate_pool,
)


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
        "#4f7cac", "#e07a5f", "#59a14f", "#f2c14e",
        "#af7aa1", "#76b7b2", "#d37295", "#8cd17d",
    ]
    return palette[sum(ord(ch) for ch in shape) % len(palette)]

DEFAULT_POLICIES = [
    "static_dp",
    "static_sp2",
    "static_sp4",
    "shape_aware",
    "elastic_hot_switch",
    "oracle",
    "slo_elastic",
]

METRIC_COLUMNS = [
    ("throughput_req_per_s", "thrpt req/s", "hi"),
    ("mean_latency_ms", "mean FCT s", "lo"),
    ("p95_latency_ms", "p95 s", "lo"),
    ("slo_miss_count", "SLO miss", "lo"),
    ("max_tardiness_ms", "max tard s", "lo"),
    ("max_slowdown", "max slow", "lo"),
    ("mean_queue_delay_ms", "queue s", "lo"),
    ("useful_token_steps_per_gpu_s", "eff ktok-step/GPU-s", "hi"),
    ("gpu_busy_ratio", "busy diag", "hi"),
    ("switch_count", "switch", None),
    ("continuous_batch_used_count", "cbQ", None),
    ("flexcache_used_count", "flexQ", None),
]


def build_serve_inputs(
    trace_path: Path, cost_model, slo_factor: float = 4.0
) -> tuple[list[RequestSpec], dict[str, StepLatencyProfile]]:
    """Build RequestSpecs + per-shape cost-model profiles from one serve trace.

    Serve traces carry no deadlines, so we synthesize a shape-proportional SLO:
    ``deadline = arrival + slo_factor * fastest_solo_service_time`` where the fastest
    solo time is ``num_steps * min_k step_ms(k)`` (i.e. the best a request could do
    standalone at its best degree). This makes deadlines achievable and lets bigger
    shapes get proportionally more budget. Set ``slo_factor <= 0`` to disable deadlines.
    """
    data = json.loads(trace_path.read_text())
    profiles: dict[str, StepLatencyProfile] = {}
    requests: list[RequestSpec] = []
    for item in data.get("requests", []):
        width = int(item["width"])
        height = int(item["height"])
        steps = int(item["num_steps"])
        shape_key = f"{width}x{height}x{steps}"
        if shape_key not in profiles:
            profiles[shape_key] = profile_from_cost_model(
                cost_model, name=shape_key, width=width, height=height, num_steps=steps
            )
        arrival = float(item["arrival_ms"])
        deadline = None
        if slo_factor > 0:
            prof = profiles[shape_key]
            fastest_step = min(prof.step_ms(m, 0) for m in ("dp1", "cp2", "cp4"))
            deadline = arrival + slo_factor * steps * fastest_step
        requests.append(
            RequestSpec(
                request_id=str(item["request_id"]),
                arrival_ms=arrival,
                num_steps=steps,
                profile=shape_key,
                deadline_ms=deadline,
            )
        )
    return requests, profiles


def _slo_attainment(result: PoolResult, slo_seconds: list[float]) -> dict[str, float]:
    latencies = [float(r["latency_ms"]) for r in result.requests if r.get("latency_ms") is not None]
    if not latencies:
        return {f"<= {s:g}s": 0.0 for s in slo_seconds}
    return {
        f"<= {s:g}s": sum(1 for v in latencies if v <= s * 1000.0) / len(latencies)
        for s in slo_seconds
    }


def _fmt_metric(metric: str, value: float) -> str:
    if metric == "throughput_req_per_s":
        return f"{value:.3f}"
    if metric in ("useful_token_steps_per_gpu_s", "busy_token_steps_per_gpu_s"):
        return f"{value / 1000.0:.1f}"
    if metric == "gpu_busy_ratio":
        return f"{value:.3f}"
    if metric == "max_slowdown":
        return f"{value:.2f}"
    if metric in ("switch_count", "slo_miss_count", "flexcache_used_count", "continuous_batch_used_count"):
        return f"{int(value)}"
    return f"{value / 1000.0:.1f}"  # latency/queue/tardiness ms -> s


def render_comparison_markdown(
    trace_name: str,
    results: dict[str, PoolResult],
    slo: dict[str, dict[str, float]],
    slo_seconds: list[float],
) -> str:
    lines = [f"### {trace_name}", ""]
    header = "| policy | " + " | ".join(label for _, label, _ in METRIC_COLUMNS)
    header += " | " + " | ".join(f"SLO<={s:g}s" for s in slo_seconds) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in METRIC_COLUMNS)
    sep += " | " + " | ".join("---:" for _ in slo_seconds) + " |"
    lines.append(header)
    lines.append(sep)
    for policy, result in results.items():
        row = [policy]
        for metric, _label, _hib in METRIC_COLUMNS:
            row.append(_fmt_metric(metric, result.metrics[metric]))
        row.extend(f"{slo[policy][f'<= {s:g}s'] * 100:.0f}%" for s in slo_seconds)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    # Highlight: elastic_hot_switch vs best static, and slo_elastic vs M7 elastic.
    statics = ["static_dp", "static_sp2", "static_sp4"]
    if "elastic_hot_switch" in results and any(s in results for s in statics):
        elastic = results["elastic_hot_switch"].metrics
        best_static_p95 = min(results[s].metrics["p95_latency_ms"] for s in statics if s in results)
        best_static_thr = max(results[s].metrics["throughput_req_per_s"] for s in statics if s in results)
        p95_gain = (best_static_p95 / elastic["p95_latency_ms"]) if elastic["p95_latency_ms"] > 0 else 0.0
        thr_gain = (elastic["throughput_req_per_s"] / best_static_thr) if best_static_thr > 0 else 0.0
        lines.append(
            f"elastic_hot_switch vs best static: p95 x{p95_gain:.2f}, throughput x{thr_gain:.2f}, "
            f"switches={int(elastic['switch_count'])}."
        )
    if "slo_elastic" in results and "elastic_hot_switch" in results:
        slo = results["slo_elastic"].metrics
        elastic = results["elastic_hot_switch"].metrics
        best_static_miss = min((results[s].metrics["slo_miss_count"] for s in statics if s in results), default=0.0)
        lines.append(
            f"slo_elastic vs elastic_hot_switch: SLO miss {int(slo['slo_miss_count'])} vs "
            f"{int(elastic['slo_miss_count'])} (best static {int(best_static_miss)}), "
            f"max_slowdown {slo['max_slowdown']:.2f} vs {elastic['max_slowdown']:.2f}, "
            f"p95 {slo['p95_latency_ms'] / 1000:.1f}s vs {elastic['p95_latency_ms'] / 1000:.1f}s, "
            f"switches {int(slo['switch_count'])}, flexcache {int(slo['flexcache_used_count'])}."
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Pool-aware GPU-usage timeline (per-GPU-lane segments + switch markers)
# --------------------------------------------------------------------------- #
def render_pool_timeline_svg(trace_name: str, result: PoolResult) -> str:
    segments = [s for s in result.segments]
    total_gpus = result.total_gpus
    res_by_req = {r["request_id"]: r.get("resolution", "unknown") for r in result.requests}

    starts = [s.start_ms for s in segments] or [0.0]
    ends = [s.end_ms for s in segments] or [1.0]
    first_ms = min(starts)
    last_ms = max(ends)
    span_ms = max(last_ms - first_ms, 1.0)

    width = 1180
    left = 92
    right = 28
    top = 72
    lane_h = 30
    gap = 6
    req_gap = 40
    req_h = 12
    req_row_gap = 7
    lanes_bottom = top + total_gpus * lane_h + max(0, total_gpus - 1) * gap
    req_top = lanes_bottom + req_gap
    plot_w = width - left - right
    rows = sorted(result.requests, key=lambda r: (r["arrival_ms"], r["request_id"]))
    height = req_top + len(rows) * (req_h + req_row_gap) + 58

    def x(ms: float) -> float:
        return left + ((ms - first_ms) / span_ms) * plot_w

    def lane_y(g: int) -> float:
        return top + g * (lane_h + gap)

    m = result.metrics
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#172033} .muted{fill:#667085}",
        ".axis{stroke:#98a2b3;stroke-width:1} .grid{stroke:#e4e7ec;stroke-width:1}",
        ".lane{fill:#f8fafc;stroke:#d0d5dd;stroke-width:1} .bar{stroke:#101828;stroke-width:.6}",
        ".queue{stroke:#98a2b3;stroke-width:3;stroke-linecap:round} .arrival{stroke:#344054;stroke-width:1.4}",
        ".finish{fill:#101828} .switch{fill:#101828;fill-opacity:0.72}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-size="18" font-weight="700">{html.escape(trace_name)} / {html.escape(result.policy)}</text>',
        (
            f'<text x="{left}" y="48" font-size="12" class="muted">'
            f'gpus={total_gpus}  thrpt={m["throughput_req_per_s"]:.3f} req/s  '
            f'mean={m["mean_latency_ms"] / 1000:.1f}s  p95={m["p95_latency_ms"] / 1000:.1f}s  '
            f'queue={m["mean_queue_delay_ms"] / 1000:.1f}s  busy={m["gpu_busy_ratio"]:.3f}  '
            f'eff={m["useful_token_steps_per_gpu_s"] / 1000:.1f}k tok-step/GPU-s  '
            f'switches={int(m["switch_count"])}'
            "</text>"
        ),
    ]

    max_s = span_ms / 1000.0
    tick_s = _nice_tick_seconds(max_s)
    tick = 0.0
    while tick <= max_s + 1e-9:
        px = left + (tick / max_s) * plot_w if max_s > 0 else left
        parts.append(f'<line x1="{px:.1f}" y1="{top - 10}" x2="{px:.1f}" y2="{lanes_bottom + 6}" class="grid"/>')
        parts.append(f'<text x="{px:.1f}" y="{lanes_bottom + 22}" font-size="11" text-anchor="middle" class="muted">{tick:g}s</text>')
        tick += tick_s

    for g in range(total_gpus):
        y = lane_y(g)
        parts.append(f'<rect x="{left}" y="{y:.1f}" width="{plot_w}" height="{lane_h}" rx="3" class="lane"/>')
        parts.append(f'<text x="{left - 12}" y="{y + lane_h / 2 + 4:.1f}" font-size="12" text-anchor="end" class="muted">GPU{g}</text>')

    for seg in segments:
        px = x(seg.start_ms)
        w = max(1.0, x(seg.end_ms) - px)
        for g in seg.gpus:
            if g < 0 or g >= total_gpus:
                continue
            y = lane_y(g)
            if seg.event == "switch":
                parts.append(
                    f'<rect x="{px:.1f}" y="{y:.1f}" width="{w:.1f}" height="{lane_h}" class="switch">'
                    f'<title>{html.escape(seg.request_id)} switch -> width {seg.width}</title></rect>'
                )
                continue
            color = _shape_color(res_by_req.get(seg.request_id, "unknown"))
            batch_suffix = f' batch={seg.batch_size} ids={",".join(seg.request_ids)}' if getattr(seg, "batch_size", 1) > 1 else ""
            title = f'{seg.request_id} step {seg.step_index} width={seg.width}{batch_suffix} ({seg.reason})'
            parts.append(
                f'<rect x="{px:.1f}" y="{y:.1f}" width="{w:.1f}" height="{lane_h}" rx="2.5" '
                f'fill="{color}" fill-opacity="0.82" class="bar"><title>{html.escape(title)}</title></rect>'
            )
    # request-id labels on the first segment of each request (only on its lowest lane)
    seen: set[str] = set()
    for seg in sorted(segments, key=lambda s: s.start_ms):
        if seg.event not in ("step", "batch_step") or seg.request_id in seen or not seg.gpus:
            continue
        seen.add(seg.request_id)
        px = x(seg.start_ms)
        w = x(seg.end_ms) - px
        if w >= 34:
            g = min(seg.gpus)
            parts.append(
                f'<text x="{px + 4:.1f}" y="{lane_y(g) + lane_h / 2 + 4:.1f}" font-size="10" fill="#ffffff">'
                f'{html.escape(seg.request_id.replace("req", "r"))}</text>'
            )

    parts.append(
        f'<text x="{left}" y="{req_top - 14:.1f}" font-size="13" font-weight="700">Requests: arrival -&gt; queue -&gt; compute -&gt; finish</text>'
    )
    for idx, row in enumerate(rows):
        y = req_top + idx * (req_h + req_row_gap)
        mid_y = y + req_h / 2
        arr_x = x(float(row["arrival_ms"]))
        start_x = x(float(row["started_ms"])) if row.get("started_ms") is not None else arr_x
        finish_x = x(float(row["completed_ms"])) if row.get("completed_ms") is not None else start_x
        color = _shape_color(row.get("resolution", "unknown"))
        label = html.escape(str(row["request_id"]))
        sw = " *" if row.get("switched") else ""
        parts.append(f'<text x="{left - 12}" y="{mid_y + 4:.1f}" font-size="10" text-anchor="end" class="muted">{label}{sw}</text>')
        parts.append(f'<line x1="{arr_x:.1f}" y1="{mid_y:.1f}" x2="{start_x:.1f}" y2="{mid_y:.1f}" class="queue"/>')
        parts.append(
            f'<rect x="{start_x:.1f}" y="{y:.1f}" width="{max(1.0, finish_x - start_x):.1f}" height="{req_h}" rx="3" '
            f'fill="{color}" fill-opacity="0.86"><title>{label}: {row.get("resolution")}, '
            f'queue {float(row.get("queue_ms", 0.0)) / 1000:.2f}s, latency {float(row["latency_ms"]) / 1000:.2f}s</title></rect>'
        )
        parts.append(f'<circle cx="{arr_x:.1f}" cy="{mid_y:.1f}" r="3.2" fill="#ffffff" stroke="#344054" stroke-width="1.4"/>')
        parts.append(f'<circle cx="{finish_x:.1f}" cy="{mid_y:.1f}" r="3.4" class="finish"/>')

    legend_y = height - 26
    parts.append(f'<circle cx="{left:.1f}" cy="{legend_y:.1f}" r="3.2" fill="#ffffff" stroke="#344054" stroke-width="1.4"/>')
    parts.append(f'<text x="{left + 10}" y="{legend_y + 4:.1f}" font-size="11" class="muted">arrival</text>')
    parts.append(f'<line x1="{left + 66}" y1="{legend_y:.1f}" x2="{left + 104}" y2="{legend_y:.1f}" class="queue"/>')
    parts.append(f'<text x="{left + 112}" y="{legend_y + 4:.1f}" font-size="11" class="muted">queue</text>')
    parts.append(f'<rect x="{left + 162}" y="{legend_y - 6:.1f}" width="30" height="12" rx="3" fill="#4f7cac" fill-opacity="0.86"/>')
    parts.append(f'<text x="{left + 198}" y="{legend_y + 4:.1f}" font-size="11" class="muted">compute</text>')
    parts.append(f'<rect x="{left + 260}" y="{legend_y - 6:.1f}" width="30" height="12" rx="3" class="switch"/>')
    parts.append(f'<text x="{left + 296}" y="{legend_y + 4:.1f}" font-size="11" class="muted">switch</text>')
    parts.append(f'<text x="{width - right}" y="{legend_y + 4:.1f}" font-size="11" text-anchor="end" class="muted">time since first start</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_timelines(all_results: dict[str, dict[str, PoolResult]], output_dir: Path) -> None:
    timeline_dir = output_dir / "serve_policy_timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    index = [
        "# Serve Policy GPU-Usage Timelines",
        "",
        "Per GPU lane: filled blocks are denoise-step compute (colored by request shape), "
        "dark blocks are CP<->DP hot switches, blank is idle. The lower panel is each request's "
        "arrival -> queue -> compute -> finish lifecycle (`*` = the request was hot-switched).",
        "",
    ]
    for trace_name, results in all_results.items():
        index.append(f"## {trace_name}")
        index.append("")
        for policy, result in results.items():
            filename = f"{_safe_name(trace_name)}__{_safe_name(policy)}.svg"
            (timeline_dir / filename).write_text(render_pool_timeline_svg(trace_name, result))
            index.append(f"### {trace_name} / {policy}")
            index.append("")
            index.append(f"![{trace_name} {policy}](serve_policy_timelines/{filename})")
            index.append("")
    (output_dir / "serve_policy_timelines.md").write_text("\n".join(index) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cost-model",
        default=None,
        help="Hardware name (e.g. 'rtx4090', resolved under cost_models/) or explicit JSON path.",
    )
    parser.add_argument("--serve-gpus", type=int, default=4)
    parser.add_argument("--policies", nargs="*", default=DEFAULT_POLICIES)
    parser.add_argument("--serve-traces", type=Path, nargs="*")
    parser.add_argument("--switch-total-ms", type=float, default=0.0, help="Total CP<->DP switch cost per switch.")
    parser.add_argument("--switch-window", type=int, default=8, help="Latest denoise step at which a switch is allowed.")
    parser.add_argument("--slo-seconds", type=float, nargs="*", default=[5.0, 10.0, 30.0, 60.0, 120.0])
    parser.add_argument("--slo-factor", type=float, default=4.0,
                        help="Per-request deadline = arrival + slo_factor * fastest solo service time. <=0 disables deadlines.")
    parser.add_argument("--horizon-events", type=int, default=6, help="slo_elastic rolling-horizon depth for scoring.")
    parser.add_argument("--starvation-ms", type=float, default=30000.0, help="Queue wait above which a request counts as starved.")
    parser.add_argument("--fairness-beta", type=float, default=3.0)
    parser.add_argument("--no-continuous-batch", action="store_true", help="Disable same-shape DP continuous batching in slo_elastic.")
    parser.add_argument("--max-batch-items", type=int, default=8, help="Max requests in one continuous-batch DP lane.")
    parser.add_argument("--enable-flexcache", action="store_true", help="Allow the simulator-only emergency step-reduction action.")
    parser.add_argument("--flexcache-min-steps", type=int, default=4)
    parser.add_argument("--flexcache-max-reduce-steps", type=int, default=8)
    parser.add_argument("--flexcache-emergency-tardiness-ms", type=float, default=1.0)
    parser.add_argument("--flexcache-emergency-slack-ms", type=float, default=2000.0)
    parser.add_argument("--decision-log", action="store_true", help="Record slo_elastic per-decision explanations into the JSON dump.")
    parser.add_argument("--output-dir", type=Path, default=_HERE / "out" / "serve_policies")
    parser.add_argument("--no-timelines", action="store_true", help="Skip SVG timeline rendering.")
    args = parser.parse_args()

    cost_model = load_cost_model(args.cost_model)
    serve_traces = (
        sorted((_HERE / "traces" / "serve").glob("*.json"))
        if args.serve_traces is None
        else args.serve_traces
    )
    if not serve_traces:
        parser.error("no serve traces found")

    switch = SwitchCostModel(
        switch_cost_ms=args.switch_total_ms,
        switch_allowed_until_step=args.switch_window,
    )
    config = SimulationConfig(
        total_gpus=args.serve_gpus,
        switch=switch,
        horizon_events=args.horizon_events,
        starvation_wait_ms=args.starvation_ms,
        fairness_beta=args.fairness_beta,
        enable_continuous_batch=not args.no_continuous_batch,
        max_batch_items=args.max_batch_items,
        enable_flexcache=args.enable_flexcache,
        flexcache_min_steps=args.flexcache_min_steps,
        flexcache_max_reduce_steps=args.flexcache_max_reduce_steps,
        flexcache_emergency_tardiness_ms=args.flexcache_emergency_tardiness_ms,
        flexcache_emergency_slack_ms=args.flexcache_emergency_slack_ms,
        decision_log=args.decision_log,
    )

    md_sections = [
        "# Serve Policy Comparison (N-GPU pool)",
        "",
        f"Cost model derived per-shape step latencies; serve_gpus={args.serve_gpus}, "
        f"switch_total_ms={args.switch_total_ms:g}, switch_window={args.switch_window}, "
        f"slo_factor={args.slo_factor:g}"
        + (f", continuous_batch=max{args.max_batch_items}" if not args.no_continuous_batch else ", continuous_batch=off")
        + (f", flexcache=on (min_steps={args.flexcache_min_steps})" if args.enable_flexcache else "")
        + ". Latencies model denoise time only. Latency/queue/tardiness columns in seconds; "
        "eff is useful image-token steps per cluster GPU-second (shown in thousands).",
        "",
    ]
    all_results: dict[str, dict[str, PoolResult]] = {}
    dump: dict = {
        "config": {
            "serve_gpus": args.serve_gpus, "switch_total_ms": args.switch_total_ms,
            "switch_window": args.switch_window, "slo_factor": args.slo_factor,
            "enable_continuous_batch": not args.no_continuous_batch,
            "max_batch_items": args.max_batch_items,
            "enable_flexcache": args.enable_flexcache,
        },
        "traces": {},
    }

    for trace_path in serve_traces:
        trace_name = f"serve/{trace_path.stem}"
        requests, profiles = build_serve_inputs(trace_path, cost_model, slo_factor=args.slo_factor)
        results: dict[str, PoolResult] = {}
        slo: dict[str, dict[str, float]] = {}
        for policy in args.policies:
            result = simulate_pool(requests, profiles, policy, config)
            results[policy] = result
            slo[policy] = _slo_attainment(result, args.slo_seconds)
        all_results[trace_name] = results
        md_sections.append(render_comparison_markdown(trace_name, results, slo, args.slo_seconds))
        dump["traces"][trace_name] = {
            policy: {
                "metrics": results[policy].metrics,
                "slo": slo[policy],
                **({"decision_log": results[policy].decision_log} if args.decision_log and policy == "slo_elastic" else {}),
            }
            for policy in results
        }

    markdown = "\n".join(md_sections)
    print(markdown)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "serve_policies.md").write_text(markdown + "\n")
    (args.output_dir / "serve_policies.json").write_text(json.dumps(dump, indent=2, ensure_ascii=False) + "\n")
    if not args.no_timelines:
        write_timelines(all_results, args.output_dir)


if __name__ == "__main__":
    main()
