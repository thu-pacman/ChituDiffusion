#!/usr/bin/env python3
"""Analyze an arrival-rate sweep: for each rate x each strategy (cp8/dp8/cfp+elastic),
read the run's serve_metrics.json, derive service metrics, and plot them vs arrival
rate (one line per strategy). This shows whether cfp+elastic holds a good trade-off
across load regimes instead of only at one operating point.

Layout expected (produced by run_sweep.sh):
    <root>/r<tag>/<policy>-<ts>-<taskid>/metrics/serve_metrics.json
with a manifest.json (rate -> trace path) written by gen_sweep_traces.py living in
the traces dir passed via --manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POLICY_ORDER = ["pure_sp", "pure_dp", "slo_elastic"]
POLICY_COLOR = {"pure_sp": "#d62728", "pure_dp": "#1f77b4", "slo_elastic": "#2ca02c"}
POLICY_LABEL = {
    "pure_sp": "cp8 (Ulysses SP=8, serial)",
    "pure_dp": "dp8 (8x DP, SP=1)",
    "slo_elastic": "cfp+elastic (DP>CFP>CP)",
}
POLICY_MARKER = {"pure_sp": "s", "pure_dp": "o", "slo_elastic": "^"}


def _pctl(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if xs else 0.0


def _attach_offline_lane_elapsed(summary: dict, metrics_path: Path) -> list[dict]:
    """Merge rank-local lane timing into rank0 pool_round records.

    ``summary_rank*.json`` files are written without in-band collectives. Offline, the max
    local elapsed across ranks in the same lane is the lane's real compute span. This is
    the same clock source used by the straggler timeline plot.
    """
    timing_dir = metrics_path.parent / "timing"
    local_by_round_lane: dict[tuple[int, int], float] = {}
    for path in sorted(timing_dir.glob("summary_rank*.json")):
        data = json.loads(path.read_text())
        for ev in data.get("records", {}).get("pool_lane_local", []):
            key = (int(ev.get("round_index", -1)), int(ev.get("lane_index", -1)))
            if key[0] < 0 or key[1] < 0:
                continue
            local_by_round_lane[key] = max(
                local_by_round_lane.get(key, 0.0),
                float(ev.get("local_compute_ms", 0.0)),
            )

    records = summary.get("records", {}).get("pool_round", [])
    for rec in records:
        span = 0.0
        for lane in rec.get("lanes", []) or []:
            key = (int(rec.get("round_index", -1)), int(lane.get("lane_index", -1)))
            if key in local_by_round_lane:
                lane["local_compute_ms"] = local_by_round_lane[key]
                span = max(span, local_by_round_lane[key])
        if span > 0.0:
            rec["denoise_span_ms"] = span
    return records


def _reconstruct_first_exec_ms(metrics_path: Path, per_request: dict) -> dict[str, float]:
    """True first GPU execution time per request, reconstructed from pool layout records.

    The harness's ``first_sched_ms`` is a scheduler-observation timestamp: in cp8 a request
    can be considered scheduled/inside the pool while still waiting behind a full-width
    request. For queueing analysis we instead need the first pool round whose lane actually
    contains the request. ``pool_round`` records omit wall-clock gaps while the pool is empty,
    so when consecutive active segments have disjoint request ids we anchor the new segment to
    the earliest ``first_sched_ms`` of that segment.
    """
    timing_path = metrics_path.parent / "timing" / "summary.json"
    if not timing_path.exists():
        return {}
    records = _attach_offline_lane_elapsed(json.loads(timing_path.read_text()), metrics_path)
    first_sched = {rid: rec.get("first_sched_ms") for rid, rec in per_request.items()}

    t = 0.0
    prev_active: set[str] = set()
    first_exec: dict[str, float] = {}
    for rec in records:
        lanes = rec.get("lanes") or []
        if not lanes:
            continue
        active: set[str] = set()
        for lane in lanes:
            rid = str(lane.get("request_id") or "")
            if rid:
                active.add(rid)
            active.update(str(x) for x in (lane.get("batch_ids") or []) if x)
        if active and (not prev_active or prev_active.isdisjoint(active)):
            scheds = [float(first_sched[rid]) for rid in active if first_sched.get(rid) is not None]
            if scheds:
                t = max(t, min(scheds))
        for rid in active:
            first_exec.setdefault(rid, t)
        denoise_ms = max(0.0, float(rec.get("denoise_span_ms", rec.get("rank0_denoise_ms", 0.0))))
        round_ms = max(denoise_ms, float(rec.get("pool_round_ms", denoise_ms)))
        t += round_ms
        prev_active = active
    return first_exec


def load_arm(metrics_path: Path, deadlines: dict) -> dict:
    data = json.loads(metrics_path.read_text())
    summary = data["summary"]
    first_exec = _reconstruct_first_exec_ms(metrics_path, data["per_request"])
    rows = []
    for rid, rec in data["per_request"].items():
        arr = rec.get("trace_arrival_ms")
        fin = rec.get("finish_ms")
        if arr is None or fin is None:
            continue
        lat = fin - arr
        dl = deadlines.get(rid)
        sched_delay = (rec.get("first_sched_ms") - arr) if rec.get("first_sched_ms") is not None else None
        exec_wait = (first_exec.get(rid) - arr) if first_exec.get(rid) is not None else sched_delay
        rows.append({"latency_ms": lat, "deadline_ms": dl,
                     "met": (dl is not None and lat <= dl),
                     "sched_delay_ms": sched_delay, "exec_wait_ms": exec_wait})
    lats = [r["latency_ms"] for r in rows]
    with_dl = [r for r in rows if r["deadline_ms"] is not None]
    slo_ok = sum(1 for r in with_dl if r["met"])
    sched_delays = [r["sched_delay_ms"] for r in rows if r["sched_delay_ms"] is not None]
    exec_waits = [r["exec_wait_ms"] for r in rows if r["exec_wait_ms"] is not None]
    return {
        "makespan_s": summary["makespan_ms"] / 1000.0,
        "throughput_req_per_s": summary["throughput_req_per_s"],
        "mean_latency_s": (mean(lats) / 1000.0) if lats else 0.0,
        "p50_latency_s": _pctl(lats, 50) / 1000.0,
        "p95_latency_s": _pctl(lats, 95) / 1000.0,
        "p99_latency_s": _pctl(lats, 99) / 1000.0,
        "max_latency_s": (max(lats) / 1000.0) if lats else 0.0,
        "mean_sched_delay_s": (mean(sched_delays) / 1000.0) if sched_delays else 0.0,
        "mean_exec_wait_s": (mean(exec_waits) / 1000.0) if exec_waits else 0.0,
        "p95_exec_wait_s": _pctl(exec_waits, 95) / 1000.0,
        "slo_attainment": (slo_ok / len(with_dl)) if with_dl else None,
        "num_completed": summary.get("num_completed"),
    }


def _find_metrics(root: Path, tag: str, policy: str) -> Path | None:
    rdir = root / f"r{tag}"
    if not rdir.is_dir():
        return None
    cands = sorted(
        d
        for d in rdir.iterdir()
        if d.is_dir()
        and (
            d.name.replace("-", "_").startswith(policy + "_")
            or f"_{policy}_" in d.name.replace("-", "_")
        )
    )
    for d in reversed(cands):  # newest run wins
        mp = d / "metrics" / "serve_metrics.json"
        if mp.exists():
            return mp
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="sweep output root (holds r<tag>/ dirs)")
    ap.add_argument("--manifest", required=True, help="manifest.json from gen_sweep_traces.py")
    ap.add_argument("--out", default=None, help="output dir (default <root>/comparison)")
    ap.add_argument("--hw", default="8x H20 (NVLink)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())["sweep"]
    manifest.sort(key=lambda m: m["rate"])

    # rate -> policy -> derived metrics
    data: dict[float, dict[str, dict]] = {}
    for m in manifest:
        rate = float(m["rate"])
        tag = f"{rate:g}".replace(".", "p")
        trace = json.loads(Path(m["trace"]).read_text())
        deadlines = {r["request_id"]: r.get("deadline_ms") for r in trace["requests"]}
        per_pol = {}
        for pol in POLICY_ORDER:
            mp = _find_metrics(root, tag, pol)
            per_pol[pol] = load_arm(mp, deadlines) if mp else None
        data[rate] = per_pol

    rates = [float(m["rate"]) for m in manifest]

    def series(pol: str, key: str):
        xs, ys = [], []
        for r in rates:
            d = data[r].get(pol)
            if d is None or d.get(key) is None:
                continue
            xs.append(r)
            ys.append(d[key])
        return xs, ys

    # --- Multi-panel line chart: metric vs arrival rate, one line per strategy ---
    panels = [
        ("slo_attainment", "SLO attainment", "fraction", True),
        ("p95_latency_s", "p95 latency", "seconds", False),
        ("mean_latency_s", "mean latency", "seconds", False),
        ("mean_exec_wait_s", "mean true exec wait", "seconds", False),
        ("makespan_s", "makespan", "seconds", False),
        ("throughput_req_per_s", "throughput", "req/s", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, title, ylabel, is_frac) in zip(axes.flat, panels):
        for pol in POLICY_ORDER:
            xs, ys = series(pol, key)
            if is_frac:
                ys = [y * 100 for y in ys]
            ax.plot(xs, ys, marker=POLICY_MARKER[pol], color=POLICY_COLOR[pol],
                    label=POLICY_LABEL[pol], linewidth=2, markersize=7)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("arrival rate (req/s)")
        ax.set_ylabel("%" if is_frac else ylabel)
        if is_frac:
            ax.set_ylim(-3, 105)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=9, loc="best")
    fig.suptitle(f"Serving metrics vs arrival rate ({args.hw}, 50 steps, mixed 1024\u00b2/512\u00b2)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "sweep_metrics.png", dpi=130)
    plt.close(fig)

    # --- CSV ---
    cols = ["slo_attainment", "p95_latency_s", "mean_latency_s", "mean_exec_wait_s",
            "p95_exec_wait_s", "mean_sched_delay_s", "makespan_s",
            "throughput_req_per_s", "num_completed"]
    lines = ["rate,policy," + ",".join(cols)]
    for r in rates:
        for pol in POLICY_ORDER:
            d = data[r].get(pol)
            if d is None:
                lines.append(f"{r},{pol}," + ",".join(["n/a"] * len(cols)))
                continue
            lines.append(f"{r},{pol}," + ",".join(
                (f"{d[c]:.4f}" if isinstance(d[c], (int, float)) else str(d[c])) for c in cols))
    (out / "sweep.csv").write_text("\n".join(lines) + "\n")
    (out / "sweep.json").write_text(json.dumps(
        {str(r): {p: data[r][p] for p in POLICY_ORDER} for r in rates}, indent=2) + "\n")

    # --- Markdown report ---
    md = [f"# Arrival-rate sweep: serving strategies ({args.hw})\n",
          "Same workload (mixed 1024\u00b2/512\u00b2, 50 steps, fixed seed) replayed at increasing "
          "Poisson arrival rates. Only inter-arrival density changes across points, so the "
          "x-axis is a clean load axis. Deadlines are a fixed per-shape latency budget.\n",
          "Arms: **cp8** (pure_sp), **dp8** (pure_dp), **cfp+elastic** (slo_elastic, DP>CFP>CP).\n"]
    md.append("## SLO attainment (%) vs arrival rate\n")
    md.append("| rate (req/s) | " + " | ".join(POLICY_LABEL[p] for p in POLICY_ORDER) + " |")
    md.append("|---|" + "---|" * len(POLICY_ORDER))
    for r in rates:
        cells = []
        for pol in POLICY_ORDER:
            d = data[r].get(pol)
            v = d["slo_attainment"] if d else None
            cells.append(f"{v*100:.0f}%" if v is not None else "n/a")
        md.append(f"| {r:g} | " + " | ".join(cells) + " |")
    md.append("\n## p95 latency (s) vs arrival rate\n")
    md.append("| rate (req/s) | " + " | ".join(POLICY_LABEL[p] for p in POLICY_ORDER) + " |")
    md.append("|---|" + "---|" * len(POLICY_ORDER))
    for r in rates:
        cells = []
        for pol in POLICY_ORDER:
            d = data[r].get(pol)
            cells.append(f"{d['p95_latency_s']:.1f}" if d else "n/a")
        md.append(f"| {r:g} | " + " | ".join(cells) + " |")
    md.append("\n## mean true execution wait (s) vs arrival rate\n")
    md.append("| rate (req/s) | " + " | ".join(POLICY_LABEL[p] for p in POLICY_ORDER) + " |")
    md.append("|---|" + "---|" * len(POLICY_ORDER))
    for r in rates:
        cells = []
        for pol in POLICY_ORDER:
            d = data[r].get(pol)
            cells.append(f"{d['mean_exec_wait_s']:.1f}" if d else "n/a")
        md.append(f"| {r:g} | " + " | ".join(cells) + " |")
    md.append("\n`mean_sched_delay_s` in the CSV is the harness/scheduler-observed delay "
              "(`first_sched_ms-arrival`). It is not true queueing under serial cp8; "
              "use `mean_exec_wait_s` for request wait until first GPU execution.\n")
    md.append("\n![sweep metrics](sweep_metrics.png)\n")
    (out / "report.md").write_text("\n".join(md) + "\n")

    print("Wrote sweep analysis to", out)
    for r in rates:
        parts = []
        for pol in POLICY_ORDER:
            d = data[r].get(pol)
            if d is None:
                parts.append(f"{pol}=MISSING")
            else:
                slo = f"{d['slo_attainment']*100:.0f}%" if d["slo_attainment"] is not None else "n/a"
                parts.append(f"{pol}:SLO={slo},p95={d['p95_latency_s']:.0f}s,wait={d['mean_exec_wait_s']:.0f}s")
        print(f"  rate={r:<5g} " + "  ".join(parts))


if __name__ == "__main__":
    main()
