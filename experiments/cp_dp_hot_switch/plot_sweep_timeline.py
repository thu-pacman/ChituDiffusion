#!/usr/bin/env python3
"""Per-request timeline (Gantt) comparison of the three serving strategies at ONE
arrival rate. For each request we split its lifetime into:
  * queue wait  (arrival -> first_sched)  = faded bar
  * service     (first_sched -> finish)   = solid bar
colored per strategy, hatched for 1024^2 vs 512^2, with an arrival tick and a
per-request deadline marker (green '*' if met, red 'x' if missed). This exposes
WHERE the latency goes (queueing under saturation vs compute) across strategies.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

POLICY_ORDER = ["pure_sp", "pure_dp", "slo_elastic"]
POLICY_COLOR = {"pure_sp": "#d62728", "pure_dp": "#1f77b4", "slo_elastic": "#2ca02c"}
POLICY_LABEL = {
    "pure_sp": "cp8 (Ulysses SP=8, serial)",
    "pure_dp": "dp8 (8x DP, SP=1)",
    "slo_elastic": "cfp+elastic (DP>CFP>CP)",
}


_TS = re.compile(r"^\w+\s+(\d\d-\d\d \d\d:\d\d:\d\d)")
_ADMIT = re.compile(r"admit request=(\w+) at ([\d.]+)ms")
_LAYOUT_RID = re.compile(r"(req\d+):")


def _wall_secs(line: str) -> float | None:
    m = _TS.search(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%m-%d %H:%M:%S").timestamp()


def parse_exec_starts(log_path: Path) -> dict[str, float]:
    """True per-request execution-start time (serve-relative ms), read from the
    rank-0 run log: a request first *runs* when it first appears in a ``pool] layout:``
    line. Wall-clock (1s resolution) is calibrated to serve-relative ms via the
    ``admit ... at <ms>`` lines (which carry both), so the split is accurate to ~1s.
    Returns {} if the log is unusable (caller then falls back to first_sched_ms)."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    anchors: list[tuple[float, float]] = []  # (wall_secs, rel_ms)
    first_layout_wall: dict[str, float] = {}
    for line in text.splitlines():
        wall = _wall_secs(line)
        if wall is None:
            continue
        a = _ADMIT.search(line)
        if a:
            anchors.append((wall, float(a.group(2))))
            continue
        if "pool] layout:" in line:
            for rid in _LAYOUT_RID.findall(line):
                first_layout_wall.setdefault(rid, wall)
    if len(anchors) < 2 or not first_layout_wall:
        return {}
    # Robust linear fit rel_ms = slope*wall + intercept (slope ~ 1000 ms/s).
    xs = [w for w, _ in anchors]
    ys = [r for _, r in anchors]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    intercept = my - slope * mx
    return {rid: slope * w + intercept for rid, w in first_layout_wall.items()}


def load_rows(metrics_path: Path, deadlines: dict, shapes: dict, exec_starts: dict | None = None) -> list[dict]:
    data = json.loads(metrics_path.read_text())
    rows = []
    for rid, rec in data["per_request"].items():
        arr = rec.get("trace_arrival_ms")
        fin = rec.get("finish_ms")
        if arr is None or fin is None:
            continue
        # Prefer the log-derived true execution start (arrival->run); fall back to the
        # harness first_sched_ms (fires at admission, so it under-reports queueing).
        sched = None
        if exec_starts and rid in exec_starts:
            sched = max(arr, exec_starts[rid])
        if sched is None:
            sched = rec.get("first_sched_ms")
        dl = deadlines.get(rid)
        rows.append({
            "rid": rid, "arr": arr, "sched": sched if sched is not None else arr,
            "fin": fin, "dl": dl, "lat": fin - arr,
            "met": (dl is not None and (fin - arr) <= dl),
            "is1024": shapes.get(rid, (1024, 1024))[0] >= 1024,
        })
    rows.sort(key=lambda r: (r["arr"], r["rid"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rate-tag", required=True, help="e.g. 0p36")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", default=None, help="dir with r<tag>_<policy>.log (for true exec-start)")
    ap.add_argument("--hw", default="8x H20 (NVLink)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from rate_sweep_analyze import _find_metrics

    trace = json.loads(Path(args.trace).read_text())
    deadlines = {r["request_id"]: r.get("deadline_ms") for r in trace["requests"]}
    shapes = {r["request_id"]: (r["width"], r["height"]) for r in trace["requests"]}
    rate = float(args.rate_tag.replace("p", "."))

    logs_dir = Path(args.logs) if args.logs else Path(args.root) / "logs"
    arms = {}
    exec_src = {}
    for pol in POLICY_ORDER:
        mp = _find_metrics(Path(args.root), args.rate_tag, pol)
        if mp is None:
            raise SystemExit(f"missing metrics for {pol} @ {args.rate_tag}")
        exec_starts = parse_exec_starts(logs_dir / f"r{args.rate_tag}_{pol}.log")
        exec_src[pol] = "log" if exec_starts else "first_sched_ms"
        arms[pol] = load_rows(mp, deadlines, shapes, exec_starts)
    print("exec-start source per policy:", exec_src)

    # shared time origin + x-limit across panels for fair comparison.
    t0 = min(r["arr"] for pol in POLICY_ORDER for r in arms[pol])
    tmax = max(r["fin"] for pol in POLICY_ORDER for r in arms[pol])
    xlim = (tmax - t0) / 1000.0 * 1.02

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
    for ax, pol in zip(axes, POLICY_ORDER):
        rows = arms[pol]
        for j, r in enumerate(rows):
            arr = (r["arr"] - t0) / 1000.0
            sched = (r["sched"] - t0) / 1000.0
            fin = (r["fin"] - t0) / 1000.0
            hatch = None if r["is1024"] else "///"
            # queue wait: a distinct NEUTRAL gray (same in every panel) so the eye reads
            # "waiting" independent of the policy color; only real compute is colored.
            if sched > arr:
                ax.barh(j, sched - arr, left=arr, height=0.62, color="#bdbdbd",
                        alpha=0.85, edgecolor="#7f7f7f", linewidth=0.4)
            # service: the policy color (solid). Hatched only for 512^2 shapes.
            ax.barh(j, fin - sched, left=sched, height=0.62, color=POLICY_COLOR[pol],
                    alpha=0.95, edgecolor="black", linewidth=0.4, hatch=hatch)
            ax.plot(arr, j, "k|", markersize=9)  # arrival tick
            # deadline marker (absolute = arrival + budget)
            if r["dl"] is not None:
                dl_abs = arr + r["dl"] / 1000.0
                ax.plot(dl_abs, j, marker=("*" if r["met"] else "x"),
                        color=("#1a9641" if r["met"] else "#d7191c"),
                        markersize=(11 if r["met"] else 9),
                        markeredgewidth=1.8, zorder=5)
        met = sum(1 for r in rows if r["met"])
        ntot = sum(1 for r in rows if r["dl"] is not None)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["rid"] for r in rows], fontsize=7)
        ax.set_xlim(0, xlim)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("time since first arrival (s)")
        ax.set_title(f"{POLICY_LABEL[pol]}\nSLO met {met}/{ntot}", fontsize=10)

    legend = [
        Patch(facecolor="#bdbdbd", alpha=0.85, edgecolor="#7f7f7f", label="queue wait (arrival\u2192scheduled)"),
        Patch(facecolor="gray", alpha=0.95, edgecolor="black", label="service / compute (scheduled\u2192finish)"),
        Patch(facecolor="gray", alpha=0.95, hatch="///", edgecolor="black", label="512\u00b2 (hatched); solid=1024\u00b2"),
        plt.Line2D([], [], color="k", marker="|", linestyle="none", markersize=9, label="arrival"),
        plt.Line2D([], [], color="#1a9641", marker="*", linestyle="none", markersize=11, label="deadline met"),
        plt.Line2D([], [], color="#d7191c", marker="x", linestyle="none", markersize=9, label="deadline missed"),
    ]
    axes[0].legend(handles=legend, fontsize=8, loc="lower right")
    fig.suptitle(f"Request timeline @ \u03bb={rate:g} req/s ({args.hw}, 50 steps, mixed 1024\u00b2/512\u00b2)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=135)
    plt.close(fig)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
