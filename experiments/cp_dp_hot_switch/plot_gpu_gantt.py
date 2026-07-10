#!/usr/bin/env python3
"""Resource-centric GPU Gantt for the serving strategies at ONE arrival rate.

Unlike the task-centric timeline, here each panel's rows are the 8 *physical GPUs*
and the x-axis is time, so the picture directly shows:
  1. the GPU parallel-execution strategy -- how the pool is partitioned into lanes
     each round (dp8 = eight width-1 lanes; cp8 = one width-8 lane; cfp+elastic =
     an adaptive mix of sp1 / cfp2xcpK lanes), read from the rank-0 ``pool] layout:``
     logs and drawn as colored blocks spanning exactly the GPUs a request holds;
  2. each task's wait / exec / finish -- an arrival tick with a faint "waiting" line
     to its first execution, then solid colored compute blocks up to finish;
  3. GPU idle / sync-spin time -- unassigned GPUs are hatched grey (pool idle), and
     within a round the barrier makes faster lanes wait for the slowest, so the
     estimated wait fraction (from the cost model) is hatched at the tail of each
     concurrent lane block (sync-spin).

Panels are stacked and share the x-axis (time-aligned). Default order top->bottom is
dp8 (eight lanes), cfp+elastic (adaptive), cp8 (single lane).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from rate_sweep_analyze import _find_metrics  # noqa: E402
from simulate import load_cost_model  # noqa: E402
from chitu_diffusion.runtime.elastic_policy import profile_from_cost_model  # noqa: E402

POLICY_LABEL = {
    "pure_sp": "cp8  (single width-8 lane, serial)",
    "pure_dp": "dp8  (eight width-1 lanes)",
    "slo_elastic": "cfp+elastic  (adaptive: sp1 / cfp2xcpK)",
}

_TS = re.compile(r"^\w+\s+(\d\d-\d\d \d\d:\d\d:\d\d)")
_ADMIT = re.compile(r"admit request=(\w+) at ([\d.]+)ms")
_ASSIGN = re.compile(r"(req\d+):([a-z0-9]+)@\[([0-9, ]+)\]")


def _wall(line: str):
    m = _TS.search(line)
    return datetime.strptime(m.group(1), "%m-%d %H:%M:%S").timestamp() if m else None


def parse_layouts(log_path: Path):
    """Return (fit(wall)->rel_ms, [(start_rel_ms, {req:(mode,[gpus])}), ...])."""
    text = log_path.read_text(errors="ignore")
    anchors, raw = [], []
    for line in text.splitlines():
        w = _wall(line)
        if w is None:
            continue
        a = _ADMIT.search(line)
        if a:
            anchors.append((w, float(a.group(2))))
        if "pool] layout:" in line:
            assigns = {m.group(1): (m.group(2), [int(x) for x in m.group(3).split(",")])
                       for m in _ASSIGN.finditer(line)}
            if assigns:
                raw.append((w, assigns))
    # linear fit rel_ms = slope*wall + b (slope ~ 1000 ms/s)
    n = len(anchors)
    mx = sum(w for w, _ in anchors) / n
    my = sum(r for _, r in anchors) / n
    den = sum((w - mx) ** 2 for w, _ in anchors) or 1.0
    slope = sum((w - mx) * (r - my) for w, r in anchors) / den
    b = my - slope * mx
    fit = lambda w: slope * w + b  # noqa: E731
    intervals = [(fit(w), assigns) for w, assigns in raw]
    intervals.sort(key=lambda t: t[0])
    return fit, intervals


def load_barriers(metrics_path: Path):
    """Reconstruct the absolute time (serve-relative ms) of each global barrier from
    the per-round ``records.pool_round`` list. The engine barriers once per round; the
    records carry per-round rank-0 denoise but no wall clock, so we place the R barrier
    ticks at the cumulative rank-0 compute normalized to the measured serving span
    (exact count + end, approximate interior spacing). Returns ([rel_ms...], meta)."""
    summ = metrics_path.parent / "timing" / "summary.json"
    if not summ.exists():
        return [], {}
    d = json.loads(summ.read_text())
    recs = d.get("records", {}).get("pool_round", [])
    if not recs:
        return [], {}
    serving_ms = float(d.get("serving_elapsed_s", 0.0)) * 1000.0
    weights = [max(1e-6, float(r.get("rank0_denoise_ms", 1.0))) for r in recs]
    total = sum(weights)
    barriers, acc = [], 0.0
    for w in weights:
        acc += w
        barriers.append(serving_ms * acc / total)
    solo_idle = sum(1 for r in recs if int(r.get("used_gpus", 0)) < int(r.get("world_size", 8)))
    meta = {"n_rounds": len(recs), "underfilled_rounds": solo_idle}
    return barriers, meta


def _mode_width_cfp(mode: str):
    """(width, uses_cfp) from a layout tag: 'sp8'->(8,False); 'cfp2xcp4'->(8,True)."""
    m = re.fullmatch(r"cfp2xcp(\d+)", mode)
    if m:
        return 2 * int(m.group(1)), True
    m = re.fullmatch(r"sp(\d+)", mode)
    if m:
        return int(m.group(1)), False
    return 1, False


def build_step_fn(model):
    _MODE = {1: "dp1", 2: "cp2", 4: "cp4", 8: "cp8"}
    cache: dict = {}

    def step_ms(shape, width, uses_cfp):
        key = (shape, width, uses_cfp)
        if key not in cache:
            w, h = shape
            prof = profile_from_cost_model(model, name="g", width=w, height=h, num_steps=50,
                                           guidance_scale=2.0, cfg_parallel_max=(2 if uses_cfp else 1))
            cache[key] = prof.step_ms(_MODE.get(width, "dp1"), 0)
        return cache[key]

    return step_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rate-tag", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--logs", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policies", default="pure_dp,slo_elastic,pure_sp",
                    help="top->bottom panel order")
    ap.add_argument("--cost-model", default="h20")
    ap.add_argument("--hw", default="8x H20 (NVLink)")
    args = ap.parse_args()

    root = Path(args.root)
    logs_dir = Path(args.logs) if args.logs else root / "logs"
    policies = args.policies.split(",")
    rate = float(args.rate_tag.replace("p", "."))

    trace = json.loads(Path(args.trace).read_text())
    arrival = {r["request_id"]: r["arrival_ms"] for r in trace["requests"]}
    shapes = {r["request_id"]: (r["width"], r["height"]) for r in trace["requests"]}
    deadlines = {r["request_id"]: r.get("deadline_ms") for r in trace["requests"]}
    step_ms = build_step_fn(load_cost_model(args.cost_model))

    # stable per-request color
    rids = [r["request_id"] for r in trace["requests"]]
    cmap = plt.get_cmap("tab20")
    rcolor = {rid: cmap(i % 20) for i, rid in enumerate(rids)}

    # gather per-policy data
    data = {}
    tmax = 0.0
    for pol in policies:
        mp = _find_metrics(root, args.rate_tag, pol)
        finish = {}
        if mp:
            per = json.loads(mp.read_text())["per_request"]
            finish = {rid: rec["finish_ms"] for rid, rec in per.items() if rec.get("finish_ms") is not None}
        _, intervals = parse_layouts(logs_dir / f"r{args.rate_tag}_{pol}.log")
        end = max(finish.values()) if finish else (intervals[-1][0] if intervals else 0)
        # exec-start per request = first interval it appears
        exec_start = {}
        for t, assigns in intervals:
            for rid in assigns:
                exec_start.setdefault(rid, t)
        barriers, bmeta = (load_barriers(mp) if mp else ([], {}))
        data[pol] = {"intervals": intervals, "finish": finish, "exec_start": exec_start,
                     "end": end, "barriers": barriers, "bmeta": bmeta}
        tmax = max(tmax, end)
    tmax_s = tmax / 1000.0 * 1.02

    NG = 8
    fig, axes = plt.subplots(len(policies), 1, figsize=(16, 3.2 * len(policies)), sharex=True)
    if len(policies) == 1:
        axes = [axes]

    for ax, pol in zip(axes, policies):
        d = data[pol]
        intervals = d["intervals"]
        labeled = set()
        for i, (t0, assigns) in enumerate(intervals):
            t1 = intervals[i + 1][0] if i + 1 < len(intervals) else d["end"]
            if t1 <= t0:
                continue
            dur = (t1 - t0) / 1000.0
            x0 = t0 / 1000.0
            used = set()
            # bottleneck step for this interval (barrier paces to the slowest lane)
            lane_step = {}
            for rid, (mode, gpus) in assigns.items():
                w, cfp = _mode_width_cfp(mode)
                lane_step[rid] = step_ms(shapes.get(rid, (1024, 1024)), w, cfp)
            tmaxstep = max(lane_step.values()) if lane_step else 1.0
            for rid, (mode, gpus) in assigns.items():
                busy_frac = lane_step[rid] / tmaxstep if tmaxstep > 0 else 1.0
                busy_w = dur * busy_frac
                for g in gpus:
                    used.add(g)
                    ax.barh(g, busy_w, left=x0, height=0.82, color=rcolor[rid],
                            alpha=0.95, edgecolor="black", linewidth=0.3, zorder=2)
                    if busy_frac < 0.999:  # barrier sync-spin tail
                        ax.barh(g, dur - busy_w, left=x0 + busy_w, height=0.82,
                                color=rcolor[rid], alpha=0.35, hatch="xx",
                                edgecolor="black", linewidth=0.3, zorder=2)
                if rid not in labeled and busy_w > tmax_s * 0.012:
                    ax.text(x0 + 0.5 * busy_w, min(gpus), f"{rid[-2:]}\n{mode}",
                            ha="center", va="center", fontsize=6.0, zorder=4,
                            color="white" if busy_frac > 0.5 else "black")
                    labeled.add(rid)
            # pool-idle GPUs this interval
            for g in range(NG):
                if g not in used:
                    ax.barh(g, dur, left=x0, height=0.82, color="#e8e8e8",
                            hatch="///", edgecolor="#c0c0c0", linewidth=0.3, zorder=1)

        # arrivals strip (above GPU rows): arrival tick + faint wait line to exec-start
        ystrip = NG + 0.4
        for rid in rids:
            arr = arrival[rid] / 1000.0
            ax.plot(arr, ystrip, marker="v", color=rcolor[rid], markersize=7,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=5)
            es = d["exec_start"].get(rid)
            if es is not None and es / 1000.0 > arr:
                ax.plot([arr, es / 1000.0], [ystrip, ystrip], color=rcolor[rid],
                        linewidth=1.6, alpha=0.6, zorder=4)  # waiting

        # global-barrier lines: one per pool round (engine syncs the whole world here).
        # Drawn as bold RED verticals on top of everything so the sync cadence is obvious.
        for bx in d["barriers"]:
            ax.axvline(bx / 1000.0, ymin=0.02, ymax=0.90, color="red",
                       linestyle="-", linewidth=1.1, alpha=0.9, zorder=6)

        ax.set_ylim(-0.6, NG + 0.9)
        ax.set_yticks(list(range(NG)) + [ystrip])
        ax.set_yticklabels([f"GPU{g}" for g in range(NG)] + ["arrivals"], fontsize=7)
        bm = d["bmeta"]
        subtitle = ""
        if bm:
            subtitle = f"   [{bm['n_rounds']} global syncs; {bm['underfilled_rounds']} rounds under-fill the pool]"
        ax.set_title(POLICY_LABEL.get(pol, pol) + subtitle, fontsize=10.5, loc="left")
        ax.grid(axis="x", alpha=0.25)

    axes[-1].set_xlabel("time since serve start (s)")
    axes[-1].set_xlim(0, tmax_s)
    legend = [
        Patch(facecolor="gray", alpha=0.95, edgecolor="black", label="compute (lane on its GPUs)"),
        Patch(facecolor="gray", alpha=0.35, hatch="xx", edgecolor="black", label="sync-spin (barrier wait, est.)"),
        Patch(facecolor="#e8e8e8", hatch="///", edgecolor="#c0c0c0", label="idle GPU (unassigned)"),
        plt.Line2D([], [], color="k", marker="v", linestyle="none", markersize=7, label="arrival"),
        plt.Line2D([], [], color="k", linewidth=1.6, alpha=0.6, label="waiting (arrival\u2192exec)"),
        plt.Line2D([], [], color="red", linestyle="-", linewidth=1.4, label="global sync / barrier (per round)"),
    ]
    axes[0].legend(handles=legend, fontsize=8, ncol=6, loc="lower left",
                   bbox_to_anchor=(0.0, 1.12))
    fig.suptitle(f"GPU occupancy timeline @ \u03bb={rate:g} req/s ({args.hw}, 50 steps, mixed 1024\u00b2/512\u00b2)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=135)
    plt.close(fig)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
