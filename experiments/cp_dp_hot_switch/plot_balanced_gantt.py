#!/usr/bin/env python3
"""GPU-occupancy Gantt comparison for the latency-balanced-K / straggler-widening arms.

All arms run the SAME policy (``slo_elastic``, cfp+elastic) at λ=0.36 on 8x H20; they
differ only in the engine/planner flags. Reusing the resource-centric Gantt from
``plot_gpu_gantt`` (per-GPU rows, time-aligned), this shows for each arm:
  1. how the pool is partitioned into lanes each round (sp1 / cfp2xcpK), drawn as
     colored blocks spanning exactly the GPUs a request holds;
  2. each task's arrival tick + faint waiting line to its first execution, then compute;
  3. GPU idle (hatched grey) ONLY when a GPU is genuinely unassigned by the logged
     layout; assigned lane time is drawn solid for the whole interval. This is important
     for latency-balanced K: the old uniform-K visualization estimated a sync-spin tail
     from ``lane_step / max_lane_step`` and incorrectly painted useful extra steps as idle;
  4. the per-round global-barrier ticks as red verticals.

So the reader can see: base burns huge sync-spin tails; balanced-K shrinks them (fast
lanes do their own steps); widening lets the tail 1024^2 straggler grow into the GPUs
that free up instead of running sp1 on one card while the rest idle.
Panels top->bottom follow ``--arms`` (default base,full,widen).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from simulate import load_cost_model  # noqa: E402
from plot_gpu_gantt import (  # noqa: E402
    parse_layouts,
    load_barriers,
    _mode_width_cfp,
    build_step_fn,
)

ARM_LABEL = {
    "base": "base  (roofline cost model, uniform-K)",
    "calib": "+calibration  (online monitor, uniform-K)",
    "full": "+balanced-K  (calibration + latency-balanced K)",
    "widen": "+widen  (balanced-K + straggler widening)",
}


def find_serve_metrics(root: Path, tag: str):
    hits = sorted((root / tag).glob("*/metrics/serve_metrics.json"))
    return hits[0] if hits else None


def load_runtime_stats(metrics_path: Path | None, intervals, end_ms: float, world_size: int):
    """Return real unassigned-GPU idle and measured barrier totals.

    The only idle painted in this corrected plot is layout-level underfill: a GPU absent
    from the rank-0 ``pool] layout`` interval. Barrier/spin is reported numerically from
    timing records instead of estimated per-lane from the offline cost model.
    """
    total_gpu_s = world_size * max(0.0, end_ms) / 1000.0
    idle_gpu_s = 0.0
    used_gpu_s = 0.0
    underfilled_intervals = 0
    for i, (t0, assigns) in enumerate(intervals):
        t1 = intervals[i + 1][0] if i + 1 < len(intervals) else end_ms
        if t1 <= t0:
            continue
        dur_s = (t1 - t0) / 1000.0
        used = len({g for _rid, (_mode, gpus) in assigns.items() for g in gpus})
        idle = max(0, world_size - used)
        used_gpu_s += used * dur_s
        idle_gpu_s += idle * dur_s
        if idle > 0:
            underfilled_intervals += 1

    barrier_s = None
    denoise_s = None
    if metrics_path is not None:
        summary = metrics_path.parent / "timing" / "summary.json"
        if summary.exists():
            d = json.loads(summary.read_text())
            timers = d.get("timers", {})
            barrier_s = float(timers.get("pool_barrier_ms", {}).get("total_ms", 0.0)) / 1000.0
            denoise_s = float(timers.get("pool_denoise_ms", {}).get("total_ms", 0.0)) / 1000.0

    return {
        "used_gpu_s": used_gpu_s,
        "idle_gpu_s": idle_gpu_s,
        "idle_frac": (idle_gpu_s / total_gpu_s) if total_gpu_s > 0 else 0.0,
        "underfilled_intervals": underfilled_intervals,
        "barrier_s": barrier_s,
        "denoise_s": denoise_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/bench_balanced_k")
    ap.add_argument("--arms", default="base,full,widen", help="top->bottom panel order")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost-model", default="h20")
    ap.add_argument("--hw", default="8x H20 (NVLink)")
    ap.add_argument("--rate", default="0.36")
    args = ap.parse_args()

    root = Path(args.root)
    arms = args.arms.split(",")

    trace = json.loads(Path(args.trace).read_text())
    arrival = {r["request_id"]: r["arrival_ms"] for r in trace["requests"]}
    shapes = {r["request_id"]: (r["width"], r["height"]) for r in trace["requests"]}
    step_ms = build_step_fn(load_cost_model(args.cost_model))
    rids = [r["request_id"] for r in trace["requests"]]
    cmap = plt.get_cmap("tab20")
    rcolor = {rid: cmap(i % 20) for i, rid in enumerate(rids)}

    data = {}
    tmax = 0.0
    for tag in arms:
        mp = find_serve_metrics(root, tag)
        finish = {}
        if mp:
            per = json.loads(mp.read_text())["per_request"]
            finish = {rid: rec["finish_ms"] for rid, rec in per.items() if rec.get("finish_ms") is not None}
        _, intervals = parse_layouts(root / "logs" / f"{tag}.log")
        end = max(finish.values()) if finish else (intervals[-1][0] if intervals else 0)
        exec_start = {}
        for t, assigns in intervals:
            for rid in assigns:
                exec_start.setdefault(rid, t)
        barriers, bmeta = (load_barriers(mp) if mp else ([], {}))
        stats = load_runtime_stats(mp, intervals, end, 8)
        data[tag] = {"intervals": intervals, "finish": finish, "exec_start": exec_start,
                     "end": end, "barriers": barriers, "bmeta": bmeta, "stats": stats}
        tmax = max(tmax, end)
    tmax_s = tmax / 1000.0 * 1.02

    NG = 8
    fig, axes = plt.subplots(len(arms), 1, figsize=(16, 3.2 * len(arms)), sharex=True)
    if len(arms) == 1:
        axes = [axes]

    for ax, tag in zip(axes, arms):
        d = data[tag]
        intervals = d["intervals"]
        stats = d["stats"]
        # Interval timestamps come from layout logs, so each interval includes denoise
        # body plus the explicit world barrier/overheads before the next layout. We do
        # not have per-round barrier samples in this run, only totals, so distribute the
        # explicit barrier as a fixed fraction of each interval. The much larger
        # intra-denoise sync-spin for uniform-K is estimated separately below.
        denoise_s = stats.get("denoise_s") or 0.0
        barrier_s = stats.get("barrier_s") or 0.0
        body_frac = denoise_s / (denoise_s + barrier_s) if (denoise_s + barrier_s) > 0 else 1.0
        balanced_k_active = tag in {"full", "widen"}
        labeled = set()
        for i, (t0, assigns) in enumerate(intervals):
            t1 = intervals[i + 1][0] if i + 1 < len(intervals) else d["end"]
            if t1 <= t0:
                continue
            dur = (t1 - t0) / 1000.0
            x0 = t0 / 1000.0
            body_w = dur * body_frac
            barrier_w = max(0.0, dur - body_w)
            used = set()
            lane_step = {}
            for rid, (mode, _gpus) in assigns.items():
                width, cfp = _mode_width_cfp(mode)
                lane_step[rid] = step_ms(shapes.get(rid, (1024, 1024)), width, cfp)
            slowest = max(lane_step.values()) if lane_step else 1.0
            for rid, (mode, gpus) in assigns.items():
                # base/calib: uniform-K => fast lanes finish their K steps early and spin
                # inside ``pool_denoise_ms``. balanced-K/widen: fast lanes run more local
                # steps, so the old uniform-K ratio would be wrong; without per-lane K logs
                # the best faithful view is to treat denoise body as useful assigned work.
                busy_frac = 1.0 if balanced_k_active else (lane_step[rid] / slowest if slowest > 0 else 1.0)
                compute_w = body_w * min(1.0, max(0.0, busy_frac))
                spin_w = max(0.0, body_w - compute_w)
                for g in gpus:
                    used.add(g)
                    ax.barh(g, compute_w, left=x0, height=0.82, color=rcolor[rid],
                            alpha=0.95, edgecolor="black", linewidth=0.3, zorder=2)
                    if spin_w > 1e-6:
                        ax.barh(g, spin_w, left=x0 + compute_w, height=0.82,
                                color=rcolor[rid], alpha=0.30, hatch="xx",
                                edgecolor="black", linewidth=0.25, zorder=2)
                    if barrier_w > 1e-6:
                        ax.barh(g, barrier_w, left=x0 + body_w, height=0.82,
                                color="#ffb3b3", alpha=0.65, hatch="..",
                                edgecolor="#cc3333", linewidth=0.25, zorder=3)
                if rid not in labeled and compute_w > tmax_s * 0.012:
                    ax.text(x0 + 0.5 * compute_w, min(gpus), f"{rid[-2:]}\n{mode}",
                            ha="center", va="center", fontsize=6.0, zorder=4,
                            color="white")
                    labeled.add(rid)
            for g in range(NG):
                if g not in used:
                    ax.barh(g, dur, left=x0, height=0.82, color="#e8e8e8",
                            hatch="///", edgecolor="#c0c0c0", linewidth=0.3, zorder=1)

        ystrip = NG + 0.4
        for rid in rids:
            arr = arrival[rid] / 1000.0
            ax.plot(arr, ystrip, marker="v", color=rcolor[rid], markersize=7,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=5)
            es = d["exec_start"].get(rid)
            if es is not None and es / 1000.0 > arr:
                ax.plot([arr, es / 1000.0], [ystrip, ystrip], color=rcolor[rid],
                        linewidth=1.6, alpha=0.6, zorder=4)

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
        mk = d["finish"]
        if mk:
            subtitle += f"   makespan={max(mk.values()) / 1000.0:.1f}s"
        st = d["stats"]
        subtitle += f"   unassigned-idle={st['idle_frac'] * 100:.1f}%"
        if st["barrier_s"] is not None:
            subtitle += f"   barrier={st['barrier_s']:.1f}s"
        ax.set_title(ARM_LABEL.get(tag, tag) + subtitle, fontsize=10.5, loc="left")
        ax.grid(axis="x", alpha=0.25)

    axes[-1].set_xlabel("time since serve start (s)")
    axes[-1].set_xlim(0, tmax_s)
    legend = [
        Patch(facecolor="gray", alpha=0.95, edgecolor="black", label="estimated useful denoise compute"),
        Patch(facecolor="gray", alpha=0.30, hatch="xx", edgecolor="black", label="estimated intra-denoise sync-spin"),
        Patch(facecolor="#ffb3b3", alpha=0.65, hatch="..", edgecolor="#cc3333", label="explicit global barrier/overhead (distributed)"),
        Patch(facecolor="#e8e8e8", hatch="///", edgecolor="#c0c0c0", label="idle GPU (unassigned)"),
        plt.Line2D([], [], color="k", marker="v", linestyle="none", markersize=7, label="arrival"),
        plt.Line2D([], [], color="k", linewidth=1.6, alpha=0.6, label="waiting (arrival\u2192exec)"),
        plt.Line2D([], [], color="red", linestyle="-", linewidth=1.4, label="global sync / barrier (per round)"),
    ]
    axes[0].legend(handles=legend, fontsize=8, ncol=4, loc="lower left",
                   bbox_to_anchor=(0.0, 1.12))
    fig.suptitle(
        f"cfp+elastic GPU-occupancy timeline @ \u03bb={args.rate} req/s "
        f"({args.hw}, 50 steps, mixed 1024\u00b2/512\u00b2) \u2014 balanced-K + straggler widening",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=135)
    plt.close(fig)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
