#!/usr/bin/env python3
"""Round-accurate GPU timeline from instrumented ``pool_round`` records.

This plot is meant to answer: *who is the straggler in each synchronous pool round?*

It requires runs produced after the instrumentation that writes, for every pool round:
``pool_round_ms``, ``rank0_denoise_ms``, ``pool_barrier_ms`` and a ``lanes`` list with
``request_id/mode/gpus/steps_run/pred_work_ms``.  The x-axis is reconstructed by
accumulating measured round durations, anchored to the first rank-0 layout log so it
stays aligned with request arrivals / finish times.
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
from plot_gpu_gantt import parse_layouts  # noqa: E402


LABELS = {
    "dp8": "static dp8",
    "cp8": "static cp8",
    "elastic": "elastic (balanced-K + straggler widening)",
}


def find_serve_metrics(root: Path, tag: str) -> Path | None:
    hits = sorted((root / tag).glob("*/metrics/serve_metrics.json"))
    return hits[0] if hits else None


def load_summary(metrics_path: Path) -> dict:
    return json.loads((metrics_path.parent / "timing" / "summary.json").read_text())


def attach_offline_lane_elapsed(summary: dict, metrics_path: Path) -> dict:
    """Merge non-intrusive per-rank lane-local timing into rank0 pool_round records.

    Each rank writes ``summary_rank{rank}.json`` after the run. Those files contain
    local ``pool_lane_local`` events with no extra runtime collectives.  Offline, we take
    the max local elapsed across ranks in the same lane as that lane's real compute span.
    """
    timing_dir = metrics_path.parent / "timing"
    local_by_round_lane: dict[tuple[int, int], dict] = {}
    for path in sorted(timing_dir.glob("summary_rank*.json")):
        data = json.loads(path.read_text())
        for ev in data.get("records", {}).get("pool_lane_local", []):
            key = (int(ev.get("round_index", -1)), int(ev.get("lane_index", -1)))
            if key[0] < 0 or key[1] < 0:
                continue
            item = local_by_round_lane.setdefault(
                key, {"local_compute_ms": 0.0, "steps_run": 0, "rank_compute_ms": []}
            )
            ms = float(ev.get("local_compute_ms", 0.0))
            rank = int(ev.get("rank", -1))
            item["local_compute_ms"] = max(float(item["local_compute_ms"]), ms)
            item["steps_run"] = max(int(item["steps_run"]), int(ev.get("steps_run", 0)))
            item["rank_compute_ms"].append([rank, ms])

    records = summary.get("records", {}).get("pool_round", [])
    for rec in records:
        denoise_span = 0.0
        for lane in rec.get("lanes", []) or []:
            key = (int(rec.get("round_index", -1)), int(lane.get("lane_index", -1)))
            local = local_by_round_lane.get(key)
            if not local:
                continue
            lane["local_compute_ms"] = float(local["local_compute_ms"])
            lane["local_sync_wait_ms"] = 0.0  # filled after denoise span is known
            lane["rank_compute_ms"] = sorted(local["rank_compute_ms"])
            lane["steps_run"] = int(local["steps_run"] or lane.get("steps_run", 0))
            denoise_span = max(denoise_span, float(local["local_compute_ms"]))
        if denoise_span > 0.0:
            rec["denoise_span_ms"] = denoise_span
            for lane in rec.get("lanes", []) or []:
                if "local_compute_ms" in lane:
                    lane["local_sync_wait_ms"] = max(0.0, denoise_span - float(lane["local_compute_ms"]))
    return summary


def first_layout_ms(root: Path, tag: str) -> float:
    log_path = root / "logs" / f"{tag}.log"
    try:
        _fit, intervals = parse_layouts(log_path)
    except Exception:
        return 0.0
    return float(intervals[0][0]) if intervals else 0.0


def lane_color_map(rids: list[str]):
    cmap = plt.get_cmap("tab20")
    return {rid: cmap(i % 20) for i, rid in enumerate(rids)}


def draw_panel(ax, tag: str, d: dict, arrivals: dict[str, float], colors: dict[str, tuple], world_size: int):
    records = d["records"]
    start_ms = d["first_layout_ms"]
    end_ms = d["end_ms"]
    exec_start = d["exec_start"]
    rids = list(arrivals)

    t = start_ms
    straggler_labels = set()
    tail_total_ms = 0.0
    prev_active_ids: set[str] = set()
    for rec in records:
        lanes = rec.get("lanes") or []
        if not lanes:
            # New instrumentation is required for straggler-aware plotting.
            continue
        current_ids: set[str] = set()
        for lane in lanes:
            rid = str(lane.get("request_id") or "")
            if rid:
                current_ids.add(rid)
            current_ids.update(str(x) for x in (lane.get("batch_ids") or []) if x)
        # ``pool_round`` records are emitted only for active work. When the pool drains
        # under a low arrival rate, there are no records for the wall-clock interval spent
        # waiting for the next request. Preserve that idle gap by anchoring a new disjoint
        # run segment to the first scheduled time of the requests in this round.
        if current_ids and (not prev_active_ids or prev_active_ids.isdisjoint(current_ids)):
            scheds = [
                float(exec_start[rid])
                for rid in current_ids
                if exec_start.get(rid) is not None
            ]
            if scheds:
                t = max(t, min(scheds))
        # Honest wall-clock model of one synchronous round:
        #   * each lane computes independently for its own local elapsed;
        #   * the world barrier releases when the SLOWEST lane arrives, i.e. at
        #     denoise_span = max lane local elapsed. A lane that finished earlier sits
        #     idle ON ITS OWN gpus from its finish until denoise_span -- this real bubble
        #     OVERLAPS the straggler's compute and is NOT a separate global stop.
        #   * only after the barrier does a small GLOBAL tail run (retire/replicate/admit
        #     collectives): tail = pool_round_ms - denoise_span, on all used gpus.
        # Round wall width is therefore exactly pool_round_ms. We deliberately do NOT add
        # rank0's pool_barrier_ms on top of denoise_span: rank0's wait already lives inside
        # denoise_span (it waited while the straggler computed), so adding it double-counted
        # and produced a phantom full-width "all-GPU idle" band after the barrier.
        denoise_ms = max(0.0, float(rec.get("denoise_span_ms", rec.get("rank0_denoise_ms", 0.0))))
        round_ms = max(float(rec.get("pool_round_ms", denoise_ms)), denoise_ms)
        tail_ms = max(0.0, round_ms - denoise_ms)
        tail_total_ms += tail_ms

        x0 = t / 1000.0
        used = {int(g) for lane in lanes for g in lane.get("gpus", [])}
        has_real_elapsed = any("local_compute_ms" in lane for lane in lanes)
        works = [
            max(0.0, float(lane.get("local_compute_ms", lane.get("pred_work_ms", 0.0))))
            for lane in lanes
        ]
        max_work = max(works) if works else 0.0
        strag_idx = works.index(max_work) if max_work > 0.0 else -1

        for idx, lane in enumerate(lanes):
            rid = str(lane.get("request_id") or (lane.get("batch_ids") or ["unknown"])[0])
            gpus = [int(g) for g in lane.get("gpus", [])]
            mode = str(lane.get("mode") or f"w{lane.get('width', '?')}")
            work = max(0.0, float(lane.get("local_compute_ms", lane.get("pred_work_ms", 0.0))))
            if has_real_elapsed:
                compute_ms = work
            else:
                compute_ms = denoise_ms if max_work <= 0 else denoise_ms * min(1.0, work / max_work)
            # Real idle bubble on THIS lane's gpus only: from its finish until the barrier
            # releases (denoise_span). This is the load-imbalance wait the user accepts.
            idle_ms = max(0.0, denoise_ms - compute_ms)
            is_straggler = idx == strag_idx and len(lanes) > 1
            edge = "#cc0000" if is_straggler else "black"
            lw = 1.2 if is_straggler else 0.3

            for g in gpus:
                ax.barh(
                    g,
                    compute_ms / 1000.0,
                    left=x0,
                    height=0.82,
                    color=colors.get(rid, "tab:gray"),
                    alpha=0.95,
                    edgecolor=edge,
                    linewidth=lw,
                    zorder=2,
                )
                if idle_ms > 1e-6:
                    ax.barh(
                        g,
                        idle_ms / 1000.0,
                        left=x0 + compute_ms / 1000.0,
                        height=0.82,
                        color=colors.get(rid, "tab:gray"),
                        alpha=0.28,
                        hatch="xx",
                        edgecolor=edge,
                        linewidth=0.35,
                        zorder=2,
                    )

            if gpus and compute_ms / 1000.0 > 0.8:
                label = f"{rid[-2:]}\n{mode}"
                if is_straggler and rec.get("round_index") not in straggler_labels:
                    label += "\nSTR"
                    straggler_labels.add(rec.get("round_index"))
                ax.text(
                    x0 + max(0.2, compute_ms / 2000.0),
                    min(gpus),
                    label,
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="white",
                    zorder=4,
                )

        # Post-barrier GLOBAL tail (retire/replicate/admit collectives): genuinely all used
        # gpus, drawn once the slowest lane has arrived at the barrier.
        if tail_ms > 1e-6:
            for g in sorted(used):
                ax.barh(
                    g,
                    tail_ms / 1000.0,
                    left=x0 + denoise_ms / 1000.0,
                    height=0.82,
                    color="#ffb3b3",
                    alpha=0.65,
                    hatch="..",
                    edgecolor="#cc3333",
                    linewidth=0.25,
                    zorder=3,
                )

        for g in range(world_size):
            if g not in used:
                ax.barh(
                    g,
                    round_ms / 1000.0,
                    left=x0,
                    height=0.82,
                    color="#e8e8e8",
                    hatch="///",
                    edgecolor="#c0c0c0",
                    linewidth=0.3,
                    zorder=1,
                )

        # barrier-release marker: the slowest lane arrives here and everyone proceeds.
        ax.axvline((t + denoise_ms) / 1000.0, ymin=0.02, ymax=0.90, color="red", linewidth=0.7, alpha=0.65, zorder=5)
        t += round_ms
        prev_active_ids = current_ids

    ystrip = world_size + 0.4
    for rid in rids:
        arr = arrivals[rid] / 1000.0
        ax.plot(arr, ystrip, marker="v", color=colors[rid], markersize=7, markeredgecolor="black", markeredgewidth=0.4, zorder=6)
        es = exec_start.get(rid)
        if es is not None and es / 1000.0 > arr:
            ax.plot([arr, es / 1000.0], [ystrip, ystrip], color=colors[rid], linewidth=1.6, alpha=0.6, zorder=4)

    summary = d["serve_summary"]
    timers = d["timers"]
    denoise_total = float(timers.get("pool_denoise_ms", {}).get("total_ms", 0.0)) / 1000.0
    ax.set_title(
        f"{LABELS.get(tag, tag)}   makespan={summary['makespan_ms'] / 1000.0:.1f}s"
        f"   denoise(max local elapsed)={denoise_total:.1f}s   post-barrier global={tail_total_ms / 1000.0:.1f}s",
        fontsize=10.5,
        loc="left",
    )
    ax.set_ylim(-0.6, world_size + 0.9)
    ax.set_yticks(list(range(world_size)) + [ystrip])
    ax.set_yticklabels([f"GPU{g}" for g in range(world_size)] + ["arrivals"], fontsize=7)
    ax.grid(axis="x", alpha=0.25)
    return max(end_ms, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--arms", default="dp8,cp8,elastic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--rate", default="0.36")
    args = ap.parse_args()

    root = Path(args.root)
    arms = args.arms.split(",")
    trace = json.loads(Path(args.trace).read_text())
    arrivals = {r["request_id"]: float(r["arrival_ms"]) for r in trace["requests"]}
    rids = [r["request_id"] for r in trace["requests"]]
    colors = lane_color_map(rids)

    panels = {}
    tmax = 0.0
    for tag in arms:
        mp = find_serve_metrics(root, tag)
        if mp is None:
            raise FileNotFoundError(f"missing serve_metrics for arm {tag} under {root}")
        serve = json.loads(mp.read_text())
        summary = attach_offline_lane_elapsed(load_summary(mp), mp)
        per = serve["per_request"]
        panels[tag] = {
            "serve_summary": serve["summary"],
            "records": summary.get("records", {}).get("pool_round", []),
            "timers": summary.get("timers", {}),
            # ``pool_round`` records and trace arrivals are both relative to the timed
            # serve run (after warmup + Timer.reset). Do not add the log-derived
            # ``first_layout_ms`` offset here: it is fitted from rank-0 log timestamps with
            # only second-level wall-clock precision, and mixing that clock with measured
            # per-round durations visibly misaligns arrival markers and compute bars.
            "first_layout_ms": 0.0,
            "exec_start": {rid: rec.get("first_sched_ms") for rid, rec in per.items()},
            "end_ms": max(rec["finish_ms"] for rec in per.values() if rec.get("finish_ms") is not None),
        }
        tmax = max(tmax, panels[tag]["end_ms"])

    fig, axes = plt.subplots(len(arms), 1, figsize=(16, 3.4 * len(arms)), sharex=True)
    if len(arms) == 1:
        axes = [axes]
    for ax, tag in zip(axes, arms):
        tmax = max(tmax, draw_panel(ax, tag, panels[tag], arrivals, colors, args.world_size))

    axes[-1].set_xlim(0, tmax / 1000.0 * 1.02)
    axes[-1].set_xlabel("time since serve start (s)")
    legend = [
        Patch(facecolor="gray", alpha=0.95, edgecolor="black", label="real local lane elapsed (compute)"),
        Patch(facecolor="gray", alpha=0.28, hatch="xx", edgecolor="black", label="real idle: lane finished early, waits for barrier (overlaps straggler)"),
        Patch(facecolor="none", edgecolor="#cc0000", linewidth=1.2, label="round straggler lane (red outline / STR)"),
        Patch(facecolor="#ffb3b3", alpha=0.65, hatch="..", edgecolor="#cc3333", label="post-barrier global (retire/replicate/admit)"),
        Patch(facecolor="#e8e8e8", hatch="///", edgecolor="#c0c0c0", label="idle GPU (unassigned)"),
        plt.Line2D([], [], color="k", marker="v", linestyle="none", markersize=7, label="arrival"),
        plt.Line2D([], [], color="k", linewidth=1.6, alpha=0.6, label="waiting (arrival->exec)"),
        plt.Line2D([], [], color="red", linewidth=0.9, label="barrier release (slowest lane arrives)"),
    ]
    axes[0].legend(handles=legend, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.12))
    fig.suptitle(
        f"Straggler-aware GPU timeline @ lambda={args.rate} req/s (8x H20, 50 steps, mixed 1024^2/512^2)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
