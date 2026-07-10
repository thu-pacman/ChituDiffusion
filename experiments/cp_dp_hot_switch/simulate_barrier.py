#!/usr/bin/env python3
"""Barrier-aware replay: quantify the synchronous-pool spin tax and how much a
latency-balanced chunking (variable per-lane K) would recover.

Why a *new* simulator: the existing ``simulate_pool`` advances every lane
independently -- it has NO global barrier -- which is exactly why it predicted
elastic would win while the real engine (a ``dist.barrier()`` every ``phase_k``
steps, see generator._pool_round) lost. To validate the fix before touching the
runtime we replay the REAL layouts the planner chose (from the run log) with the
REAL measured per-(mode,shape) step times, reconstruct the observed barrier tax
(sanity-check against the measured ``pool_barrier_ms``), then re-cost the same
work under a latency-balanced barrier that lets fast lanes run more steps.

Key finding this quantifies: at the dominant shape (1024^2) the elastic planner's
cost model is already accurate (corr ~0.9), so the spin is NOT a prediction error
-- it is STRUCTURAL. The planner knows sp1=1052ms and cfp2xcp2=322ms (3.3x apart)
yet the engine forces both to run the same phase_k=5 steps and then barrier, so the
fast width-4 lane idles ~3.6s every phase. Latency-balanced K (fast lane runs ~16
steps while the straggler runs 5, so both hit the barrier together) removes it --
and needs exactly the accurate/calibrated per-lane step time to size each K_i.

Run:
    python simulate_barrier.py \
      --run outputs/bench_rate_sweep_h20_8gpu/r0p36/slo_elastic-20260708_223009-req0000 \
      --log outputs/bench_rate_sweep_h20_8gpu/logs/r0p36_slo_elastic.log \
      --trace experiments/cp_dp_hot_switch/traces/serve/sweep_tight/poisson_r0p36.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chitu_diffusion.runtime.cost_model import CostModel, image_token_count


# --------------------------------------------------------------------------- #
# execution step-time model: measured anchors where we have clean data, roofline
# (guidance=2, as the elastic planner priced it) x measured correction elsewhere.
# --------------------------------------------------------------------------- #
# mode -> (sp_degree, cfg_parallel). "sp1" is a width-1 lane: 1 GPU runs both CFG
# conditions serially (no CFG parallelism possible on one card).
MODE_SPEC = {
    "sp1":      (1, 1),
    "cfp2xcp1": (1, 2),
    "cfp2xcp2": (2, 2),
    "cfp2xcp4": (4, 2),
    "cfp2xcp8": (8, 2),
}
MODE_WIDTH = {"sp1": 1, "cfp2xcp1": 2, "cfp2xcp2": 4, "cfp2xcp4": 8, "cfp2xcp8": 16}

# Clean measured anchors (ms/step) from this study (rank0_denoise/phase_k + baselines).
MEASURED = {
    ("sp1", 1024): 1052.0,
    ("sp1", 512): 247.0,
    ("cfp2xcp2", 1024): 305.0,   # rank0 lane in the mixed rounds (gpu0 lane, width-4)
    ("cfp2xcp4", 1024): 174.0,   # single width-8 lane, rounds 0-19
}


@dataclass
class Lane:
    req: str
    mode: str
    gpus: list[int]
    shape: int


def _exec_step(model: CostModel, mode: str, shape: int) -> float:
    if (mode, shape) in MEASURED:
        return MEASURED[(mode, shape)]
    sp, cfgp = MODE_SPEC[mode]
    tok = image_token_count(shape, shape)
    roof = model.predict_step_ms(img_tokens=tok, sp_degree=sp, cfg_conditions=2, cfg_parallel=cfgp)
    # apply the geomean 1024 correction (~0.92) as a mild, honest derate for gaps.
    return roof * 0.92


def _parse_layouts(log_path: str) -> list[list[tuple[str, str, list[int]]]]:
    """Return, per layout-change log line, the list of (req, mode, gpus)."""
    epochs: list[list[tuple[str, str, list[int]]]] = []
    lane_re = re.compile(r"(req\d+):([a-z0-9]+)@\[([0-9,\s]*)\]")
    for line in open(log_path):
        if "pool] layout:" not in line:
            continue
        body = line.split("layout:", 1)[1]
        lanes = []
        for m in lane_re.finditer(body):
            req, mode, gpus = m.group(1), m.group(2), m.group(3)
            gset = [int(x) for x in gpus.split(",") if x.strip() != ""]
            lanes.append((req, mode, gset))
        if lanes:
            epochs.append(lanes)
    return epochs


def _load_rounds(run_dir: str) -> tuple[list[dict], dict]:
    summ = glob.glob(os.path.join(run_dir, "metrics", "timing", "summary.json"))
    d = json.load(open(summ[0]))
    return d["records"]["pool_round"], d["timers"]


def _shapes(trace_path: str) -> dict[str, int]:
    d = json.load(open(trace_path))
    return {r["request_id"]: int(r["width"]) for r in d["requests"]}


def _align(rounds: list[dict], epochs: list[list]) -> list[int]:
    """Map each round -> epoch index. Advance the epoch pointer when the round's
    sorted widths change from the previous round OR a new admit happened."""
    mapping = []
    e = 0
    prev_sig = None
    for i, r in enumerate(rounds):
        sig = tuple(sorted(r["layout_widths"]))
        if i > 0 and (sig != prev_sig or int(r.get("n_admit", 0)) > 0):
            e = min(e + 1, len(epochs) - 1)
        mapping.append(e)
        prev_sig = sig
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir (…-req0000)")
    ap.add_argument("--log", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--cost-model", default="experiments/cp_dp_hot_switch/cost_models/h20.json")
    ap.add_argument("--total-gpus", type=int, default=8)
    ap.add_argument("--out", default="experiments/cp_dp_hot_switch/calibrate_offline")
    args = ap.parse_args()

    model = CostModel.load(args.cost_model)
    epochs = _parse_layouts(args.log)
    rounds, timers = _load_rounds(args.run)
    shapes = _shapes(args.trace)
    mapping = _align(rounds, epochs)

    G = args.total_gpus
    # accumulators
    sum_barrier = 0.0           # Σ max-lane phase  (wall time in denoise+barrier)
    sum_rank0_compute = 0.0     # Σ rank0 lane phase  (~ measured pool_denoise)
    sum_rank0_spin = 0.0        # Σ (barrier - rank0 phase)  (~ measured pool_barrier)
    sum_total_spin_gpu = 0.0    # Σ_lanes width*(barrier - phase)  (recoverable GPU-sec)
    gpu_busy = [0.0] * G        # per-GPU useful compute (no spin)
    gpu_spin = [0.0] * G
    recon_rows = []             # (round, my rank0 phase, measured rank0_denoise)

    for r, eidx in zip(rounds, mapping):
        k = max(1, int(r["phase_k"]))
        lanes_raw = epochs[eidx]
        lanes = [Lane(req, mode, gpus, shapes.get(req, 1024)) for (req, mode, gpus) in lanes_raw]
        # lane phase = k * exec_step
        phase = {id(ln): k * _exec_step(model, ln.mode, ln.shape) for ln in lanes}
        barrier = max(phase.values()) if phase else 0.0
        # rank0 lane = the lane owning gpu 0
        rank0_lane = next((ln for ln in lanes if 0 in ln.gpus), None)
        if rank0_lane is None:  # gpu0 idle this round (tail rounds) -> use fastest lane
            rank0_lane = min(lanes, key=lambda l: phase[id(l)]) if lanes else None
        r0_phase = phase[id(rank0_lane)] if rank0_lane else 0.0

        sum_barrier += barrier
        sum_rank0_compute += r0_phase
        sum_rank0_spin += (barrier - r0_phase)
        for ln in lanes:
            w = len(ln.gpus)
            sum_total_spin_gpu += w * (barrier - phase[id(ln)])
            for g in ln.gpus:
                if 0 <= g < G:
                    gpu_busy[g] += phase[id(ln)]
                    gpu_spin[g] += (barrier - phase[id(ln)])
        recon_rows.append((r["round_index"], r0_phase, float(r["rank0_denoise_ms"])))

    # ---- validation vs measured timers ---- #
    meas_den = timers.get("pool_denoise_ms", {}).get("total_ms", 0.0)
    meas_bar = timers.get("pool_barrier_ms", {}).get("total_ms", 0.0)

    # ---- balanced-K projection ---- #
    # Under a latency-balanced barrier every GPU does useful work through the whole
    # epoch instead of spinning; makespan floor = busiest GPU's useful compute +
    # the unavoidable inter-lane sync (kept as the measured broadcast/retire tax).
    busiest_gpu = max(gpu_busy)
    total_useful_gpu = sum(gpu_busy)
    throughput_floor = total_useful_gpu / G

    print("=" * 74)
    print("BARRIER-AWARE REPLAY  (real layouts x measured step times)")
    print(f"  run: {os.path.basename(args.run)}   epochs={len(epochs)} rounds={len(rounds)}")
    print("=" * 74)
    print("\n[VALIDATION] reconstructed vs measured pool timers")
    print(f"    rank0 compute (Σ lane phase) : {sum_rank0_compute/1000:7.1f}s   "
          f"measured pool_denoise = {meas_den/1000:6.1f}s")
    print(f"    rank0 spin    (Σ barrier gap): {sum_rank0_spin/1000:7.1f}s   "
          f"measured pool_barrier = {meas_bar/1000:6.1f}s")
    err = abs(sum_rank0_spin - meas_bar) / meas_bar * 100 if meas_bar else 0.0
    print(f"    => spin reconstruction error = {err:.0f}%  "
          f"({'MODEL VALIDATED' if err < 25 else 'check model'})")

    print("\n[TAX] where the GPU-seconds go (denoise phase only)")
    print(f"    useful compute      : {total_useful_gpu/1000:7.1f} GPU·s")
    print(f"    barrier spin (waste): {sum_total_spin_gpu/1000:7.1f} GPU·s  "
          f"({sum_total_spin_gpu/(total_useful_gpu+sum_total_spin_gpu)*100:.0f}% of GPU time)")

    print("\n[PROJECTION] latency-balanced K removes intra-epoch spin")
    print(f"    current denoise+barrier wall (Σ max-lane phase) : {sum_barrier/1000:7.1f}s")
    print(f"    busiest-GPU useful compute (makespan floor)     : {busiest_gpu/1000:7.1f}s")
    print(f"    perfect-balance throughput floor (useful/{G})     : {throughput_floor/1000:7.1f}s")
    reduction = (sum_barrier - busiest_gpu) / sum_barrier * 100 if sum_barrier else 0.0
    print(f"    => denoise-phase wall reduction if spin removed : {reduction:.0f}%")

    # per-GPU imbalance
    print("\n[IMBALANCE] per-GPU useful vs spin (s)")
    for g in range(G):
        bar = "#" * int(gpu_busy[g] / 1000)
        print(f"    gpu{g}: busy={gpu_busy[g]/1000:5.1f}  spin={gpu_spin[g]/1000:5.1f}  {bar}")

    _plot(args, recon_rows, gpu_busy, gpu_spin, sum_barrier, busiest_gpu,
          total_useful_gpu, sum_total_spin_gpu, meas_bar, sum_rank0_spin)


def _plot(args, recon_rows, gpu_busy, gpu_spin, sum_barrier, busiest_gpu,
          total_useful, total_spin, meas_bar, recon_spin) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (1) reconstruction: my rank0 phase vs measured rank0_denoise per round
    ax = axes[0]
    rounds = [r[0] for r in recon_rows]
    mine = [r[1] / 1000 for r in recon_rows]
    meas = [r[2] / 1000 for r in recon_rows]
    ax.plot(rounds, meas, "o-", color="#e6550d", label="measured rank0_denoise", ms=3)
    ax.plot(rounds, mine, "s--", color="#3182bd", label="replay (measured step×k)", ms=3)
    ax.set_xlabel("round"); ax.set_ylabel("rank0 phase (s)")
    ax.set_title("(1) replay reproduces per-round compute")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2) per-GPU busy vs spin
    ax = axes[1]
    G = len(gpu_busy)
    x = range(G)
    ax.bar(x, [b / 1000 for b in gpu_busy], color="#31a354", label="useful compute")
    ax.bar(x, [s / 1000 for s in gpu_spin], bottom=[b / 1000 for b in gpu_busy],
           color="#de2d26", alpha=0.8, label="barrier spin (waste)")
    ax.set_xlabel("GPU"); ax.set_ylabel("time (s)")
    ax.set_title("(2) per-GPU: uniform-K wastes idle spin")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # (3) makespan projection
    ax = axes[2]
    labels = ["current\n(uniform-K)", "balanced-K\n(busiest GPU)", "throughput\nfloor"]
    vals = [sum_barrier / 1000, busiest_gpu / 1000, total_useful / 1000 / G]
    colors = ["#e6550d", "#31a354", "#9ecae1"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.0f}s", (i, v), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("denoise-phase wall (s)")
    ax.set_title("(3) latency-balanced K projection")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "simulate_barrier.png")
    fig.savefig(p, dpi=130)
    print(f"\n    plot -> {p}")


if __name__ == "__main__":
    main()
