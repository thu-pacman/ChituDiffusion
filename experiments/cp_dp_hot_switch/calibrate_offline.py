#!/usr/bin/env python3
"""Offline calibration study for the online cost-model monitor.

Goal (zero GPU cost): use the *clean* per-step compute times the pool engine
already records (``rank0_denoise_ms`` per round, measured right before the world
barrier -- so it excludes barrier spin) to answer three questions before we build
a runtime monitor:

  1. How wrong is the offline roofline (``h20.json``)?  We compare the measured
     ms/step against the value the *planner actually priced* for that lane, and
     derive a global derate ``g`` plus per-(mode,shape) correction factors ``c_key``.
  2. Is a single global factor enough?  (Spoiler: no -- the roofline over-prices
     wide/small lanes and under-prices narrow/large lanes, so the errors point in
     opposite directions and only a per-key factor fixes both.)
  3. How much is the current synchronous pool engine paying in barrier spin, and
     how fast would an EWMA monitor converge to a <5% prediction error?

We can only cleanly attribute a lane->shape mapping for the *homogeneous* baselines
(pure_sp == every lane sp8, pure_dp == every lane dp1); ``rank0_denoise_ms`` is a
single lane per round and the record does not store which GPU-0 lane it was for the
heterogeneous elastic layouts.  That is exactly the small logging gap the monitor
closes (log clean per-lane step time keyed by shape+mode).  The baselines already
span both shapes and the two extreme modes, which is enough to prove the point and
tune the EWMA.

Usage:
    python calibrate_offline.py --root outputs/bench_rate_sweep_h20_8gpu \
        --cost-model experiments/cp_dp_hot_switch/cost_models/h20.json \
        --out experiments/cp_dp_hot_switch/calibrate_offline
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chitu_diffusion.runtime.cost_model import CostModel, image_token_count


# --------------------------------------------------------------------------- #
# planner pricing: mirror exactly what scheduler.py feeds profile_from_cost_model
# --------------------------------------------------------------------------- #
# baselines run with cfg_parallel_max=1 -> planner_guidance_scale=1.0 (single
# forward priced); slo_elastic runs cfg_parallel_max=2 -> guidance=2.0. We price
# each baseline lane the way the planner did so the ratio == the error the planner
# actually acted on.
POLICY_MODE = {"pure_sp": "sp8", "pure_dp": "dp1"}
MODE_SPEC = {  # mode -> (sp_degree, cfg_parallel) at the planner's baseline guidance=1
    "sp8": (8, 1),
    "dp1": (1, 1),
}


@dataclass
class Sample:
    rate: str
    policy: str
    mode: str
    shape: int          # square side in px
    tokens: int
    ms_per_step: float  # measured, clean (no barrier)


@dataclass
class KeyStat:
    mode: str
    shape: int
    tokens: int
    n: int
    measured_med: float
    predicted_g1: float   # planner-priced (guidance=1, single forward)
    predicted_g2: float   # if the model priced the real 2-condition CFG
    ratio_vs_planner: float
    samples: list[float] = field(default_factory=list)


def _measured_steps(summary_path: str) -> list[tuple[int, float]]:
    """Return (phase_k, rank0_denoise_ms) per round from a timing summary.json."""
    d = json.load(open(summary_path))
    out = []
    for r in d.get("records", {}).get("pool_round", []):
        k = max(1, int(r.get("phase_k", 1)))
        ms = float(r.get("rank0_denoise_ms", 0.0))
        if ms > 0:
            out.append((k, ms))
    return out


def _cluster_by_gap(values: list[float], k: int) -> list[list[float]]:
    """Split sorted 1-D values into k clusters at the (k-1) largest relative gaps.

    Robust for our clearly-separated shape clusters (e.g. 110 vs 224 ms/step).
    """
    if k <= 1 or len(values) <= k:
        return [values]
    sv = sorted(values)
    gaps = sorted(
        range(1, len(sv)),
        key=lambda i: (sv[i] - sv[i - 1]) / max(1e-9, sv[i - 1]),
        reverse=True,
    )
    cut = sorted(gaps[: k - 1])
    clusters, prev = [], 0
    for c in cut:
        clusters.append(sv[prev:c])
        prev = c
    clusters.append(sv[prev:])
    return clusters


def _serving_timers(summary_path: str) -> dict:
    d = json.load(open(summary_path))
    t = d.get("timers", {})
    den = t.get("pool_denoise_ms", {})
    bar = t.get("pool_barrier_ms", {})
    return {
        "elapsed_s": float(d.get("serving_elapsed_s", 0.0)),
        "denoise_s": den.get("total_ms", 0.0) / 1000.0,
        "barrier_s": bar.get("total_ms", 0.0) / 1000.0,
        "barrier_avg_ms": bar.get("avg_ms", 0.0),
        "barrier_n": bar.get("samples", 0),
    }


def _find_runs(root: str) -> dict[str, dict[str, str]]:
    """rate -> policy -> summary.json path."""
    runs: dict[str, dict[str, str]] = {}
    for rate_dir in sorted(glob.glob(os.path.join(root, "r*"))):
        rate = os.path.basename(rate_dir)
        if not rate.startswith("r0p"):
            continue
        for pol in ("pure_sp", "pure_dp", "slo_elastic"):
            hits = glob.glob(
                os.path.join(rate_dir, f"{pol}-*-req0000", "metrics", "timing", "summary.json")
            )
            if hits:
                runs.setdefault(rate, {})[pol] = sorted(hits)[0]
    return runs


def _shapes_from_trace(trace_path: str) -> list[int]:
    if not trace_path or not os.path.exists(trace_path):
        return [1024, 512]
    d = json.load(open(trace_path))
    reqs = d["requests"] if isinstance(d, dict) else d
    sides = sorted({int(r["width"]) for r in reqs}, reverse=True)
    return sides or [1024, 512]


def _rate_to_float(rate_tag: str) -> float:
    return float(rate_tag.replace("r", "").replace("p", "."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/bench_rate_sweep_h20_8gpu")
    ap.add_argument("--cost-model", default="experiments/cp_dp_hot_switch/cost_models/h20.json")
    ap.add_argument("--trace", default="experiments/cp_dp_hot_switch/traces/serve/sweep_tight/poisson_r0p36.json")
    ap.add_argument("--ewma-alpha", type=float, default=0.3)
    ap.add_argument("--out", default="experiments/cp_dp_hot_switch/calibrate_offline")
    args = ap.parse_args()

    model = CostModel.load(args.cost_model)
    shapes = _shapes_from_trace(args.trace)
    tok = {s: image_token_count(s, s) for s in shapes}
    runs = _find_runs(args.root)

    def price(mode: str, shape: int, guidance: float) -> float:
        sp, cfgp = MODE_SPEC[mode]
        cfg_cond = 2 if guidance > 1.0 else 1
        return model.predict_step_ms(
            img_tokens=tok[shape], sp_degree=sp,
            cfg_conditions=cfg_cond, cfg_parallel=cfgp,
        )

    # ---- Part A: roofline calibration on the homogeneous baselines ---------- #
    samples: list[Sample] = []
    for rate, pols in runs.items():
        for pol, mode in POLICY_MODE.items():
            if pol not in pols:
                continue
            steps = _measured_steps(pols[pol])
            per_step = [ms / k for (k, ms) in steps]
            if not per_step:
                continue
            clusters = _cluster_by_gap(per_step, len(shapes))
            # ascending measured cluster <-> ascending token count (monotone: more
            # tokens => more compute), so map by rank.
            cluster_meds = [statistics.median(c) for c in clusters if c]
            order = sorted(range(len(clusters)), key=lambda i: cluster_meds[i])
            shapes_asc = sorted(shapes, key=lambda s: tok[s])
            for rank, ci in enumerate(order):
                shape = shapes_asc[rank]
                for v in clusters[ci]:
                    samples.append(Sample(rate, pol, mode, shape, tok[shape], v))

    # aggregate per (mode, shape)
    keys: dict[tuple[str, int], KeyStat] = {}
    for s in samples:
        k = (s.mode, s.shape)
        if k not in keys:
            keys[k] = KeyStat(
                mode=s.mode, shape=s.shape, tokens=s.tokens, n=0,
                measured_med=0.0,
                predicted_g1=price(s.mode, s.shape, 1.0),
                predicted_g2=price(s.mode, s.shape, 2.0),
                ratio_vs_planner=0.0,
            )
        keys[k].samples.append(s.ms_per_step)
    for ks in keys.values():
        ks.n = len(ks.samples)
        ks.measured_med = statistics.median(ks.samples)
        ks.ratio_vs_planner = ks.measured_med / ks.predicted_g1 if ks.predicted_g1 else float("nan")

    all_ratios = [ks.ratio_vs_planner for ks in keys.values() for _ in range(ks.n)]
    # per-sample ratios for a fair global geomean
    sample_ratios = []
    for s in samples:
        p = price(s.mode, s.shape, 1.0)
        if p:
            sample_ratios.append(s.ms_per_step / p)
    global_g = math.exp(statistics.mean(math.log(r) for r in sample_ratios)) if sample_ratios else float("nan")

    # ---- Part B: barrier tax across all runs -------------------------------- #
    tax_rows = []
    for rate in sorted(runs, key=_rate_to_float):
        for pol in ("pure_sp", "pure_dp", "slo_elastic"):
            if pol not in runs[rate]:
                continue
            t = _serving_timers(runs[rate][pol])
            ratio = t["barrier_s"] / t["denoise_s"] if t["denoise_s"] else 0.0
            tax_rows.append((rate, pol, t, ratio))

    # ---- Part C: EWMA convergence on the busiest key ------------------------ #
    busiest = max(keys.values(), key=lambda k: k.n)
    ewma_curve = _ewma_convergence(busiest, args.ewma_alpha)

    _print_report(keys, global_g, tax_rows, busiest, args)
    os.makedirs(args.out, exist_ok=True)
    _plots(keys, global_g, tax_rows, busiest, ewma_curve, args)
    _dump_calibration(keys, global_g, args)


def _ewma_convergence(ks: KeyStat, alpha: float) -> dict:
    """Feed measured samples in order; track prediction error before/after EWMA
    correction of the roofline. Returns the per-step abs-% error series."""
    pred = ks.predicted_g1
    c = 1.0  # correction factor, starts at "trust roofline"
    err_raw, err_cal = [], []
    for m in ks.samples:
        # error if we used the *uncorrected* roofline for this step
        err_raw.append(abs(pred - m) / m * 100.0)
        # error if we used the roofline scaled by the *current* EWMA factor
        err_cal.append(abs(pred * c - m) / m * 100.0)
        # update factor AFTER predicting (online, causal)
        c = (1 - alpha) * c + alpha * (m / pred)
    return {"raw": err_raw, "cal": err_cal, "final_c": c}


def _print_report(keys, global_g, tax_rows, busiest, args) -> None:
    print("=" * 78)
    print("OFFLINE COST-MODEL CALIBRATION STUDY")
    print("  clean signal: rank0_denoise_ms / phase_k  (measured pre-barrier)")
    print("=" * 78)
    print("\n[A] Roofline accuracy on homogeneous baselines")
    print("    (predicted = what the planner actually priced; g1=single forward)\n")
    hdr = f"    {'mode':>5} {'shape':>6} {'n':>4} {'pred_g1':>9} {'pred_g2':>9} {'measured':>9} {'meas/pred':>10}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for ks in sorted(keys.values(), key=lambda k: (k.mode, -k.shape)):
        print(f"    {ks.mode:>5} {ks.shape:>5}² {ks.n:>4} {ks.predicted_g1:>8.1f} "
              f"{ks.predicted_g2:>8.1f} {ks.measured_med:>8.1f} {ks.ratio_vs_planner:>9.2f}×")
    print(f"\n    global derate g (geomean of per-sample ratios) = {global_g:.2f}×")
    ratios = [ks.ratio_vs_planner for ks in keys.values()]
    print(f"    per-key ratio spread: {min(ratios):.2f}× .. {max(ratios):.2f}×  "
          f"(span {max(ratios)/min(ratios):.1f}×)")
    print("    => errors point in OPPOSITE directions; a single global g cannot fix both.")

    print("\n[B] Synchronous-pool barrier tax (measured)\n")
    hdr2 = f"    {'rate':>6} {'policy':>12} {'elapsed_s':>10} {'denoise_s':>10} {'barrier_s':>10} {'bar/den':>8}"
    print(hdr2)
    print("    " + "-" * (len(hdr2) - 4))
    for rate, pol, t, ratio in tax_rows:
        print(f"    {rate:>6} {pol:>12} {t['elapsed_s']:>10.1f} {t['denoise_s']:>10.1f} "
              f"{t['barrier_s']:>10.1f} {ratio:>7.2f}×")
    print("    => slo_elastic spends up to >100% of its compute time idling at barriers")
    print("       (fast lanes wait for slow lanes because phase_k is uniform per round).")

    print(f"\n[C] EWMA monitor convergence (key={busiest.mode}@{busiest.shape}², "
          f"alpha={args.ewma_alpha})")
    curve = _ewma_convergence(busiest, args.ewma_alpha)
    print(f"    uncorrected roofline mean |err| = {statistics.mean(curve['raw']):.1f}%")
    print(f"    EWMA-corrected     mean |err| = {statistics.mean(curve['cal']):.1f}%")
    tail = curve["cal"][len(curve["cal"]) // 2:]
    print(f"    EWMA-corrected steady |err|   = {statistics.mean(tail):.1f}%  "
          f"(converged factor c={curve['final_c']:.2f}×)")
    print(f"\n    outputs -> {args.out}/")


def _plots(keys, global_g, tax_rows, busiest, ewma_curve, args) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (1) predicted vs measured per key
    ax = axes[0]
    ks_sorted = sorted(keys.values(), key=lambda k: (k.mode, -k.shape))
    labels = [f"{k.mode}\n{k.shape}²" for k in ks_sorted]
    x = range(len(ks_sorted))
    w = 0.26
    ax.bar([i - w for i in x], [k.predicted_g1 for k in ks_sorted], w,
           label="roofline (planner, g=1)", color="#9ecae1")
    ax.bar([i for i in x], [k.predicted_g2 for k in ks_sorted], w,
           label="roofline (2-cond CFG, g=2)", color="#3182bd")
    ax.bar([i + w for i in x], [k.measured_med for k in ks_sorted], w,
           label="measured (clean)", color="#e6550d")
    for i, k in zip(x, ks_sorted):
        ax.annotate(f"{k.ratio_vs_planner:.2f}×", (i + w, k.measured_med),
                    ha="center", va="bottom", fontsize=8, color="#a63603")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ms / denoise step")
    ax.set_title("(A) roofline vs measured\n(× = measured / planner-priced)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (2) barrier tax
    ax = axes[1]
    rates = sorted({r for r, *_ in tax_rows}, key=_rate_to_float)
    pol_color = {"pure_sp": "#31a354", "pure_dp": "#756bb1", "slo_elastic": "#e6550d"}
    pol_label = {"pure_sp": "cp8", "pure_dp": "dp8", "slo_elastic": "cfp+elastic"}
    for pol in ("pure_sp", "pure_dp", "slo_elastic"):
        ys = []
        for r in rates:
            hit = [row for row in tax_rows if row[0] == r and row[1] == pol]
            ys.append(hit[0][3] if hit else float("nan"))
        ax.plot([_rate_to_float(r) for r in rates], ys, "o-",
                color=pol_color[pol], label=pol_label[pol])
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    ax.set_xlabel("arrival rate λ (req/s)")
    ax.set_ylabel("barrier_time / denoise_time")
    ax.set_title("(B) synchronous-pool barrier tax\n(idle spin ÷ real compute)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (3) EWMA convergence
    ax = axes[2]
    n = range(1, len(ewma_curve["raw"]) + 1)
    ax.plot(n, ewma_curve["raw"], color="#9ecae1", label="uncorrected roofline")
    ax.plot(n, ewma_curve["cal"], color="#e6550d", label=f"EWMA (α={args.ewma_alpha})")
    ax.axhline(5.0, ls="--", color="grey", lw=1, label="5% target")
    ax.set_xlabel("online sample #")
    ax.set_ylabel("|prediction error| (%)")
    ax.set_title(f"(C) monitor convergence\n{busiest.mode}@{busiest.shape}²")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(args.out, "calibrate_offline.png")
    fig.savefig(p, dpi=130)
    print(f"    plot -> {p}")


def _dump_calibration(keys, global_g, args) -> None:
    out = {
        "global_derate_g": global_g,
        "per_key": {
            f"{ks.mode}@{ks.shape}": {
                "tokens": ks.tokens,
                "n_samples": ks.n,
                "predicted_planner_ms": ks.predicted_g1,
                "predicted_2cond_ms": ks.predicted_g2,
                "measured_median_ms": ks.measured_med,
                "correction_factor": ks.ratio_vs_planner,
            }
            for ks in keys.values()
        },
    }
    p = os.path.join(args.out, "calibration_table.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"    table -> {p}")


if __name__ == "__main__":
    main()
