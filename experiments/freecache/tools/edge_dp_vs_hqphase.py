#!/usr/bin/env python3
"""Diagnose which edge-DP cost function explains the verified hqphase groundtruth.

Context (see experiments/freecache/WORKLOG.md):

hqphase (Qwen-Image B25 forced schedule) is now the project groundtruth: it has
been verified across 3 seeds AND 3 prompts (coffee_sign / landscape / color,
2026-06-24). The open planner question is no longer "is hqphase good" but "which
automatic edge-DP cost function reproduces hqphase".

This script treats hqphase as the optimization target / label and, for every
cost_key x gamma, asks:

  - what is hqphase's OWN path cost under that (cost_key, gamma)?
  - what is the DP-optimal path cost at the same budget (B=len(hqphase)),
    both unconstrained (max_gap=8) and gap-4-constrained (hqphase's own max gap)?
  - ratio = hqphase_cost / optimal_cost. A ratio close to 1.0 means the cost
    function RANKS hqphase as (near-)optimal -> that cost "explains" the
    groundtruth and is the right planner objective.

It also reports the STRUCTURAL distance between each DP-optimal schedule and
hqphase: fresh-step Jaccard, warmup density (fresh in first 9 steps), gap
histogram, and max gap used.

Pure CPU. Consumes the saved full-compute steptrace vectors. No GPU, no FreeCache
replay (this is a diagnostic, not a quality claim -- any promising cost still has
to be replayed through FreeCache afterwards).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Make sibling scripts importable when invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from order_probe_steptrace import (  # type: ignore  # noqa: E402
    DEFAULT_QWEN_HQ_ANCHORS,
    load_payloads,
    select_trace,
)
from traceplanner_edge_dp import (  # type: ignore  # noqa: E402
    COST_KEYS,
    compute_edge_costs,
    edge_dp,
)

# hqphase = the verified groundtruth fresh schedule (== order_probe DEFAULT).
HQPHASE = sorted(set(DEFAULT_QWEN_HQ_ANCHORS))


def path_cost(fresh: list[int], costs: dict, cost_key: str, gamma: float) -> float:
    """Sum of peak-suppressed edge costs along the given fresh path.

    Returns inf if any edge is missing (gap exceeds the edge table's max_gap).
    """
    total = 0.0
    for t, s in zip(fresh, fresh[1:]):
        edge = costs.get((t, s))
        if edge is None:
            return float("inf")
        total += edge[cost_key] ** gamma
    return total


def gap_histogram(fresh: list[int]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for t, s in zip(fresh, fresh[1:]):
        hist[s - t] = hist.get(s - t, 0) + 1
    return dict(sorted(hist.items()))


def warmup_density(fresh: list[int], window: int = 9) -> int:
    return sum(1 for step in fresh if step < window)


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def schedule_struct(fresh: list[int]) -> dict:
    gaps = [s - t for t, s in zip(fresh, fresh[1:])]
    return {
        "n_fresh": len(fresh),
        "warmup9": warmup_density(fresh, 9),
        "max_gap": max(gaps) if gaps else 0,
        "gap_hist": gap_histogram(fresh),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether hqphase is near-optimal under each edge-DP cost.")
    parser.add_argument("result_root", type=Path, help="steptrace-vectors run root (holds flexcache_steptrace_vectors_*.pt)")
    parser.add_argument("--gammas", default="1,2,4", help="comma-separated peak-suppression exponents")
    parser.add_argument("--max-gap", type=int, default=8, help="edge-table max gap (must >= hqphase max gap)")
    parser.add_argument("--gap-constrained", type=int, default=4, help="second DP run with this max gap (hqphase's own)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    trace = select_trace(load_payloads(args.result_root))
    n = int(len(trace["steps"]))
    steps = [int(s) for s in trace["steps"]]
    if steps != list(range(n)):
        raise SystemExit(f"Vector trace must be contiguous zero-based steps, got {steps[:5]}...{steps[-5:]}")
    budget = len(HQPHASE)
    hq_struct = schedule_struct(HQPHASE)
    print(f"trace mode={trace['mode']} steps={n} guidance_scale={trace['guidance_scale']}")
    print(f"hqphase: B={budget} warmup9={hq_struct['warmup9']} max_gap={hq_struct['max_gap']} gap_hist={hq_struct['gap_hist']}")
    if hq_struct["max_gap"] > args.max_gap:
        raise SystemExit(f"--max-gap={args.max_gap} < hqphase max gap {hq_struct['max_gap']}; hqphase edges would be missing.")

    out_dir = args.out_dir or (args.result_root / "edge_dp_vs_hqphase")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"computing edge costs (order=1 multi-step rollout, max_gap={args.max_gap})...")
    costs = compute_edge_costs(trace, args.max_gap)
    print(f"  {len(costs)} edges")

    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]

    rows = []
    for cost_key in COST_KEYS:
        for gamma in gammas:
            hq_cost = path_cost(HQPHASE, costs, cost_key, gamma)
            # DP optimum, unconstrained (full edge table max_gap).
            opt_free = edge_dp(costs, n, budget, gamma, cost_key, args.max_gap)
            # DP optimum, constrained to hqphase's own max gap.
            opt_gap = edge_dp(costs, n, budget, gamma, cost_key, args.gap_constrained)

            def summarize(res, tag):
                if res is None:
                    return {
                        f"{tag}_cost": float("inf"),
                        f"{tag}_ratio": float("inf"),
                        f"{tag}_jaccard": 0.0,
                        f"{tag}_warmup9": 0,
                        f"{tag}_max_gap": 0,
                        f"{tag}_fresh": "",
                    }
                fresh = res["fresh_steps"]
                st = schedule_struct(fresh)
                ratio = hq_cost / res["path_cost"] if res["path_cost"] > 0 else float("inf")
                return {
                    f"{tag}_cost": res["path_cost"],
                    f"{tag}_ratio": ratio,
                    f"{tag}_jaccard": jaccard(fresh, HQPHASE),
                    f"{tag}_warmup9": st["warmup9"],
                    f"{tag}_max_gap": st["max_gap"],
                    f"{tag}_fresh": " ".join(str(x) for x in fresh),
                }

            row = {
                "cost_key": cost_key,
                "gamma": gamma,
                "hq_path_cost": hq_cost,
                **summarize(opt_free, "free"),
                **summarize(opt_gap, f"gap{args.gap_constrained}"),
            }
            rows.append(row)
            print(
                f"  {cost_key:>26s} g{gamma:g} | "
                f"hq/opt(free)={row['free_ratio']:.3f} J={row['free_jaccard']:.2f} maxgap={row['free_max_gap']} | "
                f"hq/opt(gap{args.gap_constrained})={row[f'gap{args.gap_constrained}_ratio']:.3f} "
                f"J={row[f'gap{args.gap_constrained}_jaccard']:.2f}"
            )

    keys = list(rows[0].keys())
    with (out_dir / "edge_dp_vs_hqphase.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "edge_dp_vs_hqphase.json").write_text(
        json.dumps({"hqphase": HQPHASE, "hq_struct": hq_struct, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    # Ranking: which (cost_key, gamma) makes hqphase closest to the UNCONSTRAINED
    # optimum (ratio nearest 1 from above; lower is better explanation).
    finite = [r for r in rows if r["free_ratio"] != float("inf")]
    finite.sort(key=lambda r: r["free_ratio"])
    print("\n=== best cost functions that explain hqphase (unconstrained, ratio->1 is best) ===")
    for r in finite[:8]:
        print(
            f"  {r['cost_key']:>26s} g{r['gamma']:g}: ratio={r['free_ratio']:.3f} "
            f"jaccard={r['free_jaccard']:.2f} warmup9={r['free_warmup9']} max_gap={r['free_max_gap']}"
        )
    print(f"\nwrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
