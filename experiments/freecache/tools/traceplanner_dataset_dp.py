#!/usr/bin/env python3
"""Dataset-level edge-DP over Qwen StepTrace vectors.

This consumes a multi prompt/seed StepTrace vector dataset, computes per-sample
edge costs with the existing rollout metric, aggregates those costs across
samples, and compares the resulting schedules. It is intentionally CPU-only:
the output plans still need FreeCache replay before becoming quality claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from order_probe_steptrace import DEFAULT_QWEN_HQ_ANCHORS, select_trace  # type: ignore  # noqa: E402
from traceplanner_edge_dp import COST_KEYS, compute_edge_costs, edge_dp, fresh_to_policy, plot_schedule  # type: ignore  # noqa: E402


TASK_RE = re.compile(r"flexcache_steptrace_vectors_(?P<task>.+)_rank(?P<rank>\d+)\.pt$")
TASK_META_RE = re.compile(r"(?P<prompt>.+)_seed(?P<seed>\d+)$")
AUTOHQ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 21, 24, 28, 32, 36, 40, 44, 47, 48, 49]
HQPHASE = sorted(set(DEFAULT_QWEN_HQ_ANCHORS))


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def load_dataset(root: Path, rank: int) -> list[dict]:
    paths = sorted(root.rglob("flexcache_steptrace_vectors_*.pt"))
    samples = []
    for path in paths:
        match = TASK_RE.match(path.name)
        if not match or int(match.group("rank")) != rank:
            continue
        task = match.group("task")
        meta = TASK_META_RE.search(task)
        prompt = meta.group("prompt") if meta else "unknown"
        prompt = re.sub(r"^.*steptrace_vectors_jvp\d+_", "", prompt)
        payload = torch.load(path, map_location="cpu")
        trace = select_trace([payload])
        samples.append(
            {
                "task": task,
                "prompt": prompt,
                "seed": int(meta.group("seed")) if meta else -1,
                "rank": rank,
                "path": path,
                "trace": trace,
            }
        )
    if not samples:
        raise SystemExit(f"No rank{rank} vector files found under {root}")
    return samples


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    vals = sorted(values)
    pos = q * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def aggregate_values(values: list[float], mode: str) -> float:
    if mode == "mean":
        return float(mean(values))
    if mode == "median":
        return percentile(values, 0.5)
    if mode == "p75":
        return percentile(values, 0.75)
    if mode == "p90":
        return percentile(values, 0.9)
    if mode == "max":
        return max(values)
    if mode.startswith("mean_plus_"):
        lam = float(mode.removeprefix("mean_plus_"))
        mu = float(mean(values))
        var = float(mean([(v - mu) ** 2 for v in values]))
        return mu + lam * (var**0.5)
    raise ValueError(f"unknown aggregate mode: {mode}")


def aggregate_costs(sample_costs: list[dict], aggregate: str) -> dict:
    keys = sorted(sample_costs[0].keys())
    out = {}
    for edge in keys:
        out[edge] = {"n_reuse": sample_costs[0][edge]["n_reuse"]}
        for cost_key in COST_KEYS:
            values = [float(costs[edge][cost_key]) for costs in sample_costs]
            out[edge][cost_key] = aggregate_values(values, aggregate)
    return out


def gap_hist(fresh: list[int]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for t, s in zip(fresh, fresh[1:]):
        gap = s - t
        hist[gap] = hist.get(gap, 0) + 1
    return dict(sorted(hist.items()))


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def edge_path_values(fresh: list[int], sample_costs: list[dict], cost_key: str) -> list[float]:
    values = []
    for costs in sample_costs:
        total = 0.0
        for t, s in zip(fresh, fresh[1:]):
            total += float(costs[(t, s)][cost_key])
        values.append(total)
    return values


def schedule_stats(fresh: list[int], sample_costs: list[dict], cost_key: str) -> dict:
    gaps = [s - t for t, s in zip(fresh, fresh[1:])]
    sample_path = edge_path_values(fresh, sample_costs, cost_key)
    return {
        "fresh_steps": fresh,
        "n_fresh": len(fresh),
        "warmup9": sum(1 for step in fresh if step < 9),
        "max_gap": max(gaps) if gaps else 0,
        "gap_smoothness": sum((b - a) ** 2 for a, b in zip(gaps, gaps[1:])),
        "gap_hist": gap_hist(fresh),
        "jaccard_hqphase": jaccard(fresh, HQPHASE),
        "jaccard_autohq": jaccard(fresh, AUTOHQ),
        "sample_path_mean": float(mean(sample_path)),
        "sample_path_p90": percentile(sample_path, 0.9),
        "sample_path_max": max(sample_path),
        "sample_path_values": sample_path,
    }


def smooth_edge_dp(
    costs: dict,
    n: int,
    budget: int,
    gamma: float,
    cost_key: str,
    max_gap: int,
    smooth_lambda: float,
) -> dict | None:
    """Second-order DP that penalizes abrupt changes in neighboring gaps.

    State is (step, fresh_count, previous_gap). The first edge has no transition
    penalty. The transition penalty is normalized by max_gap^2 so lambda remains
    roughly comparable across gap constraints.
    """
    inf = float("inf")
    dp: list[list[dict[int, float]]] = [[{} for _ in range(budget + 1)] for _ in range(n)]
    parent: list[list[dict[int, tuple[int, int]]]] = [[{} for _ in range(budget + 1)] for _ in range(n)]
    dp[0][1][0] = 0.0
    norm = float(max(1, max_gap * max_gap))

    for s in range(1, n):
        for b in range(2, budget + 1):
            t_lo = max(0, s - max_gap)
            for t in range(t_lo, s):
                edge = costs.get((t, s))
                if edge is None:
                    continue
                gap = s - t
                edge_cost = float(edge[cost_key]) ** gamma
                for prev_gap, prev_cost in dp[t][b - 1].items():
                    penalty = 0.0 if prev_gap == 0 else smooth_lambda * ((gap - prev_gap) ** 2) / norm
                    cand = prev_cost + edge_cost + penalty
                    if cand < dp[s][b].get(gap, inf):
                        dp[s][b][gap] = cand
                        parent[s][b][gap] = (t, prev_gap)

    if not dp[n - 1][budget]:
        return None
    last_gap, best = min(dp[n - 1][budget].items(), key=lambda item: item[1])
    fresh = [n - 1]
    s, b, gap = n - 1, budget, last_gap
    while b > 1:
        t, prev_gap = parent[s][b][gap]
        fresh.append(t)
        s, b, gap = t, b - 1, prev_gap
    fresh.reverse()
    if fresh[0] != 0:
        return None

    per_edge = []
    for t, s in zip(fresh, fresh[1:]):
        e = costs[(t, s)]
        per_edge.append({"t": t, "s": s, **{k: e[k] for k in COST_KEYS}, "n_reuse": e["n_reuse"]})
    return {
        "fresh_steps": fresh,
        "path_cost": best,
        "worst_edge_cost": max((e[cost_key] for e in per_edge), default=0.0),
        "per_edge": per_edge,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-level edge-DP over StepTrace vectors.")
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--budgets", default="25")
    parser.add_argument("--gammas", default="1")
    parser.add_argument("--cost-keys", default="sum_dir_err,sum_vel_rel_mse,sum_chanmax_dir_err,peak_dir_err,peak_chanmax_dir_err")
    parser.add_argument("--aggregates", default="mean,median,p90,max,mean_plus_1")
    parser.add_argument("--max-gaps", default="4,8")
    parser.add_argument("--smooth-lambdas", default="0", help="comma-separated smooth-gap penalties; 0 uses ordinary DP")
    parser.add_argument("--policy-jvp-order", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    samples = load_dataset(args.result_root, args.rank)
    n = int(len(samples[0]["trace"]["steps"]))
    for sample in samples:
        steps = [int(step) for step in sample["trace"]["steps"]]
        if steps != list(range(n)):
            raise SystemExit(f"non-contiguous or mismatched steps in {sample['path']}")

    out_dir = args.out_dir or (args.result_root / "edge_dp_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = parse_ints(args.budgets)
    gammas = parse_floats(args.gammas)
    max_gaps = parse_ints(args.max_gaps)
    smooth_lambdas = parse_floats(args.smooth_lambdas)
    cost_keys = [key.strip() for key in args.cost_keys.split(",") if key.strip()]
    aggregates = [item.strip() for item in args.aggregates.split(",") if item.strip()]
    for cost_key in cost_keys:
        if cost_key not in COST_KEYS:
            raise SystemExit(f"unknown cost key: {cost_key}")

    dataset_meta = [
        {"task": s["task"], "prompt": s["prompt"], "seed": s["seed"], "rank": s["rank"], "path": str(s["path"])}
        for s in samples
    ]
    print(f"samples={len(samples)} rank={args.rank} steps={n}")
    print("dataset:", ", ".join(f"{s['prompt']}:{s['seed']}" for s in samples))

    max_edge_gap = max(max_gaps)
    print(f"computing per-sample edge costs max_gap={max_edge_gap}...")
    sample_costs = []
    for idx, sample in enumerate(samples, 1):
        print(f"  [{idx}/{len(samples)}] {sample['prompt']} seed{sample['seed']}")
        sample_costs.append(compute_edge_costs(sample["trace"], max_edge_gap))

    rows = []
    plans = []
    for cost_key in cost_keys:
        for aggregate in aggregates:
            agg_costs = aggregate_costs(sample_costs, aggregate)
            for max_gap in max_gaps:
                for budget in budgets:
                    for gamma in gammas:
                        for smooth_lambda in smooth_lambdas:
                            if smooth_lambda == 0:
                                algorithm = "dp"
                                result = edge_dp(agg_costs, n, budget, gamma, cost_key, max_gap)
                            else:
                                algorithm = f"smooth{smooth_lambda:g}"
                                result = smooth_edge_dp(agg_costs, n, budget, gamma, cost_key, max_gap, smooth_lambda)
                            if result is None:
                                continue
                            tag = f"{cost_key}_{aggregate}_gap{max_gap}_B{budget}_g{gamma:g}_{algorithm}"
                            stats = schedule_stats(result["fresh_steps"], sample_costs, cost_key)
                            row = {
                                "tag": tag,
                                "algorithm": algorithm,
                                "smooth_lambda": smooth_lambda,
                                "cost_key": cost_key,
                                "aggregate": aggregate,
                                "max_gap_constraint": max_gap,
                                "budget": budget,
                                "gamma": gamma,
                                "path_cost": result["path_cost"],
                                "worst_edge_cost": result["worst_edge_cost"],
                                "fresh_steps": " ".join(str(x) for x in result["fresh_steps"]),
                                "gap_hist": json.dumps(stats["gap_hist"], sort_keys=True),
                                **{k: v for k, v in stats.items() if k not in {"fresh_steps", "gap_hist", "sample_path_values"}},
                            }
                            rows.append(row)
                            plan = {
                                "case_id": f"datasetdp_{tag}",
                                "flexcache_params": fresh_to_policy(result["fresh_steps"], jvp_order=args.policy_jvp_order),
                                "planner": {
                                    "dataset": dataset_meta,
                                    "algorithm": algorithm,
                                    "smooth_lambda": smooth_lambda,
                                    "cost_key": cost_key,
                                    "aggregate": aggregate,
                                    "max_gap_constraint": max_gap,
                                    "budget": budget,
                                    "gamma": gamma,
                                    "path_cost": result["path_cost"],
                                    "stats": stats,
                                    "fresh_steps": result["fresh_steps"],
                                    "per_edge": result["per_edge"],
                                },
                            }
                            plans.append(plan)
                            (out_dir / f"policy_{tag}.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
                            plot_schedule(result, n, f"dataset edge-DP {tag}", out_dir / f"schedule_{tag}.png")
                            print(
                                f"  [ok] {tag}: max_gap={stats['max_gap']} warmup9={stats['warmup9']} "
                                f"smooth={stats['gap_smoothness']} J_hq={stats['jaccard_hqphase']:.2f} "
                                f"J_auto={stats['jaccard_autohq']:.2f}"
                            )

    write_csv(out_dir / "dataset_edge_dp_summary.csv", rows)
    (out_dir / "dataset_edge_dp_summary.json").write_text(
        json.dumps({"dataset": dataset_meta, "hqphase": HQPHASE, "autohq": AUTOHQ, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dataset_edge_dp_policies.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")

    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(row["fresh_steps"], {"fresh_steps": row["fresh_steps"], "count": 0, "tags": []})
        unique[row["fresh_steps"]]["count"] += 1
        unique[row["fresh_steps"]]["tags"].append(row["tag"])
    unique_rows = sorted(unique.values(), key=lambda item: (-item["count"], item["fresh_steps"]))
    (out_dir / "unique_schedules.json").write_text(json.dumps(unique_rows, indent=2), encoding="utf-8")
    print(f"\nwrote {len(rows)} rows, {len(unique_rows)} unique schedules to {out_dir}")
    print("top repeated schedules:")
    for item in unique_rows[:8]:
        print(f"  count={item['count']:>2} fresh={item['fresh_steps']}")


if __name__ == "__main__":
    main()
