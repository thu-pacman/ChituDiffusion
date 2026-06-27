#!/usr/bin/env python3
"""TracePlanner edge-DP planner (order fixed to 1, multi-step rollout cost).

Design decisions (see experiments/freecache/traceplanner/PLAN.md and WORKLOG):

- 去掉多阶搜索: order is fixed to 1. The MeanCache order ablation (2026-06-24)
  showed order 0/1/5 are within noise on the fresh=25 schedule; the decisive
  factor is the *position* (phase) of fresh steps, not the reuse order.
- 增加搜索深度: edge cost is a multi-step rollout. For an edge (t -> s) the
  reuse steps t+1..s-1 are predicted from the single fresh anchor t (order-1
  sigma-JVP), the latent is rolled forward open-loop, and we measure the
  accumulated drift vs the full-compute trajectory at every reuse step. This is
  the "往前走一步" depth direction rather than a one-step velocity error.
- peak suppression: path cost = sum(edge_cost ** gamma). gamma > 1 makes a
  single catastrophic edge dominate so the DP cannot hide a bad span inside a
  good average.

The planner runs purely on CPU using the saved full-compute steptrace vectors.
It emits FreeCache-compatible policy JSON (forced_compute_steps) which must be
verified by an actual FreeCache replay (this script does NOT trust its own
predicted scores).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

# Reuse the vetted JVP / metric helpers from the order probe script.
from order_probe_steptrace import (  # type: ignore
    cosine,
    load_payloads,
    predict_velocity,
    rel_mse,
    select_trace,
)


COST_KEYS = (
    "peak_drift",
    "endpoint_drift",
    "peak_dir_err",
    "peak_vel_rel_mse",
    # channel-wise velocity errors: per-channel error (reduce over batch+tokens,
    # keep the last channel dim) then aggregate across channels with max
    # (worst channel) or mean. The worst-channel variants amplify the few
    # phase-sensitive channels that global flatten metrics average away.
    "peak_chanmax_dir_err",
    "peak_chanmean_dir_err",
    "peak_chanmax_vel_rel_mse",
    "peak_chanmean_vel_rel_mse",
    # integrated (sum over the reuse steps in the span) velocity costs. Unlike
    # the peak-only variants these GROW with gap length, so the DP can no longer
    # hide a long reuse run whose per-step peak happens to be modest. This
    # targets the diagnosed failure mode (2026-06-24): peak-only cost let the DP
    # place three 7-step gaps that hand-tuned hqphase (max gap 3) never allows.
    "sum_dir_err",
    "sum_vel_rel_mse",
    "sum_chanmax_dir_err",
    "sum_chanmax_vel_rel_mse",
)


def _reshape_like(scalar: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return scalar.view(-1, *([1] * (ref.ndim - scalar.dim())))


def chan_rel_mse(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-channel relative MSE. Channel is the last dim; returns shape (C,)."""
    diff = (current - reference).to(torch.float64)
    ref = reference.to(torch.float64)
    dims = tuple(range(current.ndim - 1))
    num = diff.square().mean(dim=dims)
    den = ref.square().mean(dim=dims).clamp_min(1e-20)
    return num / den


def chan_dir_err(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-channel direction error 1 - cos. Channel is last dim; returns (C,)."""
    c = current.to(torch.float64)
    r = reference.to(torch.float64)
    dims = tuple(range(current.ndim - 1))
    dot = (c * r).sum(dim=dims)
    norm = (c.square().sum(dim=dims).sqrt() * r.square().sum(dim=dims).sqrt()).clamp_min(1e-20)
    return 1.0 - dot / norm


def precompute_order1_jvp(trace: dict) -> dict | None:
    """Precompute per-anchor order-1 sigma-JVP for the guided_only mode.

    Returns None for cfg_branch mode (falls back to predict_velocity), since the
    saved Qwen post-CFG vectors are guided_only.
    """
    if trace["mode"] != "guided_only":
        return None
    velocities = trace["guided_v"].to(torch.float32)
    latents_pre = trace["latents_pre"].to(torch.float32)
    latents = trace["latents"].to(torch.float32)
    sigmas_pre = trace["sigmas_pre"].to(torch.float32)
    sigmas = trace["sigmas"].to(torch.float32)
    n = velocities.shape[0]
    jvp = torch.empty_like(velocities)
    for t in range(n):
        denom = sigmas[t] - sigmas_pre[t]
        denom_b = _reshape_like(denom, latents_pre[t])
        avg_u = (latents[t] - latents_pre[t]) / denom_b
        jvp[t] = (velocities[t] - avg_u) / denom_b
    return {
        "velocities": velocities,
        "jvp": jvp,
        "sigmas_pre": sigmas_pre,
        "sigmas": sigmas,
    }


def predict_v_hat(trace: dict, fast: dict | None, anchor: int, target: int) -> torch.Tensor:
    """order-1 velocity prediction for reuse step `target` from fresh anchor `t`."""
    if fast is not None:
        delta_sigma = fast["sigmas"][target] - fast["sigmas_pre"][target]
        v_last = fast["velocities"][anchor]
        return v_last - fast["jvp"][anchor] * _reshape_like(delta_sigma, v_last)
    return predict_velocity(trace, [anchor], 0, target, order=1)


def compute_edge_costs(trace: dict, max_gap: int) -> dict:
    """edge_cost[(t, s)] -> dict of the four cost candidates.

    The edge (t, s) means t and s are fresh; steps t+1..s-1 are reused (order-1
    JVP from anchor t). Adjacent fresh (s == t+1) has zero cost (no reuse).
    """
    fast = precompute_order1_jvp(trace)
    latents = trace["latents"].to(torch.float32)
    guided_v = trace["guided_v"].to(torch.float32)
    sigmas_pre = trace["sigmas_pre"].to(torch.float32)
    sigmas = trace["sigmas"].to(torch.float32)
    n = latents.shape[0]

    costs: dict[tuple[int, int], dict] = {}
    for t in range(n - 1):
        s_max = min(n - 1, t + max_gap)
        for s in range(t + 1, s_max + 1):
            if s == t + 1:
                costs[(t, s)] = {**{k: 0.0 for k in COST_KEYS}, "n_reuse": 0}
                continue
            x_hat = latents[t].clone()
            peak = {k: 0.0 for k in COST_KEYS}
            endpoint_drift = 0.0
            for i in range(t + 1, s):
                v_hat = predict_v_hat(trace, fast, t, i)
                dir_g = 1.0 - cosine(v_hat, guided_v[i])
                relmse_g = rel_mse(v_hat, guided_v[i])
                chan_dir = chan_dir_err(v_hat, guided_v[i])
                chan_relmse = chan_rel_mse(v_hat, guided_v[i])
                cmax_dir = float(chan_dir.max())
                cmax_relmse = float(chan_relmse.max())
                peak["peak_dir_err"] = max(peak["peak_dir_err"], dir_g)
                peak["peak_vel_rel_mse"] = max(peak["peak_vel_rel_mse"], relmse_g)
                peak["peak_chanmax_dir_err"] = max(peak["peak_chanmax_dir_err"], cmax_dir)
                peak["peak_chanmean_dir_err"] = max(peak["peak_chanmean_dir_err"], float(chan_dir.mean()))
                peak["peak_chanmax_vel_rel_mse"] = max(peak["peak_chanmax_vel_rel_mse"], cmax_relmse)
                peak["peak_chanmean_vel_rel_mse"] = max(peak["peak_chanmean_vel_rel_mse"], float(chan_relmse.mean()))
                # integrated variants accumulate over the reuse span (gap length)
                peak["sum_dir_err"] += dir_g
                peak["sum_vel_rel_mse"] += relmse_g
                peak["sum_chanmax_dir_err"] += cmax_dir
                peak["sum_chanmax_vel_rel_mse"] += cmax_relmse
                dt = sigmas[i] - sigmas_pre[i]
                x_hat = x_hat + _reshape_like(dt, x_hat) * v_hat
                drift = rel_mse(x_hat, latents[i])
                peak["peak_drift"] = max(peak["peak_drift"], drift)
                endpoint_drift = drift
            peak["endpoint_drift"] = endpoint_drift
            costs[(t, s)] = {**peak, "n_reuse": s - t - 1}
    return costs


def edge_dp(
    costs: dict,
    n: int,
    budget: int,
    gamma: float,
    cost_key: str,
    max_gap: int,
) -> dict | None:
    """Peak-suppressed shortest path with exactly `budget` fresh steps.

    State dp[s][b] = min path cost reaching fresh step s using b fresh steps,
    with step 0 and step n-1 forced fresh.
    """
    inf = float("inf")
    dp = [[inf] * (budget + 1) for _ in range(n)]
    parent = [[-1] * (budget + 1) for _ in range(n)]
    dp[0][1] = 0.0
    for s in range(1, n):
        for b in range(2, budget + 1):
            best = inf
            best_t = -1
            t_lo = max(0, s - max_gap)
            for t in range(t_lo, s):
                if dp[t][b - 1] == inf:
                    continue
                edge = costs.get((t, s))
                if edge is None:
                    continue
                cand = dp[t][b - 1] + edge[cost_key] ** gamma
                if cand < best:
                    best = cand
                    best_t = t
            dp[s][b] = best
            parent[s][b] = best_t

    if dp[n - 1][budget] == inf:
        return None

    # Backtrack.
    fresh = [n - 1]
    s, b = n - 1, budget
    while b > 1:
        t = parent[s][b]
        if t < 0:
            return None
        fresh.append(t)
        s, b = t, b - 1
    fresh.reverse()
    if fresh[0] != 0:
        return None

    per_edge = []
    for t, s in zip(fresh, fresh[1:]):
        e = costs[(t, s)]
        per_edge.append({"t": t, "s": s, **{k: e[k] for k in COST_KEYS}, "n_reuse": e["n_reuse"]})
    return {
        "fresh_steps": fresh,
        "path_cost": dp[n - 1][budget],
        "worst_edge_cost": max((e[cost_key] for e in per_edge), default=0.0),
        "per_edge": per_edge,
    }


def fresh_to_policy(fresh: list[int], jvp_order: int = 1) -> dict:
    return {
        "strategy": "freecache",
        "jvp_order": jvp_order,
        "warmup": 0,
        "cooldown": 0,
        "forced_compute_steps": list(fresh),
    }


def plot_schedule(result: dict, n: int, title: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    fresh = set(result["fresh_steps"])
    fig, ax = plt.subplots(figsize=(14, 2.2))
    for step in range(n):
        is_fresh = step in fresh
        ax.bar(step, 1.0, width=0.9, color="#1f77b4" if is_fresh else "#d9d9d9")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("denoise step (blue = fresh / compute)")
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge-DP cache planner over StepTrace vectors (order=1, rollout cost).")
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--budgets", default="20,25,30", help="comma-separated fresh-step budgets")
    parser.add_argument("--gammas", default="1,2,4", help="comma-separated peak-suppression exponents")
    parser.add_argument("--cost-key", default="peak_drift", choices=COST_KEYS + ("all",))
    parser.add_argument("--max-gap", type=int, default=8, help="maximum reuse span per edge")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    trace = select_trace(load_payloads(args.result_root))
    n = int(len(trace["steps"]))
    steps = [int(s) for s in trace["steps"]]
    if steps != list(range(n)):
        raise SystemExit(f"Vector trace must be contiguous zero-based steps, got {steps[:5]}...{steps[-5:]}")
    print(f"trace mode={trace['mode']} steps={n} guidance_scale={trace['guidance_scale']}")

    out_dir = args.out_dir or (args.result_root / "edge_dp")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("computing edge costs (order=1 multi-step rollout)...")
    costs = compute_edge_costs(trace, args.max_gap)
    print(f"  computed {len(costs)} edges (max_gap={args.max_gap})")

    cost_keys = list(COST_KEYS) if args.cost_key == "all" else [args.cost_key]
    budgets = parse_int_list(args.budgets)
    gammas = parse_float_list(args.gammas)

    summary_rows = []
    for cost_key in cost_keys:
        for budget in budgets:
            for gamma in gammas:
                result = edge_dp(costs, n, budget, gamma, cost_key, args.max_gap)
                if result is None:
                    print(f"  [skip] cost={cost_key} B={budget} gamma={gamma}: infeasible")
                    continue
                tag = f"{cost_key}_B{budget}_g{gamma:g}"
                policy = fresh_to_policy(result["fresh_steps"])
                (out_dir / f"policy_{tag}.json").write_text(
                    json.dumps(
                        {
                            "case_id": f"edgedp_{tag}",
                            "flexcache_params": policy,
                            "planner": {
                                "cost_key": cost_key,
                                "budget": budget,
                                "gamma": gamma,
                                "max_gap": args.max_gap,
                                "path_cost": result["path_cost"],
                                "worst_edge_cost": result["worst_edge_cost"],
                                "fresh_steps": result["fresh_steps"],
                                "per_edge": result["per_edge"],
                            },
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                plot_schedule(result, n, f"edge-DP {tag} | {len(result['fresh_steps'])} fresh", out_dir / f"schedule_{tag}.png")
                max_gap_used = max((e["s"] - e["t"] for e in result["per_edge"]), default=0)
                summary_rows.append(
                    {
                        "cost_key": cost_key,
                        "budget": budget,
                        "gamma": gamma,
                        "n_fresh": len(result["fresh_steps"]),
                        "path_cost": result["path_cost"],
                        "worst_edge_cost": result["worst_edge_cost"],
                        "max_gap_used": max_gap_used,
                        "fresh_steps": " ".join(str(x) for x in result["fresh_steps"]),
                    }
                )
                print(f"  [ok] {tag}: fresh={result['fresh_steps']}")

    if summary_rows:
        keys = ["cost_key", "budget", "gamma", "n_fresh", "path_cost", "worst_edge_cost", "max_gap_used", "fresh_steps"]
        with (out_dir / "edge_dp_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        (out_dir / "edge_dp_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        print(f"wrote {len(summary_rows)} configs to {out_dir}")
    else:
        print("no feasible configs")


if __name__ == "__main__":
    main()
