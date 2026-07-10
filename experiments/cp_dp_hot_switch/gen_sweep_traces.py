#!/usr/bin/env python3
"""Generate a family of Poisson client traces that differ ONLY in arrival rate.

For an arrival-rate sweep we want the *same workload* (same request shapes, seeds,
prompts, deadlines) at every load point, so the only thing changing across the
x-axis is how densely requests arrive. Because ``gen_client_trace.generate_trace``
draws the exponential inter-arrival gap and then the shape from the *same* RNG in
the same order, a fixed ``rng_seed`` yields an identical shape/seed sequence at any
rate -- only ``arrival_ms`` scales by ``mean_interarrival = 1000/rate``. This script
just calls it once per rate and writes ``poisson_r{rate}.json`` into ``--out-dir``.

Deadlines are a per-shape *relative* latency budget (ms after arrival), fixed across
rates (a request's SLO does not get easier just because the client sent it slower),
so SLO-attainment curves are comparable point-to-point.

Usage:
    python gen_sweep_traces.py --rates 0.1,0.16,0.24,0.36 --num-requests 12 \
        --sizes 1024x1024,512x512 --steps 50 \
        --deadline-map 1024:15000,512:14000 --out-dir traces/serve/sweep_tight
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gen_client_trace import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPTS,
    _parse_sizes,
    _parse_weights,
    generate_trace,
)


def _parse_deadline_map(text: str) -> dict[int, float]:
    """"1024:15000,512:14000" -> {1024: 15000.0, 512: 14000.0} (keyed by width)."""
    out: dict[int, float] = {}
    for chunk in text.split(","):
        w_str, _, ms_str = chunk.partition(":")
        out[int(w_str)] = float(ms_str)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", type=str, required=True, help="Comma list of Poisson rates (req/s).")
    ap.add_argument("--num-requests", type=int, default=12)
    ap.add_argument("--sizes", type=str, default="1024x1024,512x512")
    ap.add_argument("--size-weights", type=str, default="0.5,0.5")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--rng-seed", type=int, default=7, help="Fixed so all rates share one workload.")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--deadline-map", type=str, default="1024:15000,512:14000",
                    help="Per-width relative SLO budget (ms), e.g. '1024:15000,512:14000'.")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    sizes = _parse_sizes(args.sizes)
    size_weights = _parse_weights(args.size_weights, len(sizes))
    rates = [float(r) for r in args.rates.split(",")]
    dl_map = _parse_deadline_map(args.deadline_map)

    def deadline_fn(width: int, height: int, steps: int) -> float | None:
        return dl_map.get(int(width))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rate in rates:
        trace = generate_trace(
            rate_req_per_s=rate,
            num_requests=args.num_requests,
            sizes=sizes,
            size_weights=size_weights,
            num_steps=args.steps,
            n_sample_dist=[(1, 1.0)],
            base_seed=args.base_seed,
            rng_seed=args.rng_seed,
            prompts=DEFAULT_PROMPTS,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            solver="flowmatch_euler",
            burst_sizes=None,
            deadline_fn=deadline_fn,
        )
        trace["meta"]["sweep_deadline_map"] = {str(k): v for k, v in dl_map.items()}
        tag = f"{rate:g}".replace(".", "p")
        path = out_dir / f"poisson_r{tag}.json"
        path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
        span = trace["meta"]["duration_ms"] / 1000.0
        shapes = [r["width"] for r in trace["requests"]]
        n1024 = sum(1 for w in shapes if w >= 1024)
        print(f"rate={rate:<5g} -> {path.name}  span={span:6.1f}s  "
              f"n={len(trace['requests'])} (1024:{n1024} 512:{len(shapes)-n1024})")
        manifest.append({"rate": rate, "trace": str(path), "span_s": span})

    (out_dir / "manifest.json").write_text(json.dumps({"sweep": manifest}, indent=2) + "\n")
    print(f"\nWrote {len(rates)} traces + manifest.json -> {out_dir}")


if __name__ == "__main__":
    main()
