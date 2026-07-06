#!/usr/bin/env python3
"""Generate a Poisson client-arrival trace for end-to-end serving experiments.

This models a *client* stream hitting the ChituDiffusion *server*: requests
arrive according to a Poisson process (exponential inter-arrival times at a
configurable rate), each carrying a real prompt/size/steps/seed so it can be
replayed on GPU by ``test/test_z_image_serve.py`` (the server-side replayer that
injects each request into the task pool when wall-clock reaches its arrival_ms).

Unlike the offline simulator traces (``traces/*.json`` with abstract ``profile``
names), this trace has concrete generation parameters and is meant to drive the
*real* engine. It supports mixed shapes (for M5/M7 experiments) and a small pool
of varied prompts (so cross-request batching sees distinct conditioning).

Usage examples:
    # 20 requests, ~2 req/s, single 1024x1024 shape
    python3 gen_client_trace.py --rate 2.0 --num-requests 20 \
        --out traces/serve/poisson_2rps.json

    # mixed resolution, ~4 req/s, 40 requests
    python3 gen_client_trace.py --rate 4.0 --num-requests 40 \
        --sizes 1024x1024,512x512 --size-weights 0.6,0.4 \
        --out traces/serve/poisson_mixed.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# A small pool of diverse prompts so that batching across requests exercises
# genuinely different text conditioning (not just different seeds).
DEFAULT_PROMPTS = [
    'A compact workstation desk with a small sign reading "Z-Image x ChituDiffusion", soft morning light, crisp product photography.',
    "A serene mountain lake at dawn, mirror-still water, pine forest, volumetric fog, ultra detailed landscape photography.",
    "A cyberpunk street market at night, neon signs, rain-slick pavement, reflective puddles, cinematic lighting.",
    "A cozy bookstore cafe interior, warm lamps, wooden shelves, steam rising from a coffee cup, shallow depth of field.",
    "A macro shot of a dew-covered spider web at sunrise, backlit droplets, bokeh background.",
    "An astronaut standing on a red desert planet, two moons in the sky, dramatic golden-hour light.",
    "A watercolor painting of a Japanese garden with a red bridge, koi pond, cherry blossoms.",
    "A futuristic electric sports car on a coastal highway, motion blur, sunset reflections on the hood.",
    "A steaming bowl of ramen with soft-boiled egg and scallions, top-down food photography, rich colors.",
    "A snowy alpine village at twilight, glowing windows, chimney smoke, starry sky, painterly style.",
    "A vibrant coral reef teeming with tropical fish, sun rays piercing clear blue water, wide angle.",
    "A vintage film camera on a wooden table beside dried flowers, soft natural window light, still life.",
]

DEFAULT_NEGATIVE_PROMPT = (
    "deformed, distorted, malformed, mutated, disfigured, gibberish text, garbled text, "
    "blurry, low quality, extra objects, duplicated objects, watermark, jpeg artifacts"
)


def _parse_sizes(text: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for chunk in text.split(","):
        chunk = chunk.strip().lower().replace("x", " ").replace("*", " ")
        w, h = (int(v) for v in chunk.split())
        sizes.append((w, h))
    return sizes


def _parse_weights(text: str | None, n: int) -> list[float]:
    if not text:
        return [1.0 / n] * n
    weights = [float(v) for v in text.split(",")]
    if len(weights) != n:
        raise ValueError(f"--size-weights count ({len(weights)}) must match --sizes count ({n}).")
    total = sum(weights)
    if total <= 0:
        raise ValueError("--size-weights must sum to a positive value.")
    return [w / total for w in weights]


def _parse_n_sample_dist(text: str) -> list[tuple[int, float]]:
    """Parse "1:0.8,2:0.2" into [(1,0.8),(2,0.2)] (n_sample -> probability)."""
    pairs: list[tuple[int, float]] = []
    for chunk in text.split(","):
        n_str, _, p_str = chunk.partition(":")
        pairs.append((int(n_str), float(p_str) if p_str else 1.0))
    total = sum(p for _, p in pairs)
    return [(n, p / total) for n, p in pairs]


def generate_trace(
    *,
    rate_req_per_s: float,
    num_requests: int,
    sizes: list[tuple[int, int]],
    size_weights: list[float],
    num_steps: int,
    n_sample_dist: list[tuple[int, float]],
    base_seed: int,
    rng_seed: int,
    prompts: list[str],
    negative_prompt: str,
    solver: str,
) -> dict:
    rng = random.Random(rng_seed)
    if rate_req_per_s <= 0:
        raise ValueError("--rate must be > 0.")
    mean_interarrival_ms = 1000.0 / rate_req_per_s

    n_values = [n for n, _ in n_sample_dist]
    n_probs = [p for _, p in n_sample_dist]

    requests = []
    arrival_ms = 0.0
    seed = base_seed
    for i in range(num_requests):
        # Poisson process: exponential inter-arrival times. First request at t=0.
        if i > 0:
            arrival_ms += rng.expovariate(1.0 / mean_interarrival_ms)
        width, height = rng.choices(sizes, weights=size_weights, k=1)[0]
        n_sample = rng.choices(n_values, weights=n_probs, k=1)[0]
        prompt = prompts[i % len(prompts)]
        requests.append(
            {
                "request_id": f"req{i:04d}",
                "arrival_ms": round(arrival_ms, 3),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": int(width),
                "height": int(height),
                "num_steps": int(num_steps),
                "n_sample": int(n_sample),
                "seed": int(seed),
                "sample_solver": solver,
            }
        )
        seed += n_sample  # keep per-image seeds globally distinct

    duration_ms = requests[-1]["arrival_ms"] if requests else 0.0
    return {
        "meta": {
            "generated_by": "gen_client_trace.py",
            "rate_req_per_s": rate_req_per_s,
            "num_requests": num_requests,
            "rng_seed": rng_seed,
            "base_seed": base_seed,
            "sizes": [f"{w}x{h}" for w, h in sizes],
            "size_weights": size_weights,
            "num_steps": num_steps,
            "n_sample_dist": {str(n): p for n, p in n_sample_dist},
            "duration_ms": round(duration_ms, 3),
            "mean_interarrival_ms": round(mean_interarrival_ms, 3),
        },
        "requests": requests,
    }


def load_client_trace(path: str | Path) -> list[dict]:
    """Load a client trace and return its requests sorted by arrival_ms."""
    data = json.loads(Path(path).read_text())
    requests = list(data.get("requests", []))
    requests.sort(key=lambda r: float(r["arrival_ms"]))
    return requests


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=2.0, help="Poisson arrival rate (requests/second).")
    ap.add_argument("--num-requests", type=int, default=20)
    ap.add_argument("--sizes", type=str, default="1024x1024", help="Comma list of WxH shapes.")
    ap.add_argument("--size-weights", type=str, default=None, help="Comma weights matching --sizes.")
    ap.add_argument("--steps", type=int, default=20, help="Denoise steps per request.")
    ap.add_argument("--n-sample-dist", type=str, default="1:1.0", help='e.g. "1:0.7,2:0.3" (n_sample -> prob).')
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--rng-seed", type=int, default=0, help="Seed for the arrival/shape/prompt RNG (reproducible).")
    ap.add_argument("--solver", type=str, default="flowmatch_euler")
    ap.add_argument("--out", type=str, required=True, help="Output trace JSON path.")
    args = ap.parse_args()

    sizes = _parse_sizes(args.sizes)
    size_weights = _parse_weights(args.size_weights, len(sizes))
    n_sample_dist = _parse_n_sample_dist(args.n_sample_dist)

    trace = generate_trace(
        rate_req_per_s=args.rate,
        num_requests=args.num_requests,
        sizes=sizes,
        size_weights=size_weights,
        num_steps=args.steps,
        n_sample_dist=n_sample_dist,
        base_seed=args.base_seed,
        rng_seed=args.rng_seed,
        prompts=DEFAULT_PROMPTS,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        solver=args.solver,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")

    meta = trace["meta"]
    print(f"Wrote {len(trace['requests'])} requests -> {out_path}")
    print(
        f"  rate={meta['rate_req_per_s']} req/s  duration={meta['duration_ms']:.1f} ms  "
        f"sizes={meta['sizes']} steps={meta['num_steps']} n_sample_dist={meta['n_sample_dist']}"
    )


if __name__ == "__main__":
    main()
