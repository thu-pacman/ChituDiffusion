#!/usr/bin/env python3
"""Generate deterministic serving traces with controlled arrival phases.

Poisson traces are useful for load sweeps, but a single unlucky early cluster can
turn every visualization into "all requests are queued immediately". This file
creates small, inspectable traces with explicit warmup, burst, and cooldown
windows so DP/SP scheduling opportunities are visible in the timeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from gen_client_trace import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPTS


@dataclass(frozen=True)
class TraceItem:
    arrival_s: float
    width: int
    height: int
    n_sample: int = 1
    num_steps: int = 20


PRESETS: dict[str, list[TraceItem]] = {
    # Early requests are intentionally sparse, the middle contains one short
    # burst, and the tail cools down again. This should expose when SP lowers
    # latency under light load and when DP replica count matters under bursts.
    "staggered_mixed": [
        TraceItem(0.0, 1024, 1024),
        TraceItem(16.0, 512, 512),
        TraceItem(27.0, 1024, 1024),
        TraceItem(42.0, 512, 512, n_sample=2),
        TraceItem(50.0, 1024, 1024),
        TraceItem(52.5, 1024, 1024),
        TraceItem(55.0, 512, 512),
        TraceItem(57.5, 1024, 1024, n_sample=2),
        TraceItem(73.0, 512, 512),
        TraceItem(87.0, 1024, 1024),
        TraceItem(103.0, 512, 512, n_sample=2),
        TraceItem(112.0, 1024, 1024),
        TraceItem(114.0, 512, 512),
        TraceItem(120.0, 1024, 1024),
        TraceItem(137.0, 512, 512),
    ],
    # A more service-like synthetic trace: uneven background arrivals, two short
    # user bursts, and mixed SD3-like image sizes. It is intentionally small
    # enough for SVG inspection but has enough structure to exercise a scheduler.
    "realistic_sd3_mixed": [
        TraceItem(0.0, 512, 512, num_steps=50),
        TraceItem(18.0, 1024, 1024, num_steps=50),
        TraceItem(43.0, 256, 256, num_steps=50),
        TraceItem(64.0, 1536, 1536, num_steps=50),
        TraceItem(91.0, 128, 128, num_steps=50),
        TraceItem(116.0, 1024, 1024, num_steps=50),
        TraceItem(121.0, 512, 512, num_steps=50),
        TraceItem(123.5, 1536, 1536, num_steps=50),
        TraceItem(126.0, 256, 256, num_steps=50),
        TraceItem(129.5, 1024, 1024, num_steps=50),
        TraceItem(137.0, 128, 128, num_steps=50),
        TraceItem(162.0, 512, 512, num_steps=50),
        TraceItem(187.0, 1536, 1536, num_steps=50),
        TraceItem(219.0, 256, 256, num_steps=50),
        TraceItem(240.0, 1024, 1024, num_steps=50),
        TraceItem(245.0, 512, 512, num_steps=50),
        TraceItem(247.0, 512, 512, num_steps=50),
        TraceItem(250.5, 1536, 1536, num_steps=50),
        TraceItem(254.0, 128, 128, num_steps=50),
        TraceItem(261.0, 1024, 1024, num_steps=50),
        TraceItem(296.0, 256, 256, num_steps=50),
        TraceItem(331.0, 1536, 1536, num_steps=50),
        TraceItem(365.0, 512, 512, num_steps=50),
        TraceItem(404.0, 1024, 1024, num_steps=50),
        TraceItem(430.0, 128, 128, num_steps=50),
        TraceItem(468.0, 1536, 1536, num_steps=50),
    ],
}


def generate_trace(*, preset: str, base_seed: int, solver: str) -> dict:
    items = PRESETS[preset]
    requests = []
    seed = base_seed
    for i, item in enumerate(items):
        requests.append(
            {
                "request_id": f"req{i:04d}",
                "arrival_ms": round(item.arrival_s * 1000.0, 3),
                "prompt": DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)],
                "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
                "width": item.width,
                "height": item.height,
                "num_steps": item.num_steps,
                "n_sample": item.n_sample,
                "seed": seed,
                "sample_solver": solver,
            }
        )
        seed += item.n_sample

    arrivals = [item.arrival_s for item in items]
    return {
        "meta": {
            "generated_by": "gen_structured_trace.py",
            "preset": preset,
            "pattern": "deterministic warmup + short burst + cooldown",
            "num_requests": len(requests),
            "base_seed": base_seed,
            "duration_ms": round(max(arrivals) * 1000.0 if arrivals else 0.0, 3),
            "arrival_s": arrivals,
        },
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="staggered_mixed")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--solver", type=str, default="flowmatch_euler")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "traces" / "serve" / "staggered_mixed.json")
    args = parser.parse_args()

    trace = generate_trace(preset=args.preset, base_seed=args.base_seed, solver=args.solver)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")

    meta = trace["meta"]
    print(f"Wrote {meta['num_requests']} requests -> {args.out}")
    print(f"  preset={meta['preset']}  duration={meta['duration_ms'] / 1000.0:.1f}s")


if __name__ == "__main__":
    main()
