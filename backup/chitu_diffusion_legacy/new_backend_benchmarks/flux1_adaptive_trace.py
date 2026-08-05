from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPTS = (
    "A compact workstation desk in soft morning light, product photography.",
    "A glass greenhouse filled with tropical plants after rainfall.",
    "A futuristic train entering a clean white station.",
    "A mountain lake at dawn with pine forest and volumetric fog.",
    "A ceramic teapot beside green leaves, studio lighting.",
    "A modern library interior with warm lamps and wooden shelves.",
)


@dataclass(frozen=True, slots=True)
class ResolutionCost:
    height: int
    width: int
    cp1_latency_ms: float
    full_cp_latency_ms: float


def _request_latency_ms(row: dict[str, Any], steps: int) -> float:
    return steps * float(row["latency_ms"]) + float(row.get("terminal_ms", 0.0))


def derive_resolution_costs(
    warmup: dict[str, Any],
    *,
    steps: int,
    world_size: int,
) -> list[ResolutionCost]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    rows_by_key = {
        (tuple(int(value) for value in row["resolution"]), int(row["width"])): row
        for row in warmup.get("rows", ())
    }
    resolutions = sorted(
        {resolution for resolution, width in rows_by_key if width == 1}
    )
    if not resolutions:
        raise ValueError("warmup report has no cp1 rows")
    costs = []
    for resolution in resolutions:
        cp1 = rows_by_key.get((resolution, 1))
        full_cp = rows_by_key.get((resolution, world_size))
        if cp1 is None or full_cp is None:
            raise ValueError(
                f"resolution {resolution} requires width 1 and {world_size} warmup rows"
            )
        costs.append(
            ResolutionCost(
                height=resolution[0],
                width=resolution[1],
                cp1_latency_ms=_request_latency_ms(cp1, steps),
                full_cp_latency_ms=_request_latency_ms(full_cp, steps),
            )
        )
    return costs


def build_adaptive_trace(
    warmup: dict[str, Any],
    *,
    steps: int,
    requests_per_phase: int = 6,
    world_size: int = 4,
    slo_factor: float = 1.2,
    sparse_factor: float = 0.5,
    overload_factor: float = 1.2,
    phase_gap_factor: float = 1.1,
    seed: int = 7,
) -> dict[str, Any]:
    if requests_per_phase < 1:
        raise ValueError("requests_per_phase must be positive")
    if slo_factor <= 0 or sparse_factor <= 0 or overload_factor <= 0:
        raise ValueError("SLO and load factors must be positive")
    if phase_gap_factor < 0:
        raise ValueError("phase_gap_factor must be non-negative")

    costs = derive_resolution_costs(warmup, steps=steps, world_size=world_size)
    mean_cp1_ms = sum(item.cp1_latency_ms for item in costs) / len(costs)
    mean_full_cp_ms = sum(item.full_cp_latency_ms for item in costs) / len(costs)
    static_dp_capacity = world_size * 1000.0 / mean_cp1_ms
    static_cp_capacity = 1000.0 / mean_full_cp_ms
    lower_capacity = min(static_dp_capacity, static_cp_capacity)
    upper_capacity = max(static_dp_capacity, static_cp_capacity)
    slo_ms = slo_factor * max(item.cp1_latency_ms for item in costs)
    phase_gap_ms = phase_gap_factor * max(item.full_cp_latency_ms for item in costs)

    phases = (
        ("sparse", sparse_factor * lower_capacity),
        ("crossover", math.sqrt(lower_capacity * upper_capacity)),
        ("overload", overload_factor * upper_capacity),
    )
    rng = random.Random(seed)
    requests: list[dict[str, Any]] = []
    phase_meta = []
    phase_start_ms = 0.0
    request_index = 0
    for phase_index, (name, rate) in enumerate(phases):
        interval_ms = 1000.0 / rate
        phase_costs = [costs[index % len(costs)] for index in range(requests_per_phase)]
        rng.shuffle(phase_costs)
        for index, cost in enumerate(phase_costs):
            arrival_ms = phase_start_ms + index * interval_ms
            requests.append(
                {
                    "request_id": f"flux-{name}-{index:02d}",
                    "arrival_ms": round(arrival_ms, 3),
                    "prompt": PROMPTS[request_index % len(PROMPTS)],
                    "width": cost.width,
                    "height": cost.height,
                    "num_steps": steps,
                    "seed": seed * 1000 + request_index,
                    "deadline_ms": round(slo_ms, 3),
                    "priority": 0,
                    "phase": name,
                    "offered_rate_req_per_s": rate,
                }
            )
            request_index += 1
        phase_end_ms = phase_start_ms + (requests_per_phase - 1) * interval_ms
        phase_meta.append(
            {
                "name": name,
                "index": phase_index,
                "start_ms": round(phase_start_ms, 3),
                "end_ms": round(phase_end_ms, 3),
                "requests": requests_per_phase,
                "rate_req_per_s": rate,
                "load_vs_static_dp": rate / static_dp_capacity,
                "load_vs_static_cp": rate / static_cp_capacity,
            }
        )
        phase_start_ms = phase_end_ms + phase_gap_ms

    return {
        "meta": {
            "title": "Flux.1 warmup-adaptive static-DP / static-CP / EPE trace",
            "generated_by": "flux1_adaptive_trace.py",
            "num_requests": len(requests),
            "num_steps": steps,
            "world_size": world_size,
            "resolutions": [[item.height, item.width] for item in costs],
            "requests_per_phase": requests_per_phase,
            "duration_ms": requests[-1]["arrival_ms"],
            "capacity_estimate": {
                "method": "equal-resolution mix from measured denoise plus terminal",
                "static_dp_req_per_s": static_dp_capacity,
                "static_cp_req_per_s": static_cp_capacity,
                "lower_req_per_s": lower_capacity,
                "upper_req_per_s": upper_capacity,
            },
            "slo": {
                "factor": slo_factor,
                "deadline_ms": slo_ms,
                "definition": "factor * longest cp1 (steps * denoise + terminal)",
            },
            "phase_gap_ms": phase_gap_ms,
            "phases": phase_meta,
        },
        "requests": requests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a three-load Flux.1 trace from measured startup warmup"
    )
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--requests-per-phase", type=int, default=6)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--slo-factor", type=float, default=1.2)
    parser.add_argument("--sparse-factor", type=float, default=0.5)
    parser.add_argument("--overload-factor", type=float, default=1.2)
    parser.add_argument("--phase-gap-factor", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warmup = json.loads(args.warmup.read_text(encoding="utf-8"))
    trace = build_adaptive_trace(
        warmup,
        steps=args.steps,
        requests_per_phase=args.requests_per_phase,
        world_size=args.world_size,
        slo_factor=args.slo_factor,
        sparse_factor=args.sparse_factor,
        overload_factor=args.overload_factor,
        phase_gap_factor=args.phase_gap_factor,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps(trace["meta"], indent=2))


if __name__ == "__main__":
    main()
