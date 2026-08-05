from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROMPTS = (
    "A small sailboat crosses a calm alpine lake at sunrise, cinematic realism.",
    "A red panda walks through a bamboo forest after rain, natural documentary.",
    "A chef plates an elegant dessert in a bright modern kitchen, close-up shot.",
    "Ocean waves roll onto a black sand beach beneath a cloudy sky, wide shot.",
    "A vintage tram moves through a European street at dusk, cinematic tracking shot.",
    "A golden retriever runs through a field of wildflowers, slow motion.",
    "An astronaut explores a quiet lunar base while Earth rises in the distance.",
    "A glass sculpture rotates on a pedestal under soft studio lighting.",
)


def _request_latency_ms(row: dict[str, Any], steps: int) -> float:
    return steps * float(row["latency_ms"]) + float(row.get("terminal_ms", 0.0))


def build_wan_adaptive_trace(
    warmup: dict[str, Any],
    *,
    slo_ms: float,
    height: int,
    width: int,
    num_frames: int,
    steps: int,
    world_size: int,
    lane_width: int = 2,
    requests_per_phase: int = 8,
    sparse_factor: float = 0.5,
    balanced_factor: float = 0.9,
    overload_factor: float = 1.2,
    phase_gap_factor: float = 1.25,
    seed: int = 7,
) -> dict[str, Any]:
    if requests_per_phase < 1:
        raise ValueError("requests_per_phase must be positive")
    if slo_ms <= 0 or steps < 1 or world_size < 1:
        raise ValueError("SLO, steps, and world size must be positive")
    if lane_width < 1 or world_size % lane_width:
        raise ValueError("lane width must divide world size")
    if num_frames < 1 or (num_frames - 1) % 4:
        raise ValueError("Wan num_frames must equal 4n+1")

    target_row = next(
        (
            row
            for row in warmup.get("rows", ())
            if tuple(int(value) for value in row["resolution"]) == (height, width)
            and int(row["width"]) == lane_width
        ),
        None,
    )
    if target_row is None:
        raise ValueError(
            f"warmup report has no {(height, width)} width={lane_width} row"
        )

    measured_lane_ms = _request_latency_ms(target_row, steps)
    lane_count = world_size // lane_width
    capacity = lane_count * 1000.0 / measured_lane_ms
    phase_gap_ms = phase_gap_factor * max(slo_ms, measured_lane_ms)
    phases = (
        ("sparse", sparse_factor * capacity),
        ("balanced", balanced_factor * capacity),
        ("overload", overload_factor * capacity),
    )
    latent_frames = (num_frames - 1) // 4 + 1
    image_tokens = latent_frames * (height // 16) * (width // 16)
    requests: list[dict[str, Any]] = []
    phase_meta: list[dict[str, Any]] = []
    phase_start_ms = 0.0
    request_index = 0
    for phase_index, (name, rate) in enumerate(phases):
        interval_ms = 1000.0 / rate
        for index in range(requests_per_phase):
            requests.append(
                {
                    "request_id": f"wan-{name}-{index:02d}",
                    "arrival_ms": round(phase_start_ms + index * interval_ms, 3),
                    "prompt": PROMPTS[request_index % len(PROMPTS)],
                    "negative_prompt": "",
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "num_steps": steps,
                    "image_tokens": image_tokens,
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
                "load_vs_measured_cfp2_capacity": rate / capacity,
            }
        )
        phase_start_ms = phase_end_ms + phase_gap_ms

    return {
        "meta": {
            "title": "Wan2.1 T2V 1.3B warmup-adaptive EPAC trace",
            "generated_by": "wan_adaptive_trace.py",
            "num_requests": len(requests),
            "num_steps": steps,
            "num_frames": num_frames,
            "world_size": world_size,
            "resolution": [height, width],
            "requests_per_phase": requests_per_phase,
            "duration_ms": requests[-1]["arrival_ms"],
            "capacity_estimate": {
                "method": "measured CFP2 denoise plus leader-only VAE/D2H terminal",
                "lane_width": lane_width,
                "lane_count": lane_count,
                "request_latency_ms": measured_lane_ms,
                "capacity_req_per_s": capacity,
            },
            "slo": {
                "deadline_ms": slo_ms,
                "definition": "measured two-GPU CFP end-to-end request latency",
            },
            "phase_gap_ms": phase_gap_ms,
            "phases": phase_meta,
        },
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Wan adaptive trace from measured CFP2 latency and warmup"
    )
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slo-ms", type=float, required=True)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--lane-width", type=int, default=2)
    parser.add_argument("--requests-per-phase", type=int, default=8)
    parser.add_argument("--sparse-factor", type=float, default=0.5)
    parser.add_argument("--balanced-factor", type=float, default=0.9)
    parser.add_argument("--overload-factor", type=float, default=1.2)
    parser.add_argument("--phase-gap-factor", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    trace = build_wan_adaptive_trace(
        json.loads(args.warmup.read_text(encoding="utf-8")),
        slo_ms=args.slo_ms,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        steps=args.steps,
        world_size=args.world_size,
        lane_width=args.lane_width,
        requests_per_phase=args.requests_per_phase,
        sparse_factor=args.sparse_factor,
        balanced_factor=args.balanced_factor,
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
