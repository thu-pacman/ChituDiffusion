#!/usr/bin/env python3
"""Convert an SD3-style text trace into the local serving JSON format.

Input lines look like:

    Request(request_id=0, timestamp=0.0, height=256, width=256, ...)

The upstream trace can be much denser than a small offline 4-GPU simulator can
consume. Use ``--stride`` and ``--time-scale`` to produce a manageable slice
while preserving the request mix.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _parse_request_line(line: str) -> dict[str, Any]:
    expr = ast.parse(line.strip(), mode="eval").body
    if not isinstance(expr, ast.Call) or getattr(expr.func, "id", None) != "Request":
        raise ValueError(f"not a Request(...) line: {line[:80]}")
    parsed: dict[str, Any] = {}
    for keyword in expr.keywords:
        if keyword.arg is None:
            raise ValueError("unexpected **kwargs in Request(...) line")
        parsed[keyword.arg] = ast.literal_eval(keyword.value)
    return parsed


def load_sd3_trace(path: Path) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            requests.append(_parse_request_line(line))
        except Exception as exc:  # noqa: BLE001 - report source line in CLI tool
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    requests.sort(key=lambda item: (float(item["timestamp"]), int(item["request_id"])))
    return requests


def convert_requests(
    requests: list[dict[str, Any]],
    *,
    start_s: float,
    duration_s: float | None,
    stride: int,
    time_scale: float,
    max_requests: int | None,
    id_prefix: str,
    solver: str,
) -> dict[str, Any]:
    if stride <= 0:
        raise ValueError("--stride must be >= 1")
    if time_scale <= 0:
        raise ValueError("--time-scale must be > 0")

    end_s = float("inf") if duration_s is None else start_s + duration_s
    windowed = [item for item in requests if start_s <= float(item["timestamp"]) < end_s]
    selected = windowed[::stride]
    if max_requests is not None:
        selected = selected[:max_requests]

    converted = []
    for out_idx, item in enumerate(selected):
        arrival_ms = (float(item["timestamp"]) - start_s) * time_scale * 1000.0
        converted.append(
            {
                "request_id": f"{id_prefix}{out_idx:04d}",
                "source_request_id": int(item["request_id"]),
                "arrival_ms": round(arrival_ms, 3),
                "prompt": str(item["prompt"]),
                "negative_prompt": str(item.get("negative_prompt", "")),
                "width": int(item["width"]),
                "height": int(item["height"]),
                "num_steps": int(item["num_inference_steps"]),
                "n_sample": 1,
                "seed": int(item["request_id"]),
                "sample_solver": solver,
            }
        )

    first_ts = float(selected[0]["timestamp"]) if selected else None
    last_ts = float(selected[-1]["timestamp"]) if selected else None
    return {
        "meta": {
            "generated_by": "convert_sd3_trace.py",
            "source_format": "Request(...) text lines",
            "source_num_requests": len(requests),
            "window_num_requests": len(windowed),
            "num_requests": len(converted),
            "start_s": start_s,
            "duration_s": duration_s,
            "stride": stride,
            "time_scale": time_scale,
            "effective_source_rate_req_per_s": (len(converted) / (last_ts - first_ts)) if first_ts is not None and last_ts and last_ts > first_ts else None,
            "effective_arrival_rate_req_per_s": (len(converted) / ((last_ts - first_ts) * time_scale)) if first_ts is not None and last_ts and last_ts > first_ts else None,
        },
        "requests": converted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "traces" / "sd3_trace.txt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--id-prefix", type=str, default="sd3_")
    parser.add_argument("--solver", type=str, default="flowmatch_euler")
    args = parser.parse_args()

    requests = load_sd3_trace(args.input)
    trace = convert_requests(
        requests,
        start_s=args.start_s,
        duration_s=args.duration_s,
        stride=args.stride,
        time_scale=args.time_scale,
        max_requests=args.max_requests,
        id_prefix=args.id_prefix,
        solver=args.solver,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    meta = trace["meta"]
    print(f"Wrote {meta['num_requests']} requests -> {args.out}")
    print(
        f"  stride={meta['stride']} time_scale={meta['time_scale']} "
        f"arrival_rate={meta['effective_arrival_rate_req_per_s']:.3f} req/s"
    )


if __name__ == "__main__":
    main()
