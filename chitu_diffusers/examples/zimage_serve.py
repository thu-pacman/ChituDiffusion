from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from chitu_diffusers import EPACPipeline, EPACServeConfig


def _resolution(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(" ", "")
    if "x" not in normalized:
        size = int(normalized)
        return size, size
    height, width = normalized.split("x", 1)
    return int(height), int(width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm up Z-Image lanes and start the EPAC elastic service."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("ZIMAGE_MODEL_PATH"),
        help="Local checkpoint or Hugging Face model id (or set ZIMAGE_MODEL_PATH).",
    )
    parser.add_argument("--postprocess-workers", type=int, default=4)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18200)
    parser.add_argument("--advertise-host")
    parser.add_argument(
        "--warmup-resolutions",
        type=_resolution,
        nargs="+",
        default=((512, 512), (1024, 1024), (2048, 2048)),
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--pulse-steps", type=int, default=5)
    parser.add_argument(
        "--attention-mode",
        choices=("agkv", "usp"),
        default="agkv",
    )
    parser.add_argument("--ulysses-degree", type=int, default=2)
    parser.add_argument(
        "--parallel-vae",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the active EPAC lane for spatially tiled VAE decode.",
    )
    parser.add_argument("--vae-parallel-halo", type=int, default=8)
    parser.add_argument(
        "--default-deadline-ms",
        type=float,
        help=(
            "Default end-to-end request SLO in milliseconds; request deadline_ms "
            "overrides it."
        ),
    )
    parser.add_argument(
        "--deadline-guard-ms",
        type=float,
        default=3000.0,
        help="Reserve non-DiT runtime before each request deadline.",
    )
    parser.add_argument(
        "--schedule-strategy",
        choices=("elastic", "static_cp", "static_dp"),
        default="elastic",
    )
    parser.add_argument("--default-steps", type=int, default=50)
    parser.add_argument("--max-inflight-requests", type=int, default=4)
    parser.add_argument("--max-pending-requests", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-root", default="outputs/chitu-api")
    parser.add_argument(
        "--record-timeline",
        action="store_true",
        help="Buffer rank-local execution spans and flush them at shutdown.",
    )
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or ZIMAGE_MODEL_PATH is required")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pid_path = output_root / f"worker-rank{rank}.pid"
    pid_path.write_text(str(os.getpid()), encoding="ascii")
    pipeline = None
    try:
        pipeline = EPACPipeline.from_pretrained(
            args.model_path,
            torch_dtype=getattr(torch, args.dtype),
            local_files_only=args.local_files_only,
            attention_mode=args.attention_mode,
            ulysses_degree=args.ulysses_degree,
        )
        pipeline.serve(
            EPACServeConfig(
                host=args.host,
                port=args.port,
                advertise_host=args.advertise_host,
                warmup_resolutions=tuple(args.warmup_resolutions),
                warmup_steps=args.warmup_steps,
                pulse_steps=args.pulse_steps,
                default_deadline_ms=args.default_deadline_ms,
                deadline_guard_ms=args.deadline_guard_ms,
                schedule_strategy=args.schedule_strategy,
                default_num_steps=args.default_steps,
                max_inflight_requests=args.max_inflight_requests,
                max_pending_requests=args.max_pending_requests,
                output_root=args.output_root,
                record_timeline=args.record_timeline,
                postprocess_workers=args.postprocess_workers,
                parallel_vae=args.parallel_vae,
                vae_parallel_halo=args.vae_parallel_halo,
            )
        )
    finally:
        if pipeline is not None:
            pipeline.close()
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
