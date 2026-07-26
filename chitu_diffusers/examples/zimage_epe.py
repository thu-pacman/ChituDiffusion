from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from chitu_diffusers import EPACPipeline, EPACRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Z-Image result with EPAC static full parallelism."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("ZIMAGE_MODEL_PATH"),
        help="Local checkpoint or Hugging Face model id (or set ZIMAGE_MODEL_PATH).",
    )
    parser.add_argument("--prompt", default="a red cube on a white table")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--attention-mode",
        choices=("agkv", "usp"),
        default="agkv",
    )
    parser.add_argument("--ulysses-degree", type=int, default=2)
    parser.add_argument(
        "--cfg-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer CFP-2 before context parallelism on eligible lanes.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/chitu-diffusers/zimage_epe.png"),
    )
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or ZIMAGE_MODEL_PATH is required")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    pipeline = EPACPipeline.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, args.dtype),
        local_files_only=args.local_files_only,
        attention_mode=args.attention_mode,
        ulysses_degree=args.ulysses_degree,
        cfg_parallel=args.cfg_parallel,
    )
    try:
        output = pipeline.generate(
            EPACRequest(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )
        )
        if pipeline.parallel_context.rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output.images[0].save(args.output)
            print(f"Saved EPAC static image to {args.output.resolve()}")
        if dist.is_initialized():
            dist.barrier()
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
