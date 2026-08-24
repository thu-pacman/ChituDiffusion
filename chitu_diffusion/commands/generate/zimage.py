from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from chitu_diffusion import ZImagePipeline, ZImageRequest
from chitu_diffusion.commands.cache_args import (
    add_cache_arguments,
    cache_config_from_args,
)
from chitu_diffusion.commands.parallel_args import (
    add_parallel_transport_arguments,
    static_parallel_pipeline_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Z-Image result with EPE static full parallelism."
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
        choices=("agkv", "ulysses"),
        default="agkv",
    )
    parser.add_argument("--ulysses-degree", type=int, default=None)
    add_parallel_transport_arguments(parser)
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
    add_cache_arguments(parser)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/chitu/zimage_epe.png"),
    )
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or ZIMAGE_MODEL_PATH is required")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    cache = cache_config_from_args(args)
    cache.validate_steps(args.steps)
    pipeline = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, args.dtype),
        local_files_only=args.local_files_only,
        attention_mode=args.attention_mode,
        ulysses_degree=args.ulysses_degree,
        cfg_parallel=args.cfg_parallel,
        **static_parallel_pipeline_kwargs(args),
    )
    try:
        output = pipeline.generate(
            ZImageRequest(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                cache=cache,
            )
        )
        if pipeline.parallel_context.rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output.images[0].save(args.output)
            print(f"Saved EPE static image to {args.output.resolve()}")
            if pipeline.last_cache_stats is not None:
                print(f"FlexCache stats: {pipeline.last_cache_stats}")
        if dist.is_initialized():
            dist.barrier()
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
