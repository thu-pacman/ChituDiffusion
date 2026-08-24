from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from chitu_diffusion import Flux1Pipeline, Flux1Request
from chitu_diffusion.commands.cache_args import (
    add_cache_arguments,
    cache_config_from_args,
)
from chitu_diffusion.commands.parallel_args import (
    add_parallel_transport_arguments,
    static_parallel_pipeline_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX.1-dev through EPE")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt", default="A red cube on a white table, studio lighting"
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attention-mode", choices=("agkv", "ulysses"), default="agkv")
    parser.add_argument("--ulysses-degree", type=int)
    add_parallel_transport_arguments(parser)
    add_cache_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = cache_config_from_args(args)
    cache.validate_steps(args.steps)
    if not torch.cuda.is_available():
        raise RuntimeError("the Flux.1 EPE example requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    started = time.perf_counter()
    pipeline = Flux1Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attention_mode=args.attention_mode,
        ulysses_degree=args.ulysses_degree,
        **static_parallel_pipeline_kwargs(args),
    )
    loaded_at = time.perf_counter()
    try:
        result = pipeline.generate(
            Flux1Request(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                cache=cache,
            )
        )
        generated_at = time.perf_counter()
        output_path = Path(args.output)
        if pipeline.parallel_context.rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(output_path)
            metadata = {
                "model_path": str(Path(args.model_path).resolve()),
                "output": str(output_path.resolve()),
                "prompt": args.prompt,
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "attention_mode": args.attention_mode,
                "ulysses_degree": args.ulysses_degree,
                "ulysses_transport": args.ulysses_transport,
                "agkv_transport": args.agkv_transport,
                "world_size": pipeline.parallel_context.world_size,
                "load_seconds": loaded_at - started,
                "generate_seconds": generated_at - loaded_at,
                "cache": pipeline.last_cache_stats,
            }
            output_path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(metadata, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
