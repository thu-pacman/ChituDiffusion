from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from chitu_diffusion import QwenImageEPACPipeline, QwenImageRequest
from chitu_diffusion.examples.cache_args import (
    add_cache_arguments,
    cache_config_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Qwen-Image result with EPAC full-world parallelism."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prompt", default="A red cube on a white table, studio lighting"
    )
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attention-mode", choices=("agkv", "usp"), default="agkv")
    parser.add_argument("--ulysses-degree", type=int)
    parser.add_argument(
        "--cfg-parallel", action=argparse.BooleanOptionalAction, default=True
    )
    add_cache_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = cache_config_from_args(args)
    cache.validate_steps(args.steps)
    if not torch.cuda.is_available():
        raise RuntimeError("the Qwen-Image EPAC example requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    pipeline = QwenImageEPACPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attention_mode=args.attention_mode,
        ulysses_degree=args.ulysses_degree,
        cfg_parallel=args.cfg_parallel,
    )
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipeline.generate(
            QwenImageRequest(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_steps=args.steps,
                true_cfg_scale=args.true_cfg_scale,
                seed=args.seed,
                cache=cache,
            )
        )
        torch.cuda.synchronize()
        generate_seconds = time.perf_counter() - started
        if pipeline.parallel_context.rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(args.output)
            digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
            args.output.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "world_size": pipeline.parallel_context.world_size,
                        "steps": args.steps,
                        "true_cfg_scale": args.true_cfg_scale,
                        "cfg_parallel": args.cfg_parallel,
                        "attention_mode": args.attention_mode,
                        "ulysses_degree": args.ulysses_degree,
                        "seed": args.seed,
                        "generate_seconds": generate_seconds,
                        "sha256": digest,
                        "cache": pipeline.last_cache_stats,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Saved Qwen-Image EPAC output to {args.output.resolve()}")
            print(f"sha256={digest}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
