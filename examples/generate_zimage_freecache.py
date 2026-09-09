"""Generate a 50-step Z-Image preview with any integer Fresh budget 1..50."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chitu_diffusion import CacheConfig, FreeCacheConfig  # noqa: E402


def make_cache(budget: int = 25) -> CacheConfig:
    return CacheConfig(
        strategy="freecache", params=FreeCacheConfig.preview("zimage", budget)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument(
        "--prompt",
        default="A porcelain coffee cup on a cafe table, soft morning light.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/freecache-preview.png")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without loading model weights",
    )
    args = parser.parse_args()
    cache = make_cache(args.budget)
    cache.validate_steps(50)
    print(json.dumps(cache.to_dict(), indent=2))
    if args.dry_run:
        return
    if not args.model_path:
        parser.error("Supply --model-path")
    if args.output.exists():
        parser.error("Output exists; choose another path")
    import torch

    from chitu_diffusion import ZImagePipeline, ZImageRequest

    pipeline = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        allowed_lane_widths=(1,),
        local_files_only=True,
    )
    try:
        result = pipeline.generate(
            ZImageRequest(
                prompt=args.prompt,
                width=1024,
                height=1024,
                num_steps=50,
                guidance_scale=5.0,
                seed=args.seed,
                cache=cache,
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(args.output)
        print(pipeline.last_cache_stats)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
