from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .pipeline import EpeZImagePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone EPE Z-Image pipeline smoke")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="a red cube on a white table")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    widths = tuple(width for width in range(1, world_size + 1) if world_size % width == 0)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipeline = EpeZImagePipeline.from_pretrained(
        args.model_path,
        allowed_lane_widths=widths,
        torch_dtype=dtype,
        local_files_only=True,
    )
    device = (
        torch.device("cuda", pipeline.parallel_context.local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipeline(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        generator=generator,
        output_type="pil",
    )
    if pipeline.parallel_context.rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output)
        print(f"EPE_ZIMAGE_SMOKE_OK output={output} size={result.images[0].size}")
    if dist.is_initialized():
        pipeline.parallel_context.barrier()
    pipeline.close()


if __name__ == "__main__":
    main()
