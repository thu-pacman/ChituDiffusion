from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from chitu_diffusion.models.flux1 import EpeFlux1Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dynamic FLUX.1 EPAC lanes")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lane-widths", default="1,4,2")
    parser.add_argument(
        "--prompt", default="A red cube on a white table, studio lighting"
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attention-mode", choices=("agkv", "usp"), default="agkv")
    parser.add_argument("--ulysses-degree", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lane_widths = tuple(int(value) for value in args.lane_widths.split(","))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if any(width < 1 or world_size % width for width in lane_widths):
        raise ValueError("every lane width must be a positive divisor of world size")
    torch.cuda.set_device(local_rank)
    started = time.perf_counter()
    pipeline = EpeFlux1Pipeline.from_pretrained(
        args.model_path,
        allowed_lane_widths=tuple(sorted(set((1, world_size, *lane_widths)))),
        attention_mode=args.attention_mode,
        ulysses_degree=args.ulysses_degree,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(torch.device("cuda", local_rank))
    pipeline.set_progress_bar_config(disable=True)
    loaded_at = time.perf_counter()
    try:
        state = pipeline.prepare_request(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=len(lane_widths),
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(
                args.seed
            ),
        )
        step_max_rank_diffs = []
        for lane_width in lane_widths:
            offset = (rank // lane_width) * lane_width
            lane = tuple(range(offset, offset + lane_width))
            pipeline.denoise_step(state, lane_ranks=lane)
            reference = state.latents.clone()
            dist.broadcast(reference, src=0)
            local_max = (state.latents - reference).abs().max()
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
            step_max_rank_diffs.append(float(local_max.item()))
        generated_at = time.perf_counter()
        if rank == 0:
            result = pipeline.finalize_request(state)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(output_path)
            metadata = {
                "output": str(output_path.resolve()),
                "lane_widths": list(lane_widths),
                "attention_mode": args.attention_mode,
                "ulysses_degree": args.ulysses_degree,
                "step_max_rank_diffs": step_max_rank_diffs,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "load_seconds": loaded_at - started,
                "generate_seconds": generated_at - loaded_at,
            }
            output_path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(metadata, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
