from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the native Diffusers Flux.1 baseline"
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise ValueError("the native Flux.1 baseline requires exactly one rank")
    if not torch.cuda.is_available():
        raise RuntimeError("the native Flux.1 baseline requires a CUDA GPU")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    started = time.perf_counter()
    pipeline = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    loaded_at = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipeline(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    generated_at = time.perf_counter()

    output_path = Path(args.output)
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
        "load_seconds": loaded_at - started,
        "generate_seconds": generated_at - loaded_at,
        "torch_version": torch.__version__,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
