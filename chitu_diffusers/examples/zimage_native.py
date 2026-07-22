from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import ZImagePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one image with the unmodified Diffusers Z-Image pipeline."
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/chitu-diffusers/zimage_native.png"),
    )
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or ZIMAGE_MODEL_PATH is required")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; pass --device cpu explicitly")
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU execution requires --dtype float32")

    pipeline = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    pipeline.set_progress_bar_config(disable=False)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="pil",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(args.output)
    print(f"Saved native Diffusers image to {args.output.resolve()}")


if __name__ == "__main__":
    main()
