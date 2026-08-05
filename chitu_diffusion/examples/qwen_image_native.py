from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from diffusers import QwenImagePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one image with the unmodified Diffusers Qwen-Image pipeline."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN_IMAGE_MODEL_PATH"),
        help="Local checkpoint or Hugging Face model id (or set QWEN_IMAGE_MODEL_PATH).",
    )
    parser.add_argument(
        "--prompt", default="A red cube on a white table, studio lighting"
    )
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
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
        default=Path("outputs/chitu/qwen_image_native.png"),
    )
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or QWEN_IMAGE_MODEL_PATH is required")
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

    pipeline = QwenImagePipeline.from_pretrained(
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
        guidance_scale=None,
        true_cfg_scale=args.true_cfg_scale,
        generator=generator,
        output_type="pil",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {
        "model_path": str(args.model_path),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "seed": args.seed,
        "sha256": digest,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved native Diffusers image to {args.output.resolve()}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
