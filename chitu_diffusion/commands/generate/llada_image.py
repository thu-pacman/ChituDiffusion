from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from chitu_diffusion import LLaDAImagePipeline, LLaDAImageRequest
from chitu_diffusion.commands.parallel_args import (
    add_parallel_transport_arguments,
    static_parallel_pipeline_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one LLaDA-Image result with Chitu parallelism."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("LLADA_IMAGE_MODEL_PATH"),
        help=("Local Diffusers checkpoint directory (or set LLADA_IMAGE_MODEL_PATH)."),
    )
    parser.add_argument(
        "--generation-mode", choices=("text", "vq", "editing"), default="text"
    )
    parser.add_argument("--prompt", default="a red cube on a white table")
    parser.add_argument("--negative-prompt")
    parser.add_argument(
        "--image",
        type=Path,
        help="Local source image. Required only for --generation-mode editing.",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-mode", choices=("agkv", "ulysses"), default="agkv")
    parser.add_argument("--ulysses-degree", type=int)
    add_parallel_transport_arguments(parser)
    parser.add_argument(
        "--cfg-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer CFP-2 before context parallelism on eligible lanes.",
    )
    parser.add_argument(
        "--parallel-vae",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to approximate tiled VAE decode. AutoencoderKLFlux2 "
            "contains global normalization and attention, so tiles are not exact."
        ),
    )
    parser.add_argument("--vae-parallel-halo", type=int, default=8)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/chitu/llada_image.png"),
    )
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or LLADA_IMAGE_MODEL_PATH is required")
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_dir():
        parser.error(f"local checkpoint directory does not exist: {model_path}")
    args.model_path = str(model_path)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.generation_mode == "editing" and args.image is None:
        parser.error("--image is required for --generation-mode editing")
    if args.generation_mode != "editing" and args.image is not None:
        parser.error("--image is only valid for --generation-mode editing")
    if args.image is not None and not args.image.is_file():
        parser.error(f"source image does not exist: {args.image}")
    if args.vae_parallel_halo < 0:
        parser.error("--vae-parallel-halo must be non-negative")
    return args


def _load_source_image(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    with Image.open(path) as image:
        return image.convert("RGB")


def main() -> None:
    args = parse_args()
    source_image = _load_source_image(args.image)
    pipeline = LLaDAImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, args.dtype),
        local_files_only=args.local_files_only,
        attention_mode=args.attention_mode,
        ulysses_degree=args.ulysses_degree,
        cfg_parallel=args.cfg_parallel,
        parallel_vae=args.parallel_vae,
        vae_parallel_halo=args.vae_parallel_halo,
        **static_parallel_pipeline_kwargs(args),
    )
    try:
        output = pipeline.generate(
            LLaDAImageRequest(
                prompt=args.prompt,
                generation_mode=args.generation_mode,
                image=source_image,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                max_sequence_length=args.max_sequence_length,
            )
        )
        if pipeline.parallel_context.rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output.images[0].save(args.output)
            print(f"Saved LLaDA-Image output to {args.output.resolve()}")
        if dist.is_initialized():
            dist.barrier()
    finally:
        pipeline.close()
        if source_image is not None:
            source_image.close()


if __name__ == "__main__":
    main()
