"""Validate real-checkpoint LLaDA-Image modes through the public facade."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.distributed as dist
import transformers
from PIL import Image

from chitu_diffusion import LLaDAImagePipeline, LLaDAImageRequest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--modes", nargs="+", choices=("text", "vq", "editing"), default=["text"])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A red ceramic teapot on a white table, studio photograph")
    parser.add_argument("--attention-mode", choices=("agkv", "ulysses"), default="agkv")
    parser.add_argument("--cfg-parallel", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("real-checkpoint validation requires CUDA")
    if "editing" in args.modes and args.image is None:
        parser.error("editing requires --image so all topologies use the same source")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    cp_degree = world // 2 if args.cfg_parallel and world % 2 == 0 else world
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pipeline = LLaDAImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attention_mode=args.attention_mode,
        ulysses_degree=cp_degree if args.attention_mode == "ulysses" else None,
        cfg_parallel=args.cfg_parallel,
        allowed_lane_widths=tuple(sorted({1, world})),
        agkv_transport="torch",
        ulysses_transport="torch",
    )
    torch.cuda.synchronize()
    report = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "world_size": world,
        "attention_mode": args.attention_mode,
        "cfg_parallel": args.cfg_parallel,
        "load_seconds": time.perf_counter() - started,
        "cases": [],
    }
    try:
        for mode in args.modes:
            source = None
            if mode == "editing":
                with Image.open(args.image) as image:
                    source = image.convert("RGB")
            request = LLaDAImageRequest(
                prompt="Change the teapot to blue, preserve the composition" if mode == "editing" else args.prompt,
                generation_mode=mode,
                image=source,
                width=args.resolution,
                height=args.resolution,
                num_inference_steps=args.steps,
                guidance_scale=4.5,
                seed=args.seed,
                output_type="pt",
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = pipeline.generate(request)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if source is not None:
                source.close()
            if pipeline.parallel_context.rank == 0:
                pixels = output.images.cpu().float()
                if not torch.isfinite(pixels).all() or pixels.std() < 0.01:
                    raise RuntimeError(f"{mode} produced non-finite or flat pixels")
                tensor_path = args.output / f"{mode}.pt"
                torch.save(pixels, tensor_path)
                image = pipeline.diffusers_pipeline.image_processor.numpy_to_pil(
                    pixels.permute(0, 2, 3, 1).numpy()
                )[0]
                image.save(args.output / f"{mode}.png")
                row = {
                    "mode": mode,
                    "resolution": args.resolution,
                    "steps": args.steps,
                    "seed": args.seed,
                    "seconds": elapsed,
                    "pixel_std": pixels.std().item(),
                    "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                if args.reference is not None:
                    reference = torch.load(args.reference / f"{mode}.pt", weights_only=True)
                    error = (pixels - reference).square().mean().item()
                    row["reference_rmse"] = error**0.5
                    row["reference_psnr"] = -10 * np.log10(max(error, 1e-20))
                report["cases"].append(row)
                (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
            if dist.is_initialized():
                dist.barrier()
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
