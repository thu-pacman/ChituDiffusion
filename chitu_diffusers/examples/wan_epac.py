from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from chitu_diffusers import WanEPACPipeline, WanRequest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one Wan2.1 T2V result with EPAC full-world parallelism."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", default="A cat walking on grass.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cfg-parallel", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--no-parallel-vae", action="store_true")
    parser.add_argument("--vae-parallel-halo", type=int, default=8)
    args = parser.parse_args()
    if args.vae_parallel_halo < 0:
        parser.error("--vae-parallel-halo must be non-negative")

    pipeline = WanEPACPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attention_mode="agkv",
        cfg_parallel=args.cfg_parallel,
        parallel_vae=not args.no_parallel_vae,
        vae_parallel_halo=args.vae_parallel_halo,
        flow_shift=args.flow_shift,
    )
    started = time.perf_counter()
    try:
        result = pipeline.generate(
            WanRequest(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                num_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                output_type="np",
            )
        )
        elapsed_s = time.perf_counter() - started
        if pipeline.parallel_context.rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(result.frames[0], str(args.output), fps=16)
            args.output.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "implementation": "epac",
                        "world_size": pipeline.parallel_context.world_size,
                        "steps": args.steps,
                        "frames": args.frames,
                        "cfg_parallel": args.cfg_parallel,
                        "parallel_vae": not args.no_parallel_vae,
                        "vae_parallel_halo": args.vae_parallel_halo,
                        "elapsed_s": elapsed_s,
                        "seed": args.seed,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Saved Wan EPAC output to {args.output.resolve()}")
            print(f"elapsed_s={elapsed_s:.3f}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
