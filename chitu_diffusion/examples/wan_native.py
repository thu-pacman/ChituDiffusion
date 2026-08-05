from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from chitu_diffusers.models.wan.pipeline import EpeWanPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Wan2.1 T2V through the official Diffusers pipeline loop."
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
    args = parser.parse_args()

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("wan_native.py requires one process")
    pipeline = EpeWanPipeline.from_pretrained(
        args.model_path,
        allowed_lane_widths=(1,),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        flow_shift=args.flow_shift,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    started = time.perf_counter()
    try:
        output = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            output_type="np",
        ).frames[0]
        elapsed_s = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(output, str(args.output), fps=16)
        args.output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "implementation": "official_diffusers_loop",
                    "steps": args.steps,
                    "frames": args.frames,
                    "elapsed_s": elapsed_s,
                    "seed": args.seed,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved Wan Diffusers output to {args.output.resolve()}")
        print(f"elapsed_s={elapsed_s:.3f}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
