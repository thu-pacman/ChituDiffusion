from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from chitu_diffusers import QwenImageEPACPipeline, QwenImageRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Qwen-Image result with EPAC full-world parallelism."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prompt", default="A red cube on a white table, studio lighting"
    )
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--cfg-parallel", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the Qwen-Image EPAC example requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    pipeline = QwenImageEPACPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attention_mode="agkv",
        cfg_parallel=args.cfg_parallel,
    )
    try:
        result = pipeline.generate(
            QwenImageRequest(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_steps=args.steps,
                true_cfg_scale=args.true_cfg_scale,
                seed=args.seed,
            )
        )
        if pipeline.parallel_context.rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.images[0].save(args.output)
            digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
            args.output.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "world_size": pipeline.parallel_context.world_size,
                        "steps": args.steps,
                        "true_cfg_scale": args.true_cfg_scale,
                        "cfg_parallel": args.cfg_parallel,
                        "seed": args.seed,
                        "sha256": digest,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Saved Qwen-Image EPAC output to {args.output.resolve()}")
            print(f"sha256={digest}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
