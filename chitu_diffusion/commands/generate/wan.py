from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video

from chitu_diffusion import FppConfig, WanPipeline, WanRequest
from chitu_diffusion.commands.cache_args import (
    add_cache_arguments,
    cache_config_from_args,
)
from chitu_diffusion.commands.generate.common import (
    generation_timestamp,
    write_generation_metadata,
)
from chitu_diffusion.commands.parallel_args import (
    add_parallel_transport_arguments,
    add_vae_parallel_arguments,
    static_parallel_pipeline_kwargs,
    validate_vae_parallel_arguments,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one Wan2.1 T2V result with EPE full-world parallelism."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save lossless float frames beside the video for paired quality evaluation.",
    )
    parser.add_argument("--prompt", default="A cat walking on grass.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-mode", choices=("agkv", "ulysses"), default="agkv")
    parser.add_argument("--ulysses-degree", type=int)
    parser.add_argument("--pipeline-parallel-degree", type=int, default=1)
    parser.add_argument("--context-parallel-degree", type=int, default=1)
    parser.add_argument("--sample-solver", choices=("euler", "unipc"))
    parser.add_argument("--fpp-schedule", choices=("stream", "step"), default="stream")
    parser.add_argument("--fpp-reference-stages", type=int)
    parser.add_argument("--fpp-reference-context-degree", type=int, default=1)
    parser.add_argument("--fpp-patches", type=int, default=4)
    parser.add_argument("--fpp-warmup-steps", type=int, default=1)
    parser.add_argument("--fpp-cooldown-steps", type=int, default=1)
    parser.add_argument("--fpp-refresh-interval", type=int, default=50)
    parser.add_argument(
        "--fpp-rotate-patches", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--tensor-parallel-degree",
        type=int,
        default=1,
        help=(
            "Shard the transformer across this many ranks. Every Wan variant "
            "accepts 2, 4, and 8; the feed-forward width is what rules out the "
            "odd degrees the head count would otherwise allow."
        ),
    )
    add_parallel_transport_arguments(parser)
    parser.add_argument(
        "--cfg-parallel", action=argparse.BooleanOptionalAction, default=None
    )
    add_vae_parallel_arguments(parser)
    add_cache_arguments(parser)
    args = parser.parse_args()
    validate_vae_parallel_arguments(parser, args)
    cache = cache_config_from_args(args)
    cache.validate_steps(args.steps)
    fpp_enabled = (
        args.pipeline_parallel_degree > 1
        or args.context_parallel_degree > 1
        or args.fpp_reference_stages is not None
        or args.fpp_reference_context_degree != 1
    )
    if args.cfg_parallel is None:
        args.cfg_parallel = not fpp_enabled
    if fpp_enabled and cache.strategy != "none":
        parser.error("FPP cannot be combined with FlexCache strategies")
    fpp_config = (
        FppConfig(
            patches=args.fpp_patches,
            schedule=args.fpp_schedule,
            reference_stages=args.fpp_reference_stages,
            reference_context_degree=args.fpp_reference_context_degree,
            warmup_steps=args.fpp_warmup_steps,
            cooldown_steps=args.fpp_cooldown_steps,
            refresh_interval=args.fpp_refresh_interval,
            rotate_patches=args.fpp_rotate_patches,
        )
        if fpp_enabled
        else None
    )

    pipeline = WanPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attention_mode=args.attention_mode,
        ulysses_degree=args.ulysses_degree,
        tensor_parallel_degree=args.tensor_parallel_degree,
        pipeline_parallel_degree=args.pipeline_parallel_degree,
        fpp_config=fpp_config,
        context_parallel_degree=args.context_parallel_degree,
        sample_solver=args.sample_solver
        or ("unipc" if fpp_enabled and args.fpp_schedule == "stream" else "euler"),
        cfg_parallel=args.cfg_parallel,
        parallel_vae=args.parallel_vae,
        vae_parallel_halo=args.vae_parallel_halo,
        flow_shift=args.flow_shift,
        **static_parallel_pipeline_kwargs(args),
    )
    try:
        started = generation_timestamp()
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
                cache=cache,
            )
        )
        elapsed_s = generation_timestamp() - started
        if pipeline.parallel_context.rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.save_frames:
                np.savez_compressed(
                    args.output.with_suffix(".npz"), frames=result.frames[0]
                )
            export_to_video(result.frames[0], str(args.output), fps=16)
            write_generation_metadata(
                args.output,
                args=args,
                pipeline=pipeline,
                elapsed_s=elapsed_s,
                metadata={
                    "implementation": "fpp" if fpp_enabled else "epe",
                    "fpp": pipeline.last_fpp_stats,
                    "fpp_config": vars(fpp_config) if fpp_config else None,
                    "sample_solver": args.sample_solver
                    or (
                        "unipc"
                        if fpp_enabled and args.fpp_schedule == "stream"
                        else "euler"
                    ),
                },
            )
            print(f"Saved Wan output to {args.output.resolve()}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
