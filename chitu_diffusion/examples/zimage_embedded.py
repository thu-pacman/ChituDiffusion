from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from chitu_diffusion import (
    EmbeddedDiffusionRuntime,
    EmbeddedRuntimeConfig,
    EPACRequest,
    HotSwitchPoolConfig,
    StageWorldSpec,
    ZImageExecutorFactory,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Z-Image through the non-HTTP EmbeddedDiffusionRuntime."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("ZIMAGE_MODEL_PATH"),
    )
    parser.add_argument("--prompt", default="a red cube on a white table")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--attention-mode",
        choices=("agkv", "usp"),
        default="agkv",
    )
    parser.add_argument("--ulysses-degree", type=int, default=None)
    parser.add_argument(
        "--cfg-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", default="outputs/chitu/embedded.png")
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or ZIMAGE_MODEL_PATH is required")

    world = StageWorldSpec.from_torchrun("zimage_embedded")
    widths = tuple(
        width
        for width in range(1, world.world_size + 1)
        if world.world_size % width == 0
    )
    pool = HotSwitchPoolConfig(
        policy="elastic",
        allowed_lane_widths=widths,
        switch_allowed_until_step=args.steps,
        pulse_steps=5,
        max_inflight_requests=4,
        max_pending_requests=16,
        warmup_resolutions=((args.resolution, args.resolution),),
        warmup_steps=5,
    )
    runtime = EmbeddedDiffusionRuntime.create(
        world=world,
        pool=pool,
        executor_factory=ZImageExecutorFactory(
            model_path=args.model_path,
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ),
            default_width=args.resolution,
            default_height=args.resolution,
            default_num_steps=args.steps,
            attention_mode=args.attention_mode,
            ulysses_degree=args.ulysses_degree,
            cfg_parallel=args.cfg_parallel,
        ),
        config=EmbeddedRuntimeConfig(
            output_root=str(Path(args.output).parent),
        ),
    )
    runtime.start()
    if runtime.is_leader:
        request_id = runtime.submit(
            EPACRequest(
                request_id="embedded-example",
                prompt=args.prompt,
                width=args.resolution,
                height=args.resolution,
                num_steps=args.steps,
            )
        )
        while True:
            completions = runtime.poll(timeout_s=0.1)
            if not completions:
                continue
            completion = next(
                item for item in completions if item.request_id == request_id
            )
            if completion.status != "completed":
                detail = None if completion.error is None else completion.error.message
                raise RuntimeError(f"embedded request failed: {detail}")
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            completion.output.save(output)
            break
        runtime.stop(graceful=True)
    else:
        runtime.wait()
        runtime.stop(graceful=False)


if __name__ == "__main__":
    main()
