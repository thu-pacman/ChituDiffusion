from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chitu_diffusers import (
    EmbeddedDiffusionRuntime,
    EmbeddedRuntimeConfig,
    Flux1ExecutorFactory,
    Flux1Request,
    HotSwitchPoolConfig,
    StageWorldSpec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FLUX.1-dev in the embedded EPAC runtime"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt", default="A red cube on a white table, studio lighting"
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attention-mode", choices=("agkv", "usp"), default="agkv")
    parser.add_argument("--ulysses-degree", type=int)
    parser.add_argument("--no-parallel-vae", action="store_true")
    parser.add_argument("--vae-parallel-halo", type=int, default=8)
    args = parser.parse_args()
    if args.vae_parallel_halo < 0:
        parser.error("--vae-parallel-halo must be non-negative")

    world = StageWorldSpec.from_torchrun("flux1_embedded")
    widths = tuple(
        width
        for width in range(1, world.world_size + 1)
        if world.world_size % width == 0
    )
    pool = HotSwitchPoolConfig(
        policy="elastic",
        allowed_lane_widths=widths,
        switch_allowed_until_step=args.steps,
        pulse_steps=1,
        max_inflight_requests=4,
        max_pending_requests=16,
        warmup_resolutions=((args.resolution, args.resolution),),
        warmup_steps=args.warmup_steps,
    )
    runtime = EmbeddedDiffusionRuntime.create(
        world=world,
        pool=pool,
        executor_factory=Flux1ExecutorFactory(
            model_path=args.model_path,
            torch_dtype=torch.bfloat16,
            default_width=args.resolution,
            default_height=args.resolution,
            default_num_steps=args.steps,
            attention_mode=args.attention_mode,
            ulysses_degree=args.ulysses_degree,
            parallel_vae=not args.no_parallel_vae,
            vae_parallel_halo=args.vae_parallel_halo,
        ),
        config=EmbeddedRuntimeConfig(output_root=str(Path(args.output).parent)),
    )
    runtime.start()
    graceful = False
    try:
        if runtime.is_leader:
            request_id = runtime.submit(
                Flux1Request(
                    request_id="flux1-embedded-example",
                    prompt=args.prompt,
                    width=args.resolution,
                    height=args.resolution,
                    num_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
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
                    detail = (
                        None if completion.error is None else completion.error.message
                    )
                    raise RuntimeError(f"embedded Flux.1 request failed: {detail}")
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                completion.output.save(output)
                graceful = True
                break
        else:
            runtime.wait()
    finally:
        runtime.stop(graceful=graceful)


if __name__ == "__main__":
    main()
