from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from chitu_diffusers import (
    EmbeddedDiffusionRuntime,
    EmbeddedRuntimeConfig,
    HotSwitchPoolConfig,
    StageWorldSpec,
    WanExecutorFactory,
    WanRequest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Wan2.1 T2V in the embedded elastic EPAC runtime."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", default="A cat walking on grass.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--secondary-steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy",
        choices=("elastic", "static_cp", "static_dp"),
        default="elastic",
    )
    parser.add_argument("--no-cfg-parallel", action="store_true")
    parser.add_argument("--no-parallel-vae", action="store_true")
    parser.add_argument("--vae-parallel-halo", type=int, default=8)
    parser.add_argument("--no-balanced-k", action="store_true")
    parser.add_argument("--record-timeline", action="store_true")
    args = parser.parse_args()
    if args.warmup_steps < 3:
        parser.error("--warmup-steps must be at least 3")
    if args.vae_parallel_halo < 0:
        parser.error("--vae-parallel-halo must be non-negative")

    world = StageWorldSpec.from_torchrun("wan_embedded")
    widths = tuple(
        width
        for width in range(1, world.world_size + 1)
        if world.world_size % width == 0
    )
    pool = HotSwitchPoolConfig(
        policy=args.policy,
        allowed_lane_widths=widths,
        switch_allowed_until_step=max(args.steps, args.secondary_steps),
        pulse_steps=1,
        balanced_k=not args.no_balanced_k,
        max_inflight_requests=4,
        max_pending_requests=16,
        warmup_resolutions=((args.height, args.width),),
        warmup_steps=args.warmup_steps,
    )
    runtime = EmbeddedDiffusionRuntime.create(
        world=world,
        pool=pool,
        executor_factory=WanExecutorFactory(
            model_path=args.model_path,
            torch_dtype=torch.bfloat16,
            default_width=args.width,
            default_height=args.height,
            default_num_frames=args.frames,
            default_num_steps=args.steps,
            cfg_parallel=not args.no_cfg_parallel,
            flow_shift=args.flow_shift,
            parallel_vae=not args.no_parallel_vae,
            vae_parallel_halo=args.vae_parallel_halo,
        ),
        config=EmbeddedRuntimeConfig(
            output_root=str(args.output.parent),
            record_timeline=args.record_timeline,
        ),
    )
    runtime.start()
    graceful = False
    try:
        if runtime.is_leader:
            specs = [("wan-primary", args.steps, args.seed, args.output)]
            if args.secondary_steps > 0:
                specs.append(
                    (
                        "wan-secondary",
                        args.secondary_steps,
                        args.seed + 1,
                        args.output.with_name(
                            f"{args.output.stem}-secondary{args.output.suffix}"
                        ),
                    )
                )
            destinations: dict[str, Path] = {}
            for request_id, request_steps, seed, output in specs:
                destinations[
                    runtime.submit(
                        WanRequest(
                            request_id=request_id,
                            prompt=args.prompt,
                            negative_prompt=args.negative_prompt,
                            width=args.width,
                            height=args.height,
                            num_frames=args.frames,
                            num_steps=request_steps,
                            guidance_scale=args.guidance_scale,
                            seed=seed,
                        )
                    )
                ] = output

            results = []
            while destinations:
                for completion in runtime.poll(timeout_s=0.1):
                    output = destinations.pop(completion.request_id, None)
                    if output is None:
                        continue
                    if completion.status != "completed":
                        detail = (
                            None
                            if completion.error is None
                            else completion.error.message
                        )
                        raise RuntimeError(f"embedded Wan request failed: {detail}")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    export_to_video(completion.output, str(output), fps=16)
                    digest = hashlib.sha256(output.read_bytes()).hexdigest()
                    results.append(
                        {
                            "request_id": completion.request_id,
                            "output": str(output.resolve()),
                            "sha256": digest,
                            "submitted_at": completion.submitted_at,
                            "started_at": completion.started_at,
                            "completed_at": completion.completed_at,
                        }
                    )
                    print(f"Saved embedded Wan output to {output.resolve()}")
                    print(f"sha256={digest}")
            args.output.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "world_size": world.world_size,
                        "policy": args.policy,
                        "steps": args.steps,
                        "secondary_steps": args.secondary_steps,
                        "frames": args.frames,
                        "parallel_vae": not args.no_parallel_vae,
                        "vae_parallel_halo": args.vae_parallel_halo,
                        "results": results,
                        "warmup": runtime.warmup_report,
                    },
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            graceful = True
        else:
            runtime.wait()
    finally:
        runtime.stop(graceful=graceful)


if __name__ == "__main__":
    main()
