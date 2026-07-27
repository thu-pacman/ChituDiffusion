from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chitu_diffusers import (
    EmbeddedDiffusionRuntime,
    EmbeddedRuntimeConfig,
    HotSwitchPoolConfig,
    QwenImageExecutorFactory,
    QwenImageRequest,
    StageWorldSpec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen-Image in the embedded EPAC runtime"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt", default="A red cube on a white table, studio lighting"
    )
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--secondary-steps",
        type=int,
        default=0,
        help="Submit a second concurrent request when positive.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--policy", choices=("elastic", "static_cp", "static_dp"), default="elastic"
    )
    parser.add_argument("--no-cfg-parallel", action="store_true")
    parser.add_argument("--no-parallel-vae", action="store_true")
    parser.add_argument("--record-timeline", action="store_true")
    parser.add_argument("--vae-parallel-halo", type=int, default=8)
    args = parser.parse_args()
    if args.warmup_steps < 3:
        parser.error("--warmup-steps must be at least 3")

    world = StageWorldSpec.from_torchrun("qwen_image_embedded")
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
        max_inflight_requests=4,
        max_pending_requests=16,
        warmup_resolutions=((args.resolution, args.resolution),),
        warmup_steps=args.warmup_steps,
    )
    output = Path(args.output)
    runtime = EmbeddedDiffusionRuntime.create(
        world=world,
        pool=pool,
        executor_factory=QwenImageExecutorFactory(
            model_path=args.model_path,
            torch_dtype=torch.bfloat16,
            default_width=args.resolution,
            default_height=args.resolution,
            default_num_steps=args.steps,
            cfg_parallel=not args.no_cfg_parallel,
            parallel_vae=not args.no_parallel_vae,
            vae_parallel_halo=args.vae_parallel_halo,
        ),
        config=EmbeddedRuntimeConfig(
            output_root=str(output.parent),
            record_timeline=args.record_timeline,
        ),
    )
    runtime.start()
    graceful = False
    try:
        if runtime.is_leader:
            request_specs = [
                ("qwen-image-primary", args.steps, args.seed, output),
            ]
            if args.secondary_steps > 0:
                secondary_output = output.with_name(
                    f"{output.stem}-secondary{output.suffix}"
                )
                request_specs.append(
                    (
                        "qwen-image-secondary",
                        args.secondary_steps,
                        args.seed + 1,
                        secondary_output,
                    )
                )
            request_outputs = {}
            for request_id, request_steps, seed, request_output in request_specs:
                submitted_id = runtime.submit(
                    QwenImageRequest(
                        request_id=request_id,
                        prompt=args.prompt,
                        negative_prompt=args.negative_prompt,
                        width=args.resolution,
                        height=args.resolution,
                        num_steps=request_steps,
                        true_cfg_scale=args.true_cfg_scale,
                        seed=seed,
                    )
                )
                request_outputs[submitted_id] = request_output
            results = []
            while request_outputs:
                completions = runtime.poll(timeout_s=0.1)
                if not completions:
                    continue
                for completion in completions:
                    request_output = request_outputs.pop(completion.request_id, None)
                    if request_output is None:
                        continue
                    if completion.status != "completed":
                        detail = (
                            None
                            if completion.error is None
                            else completion.error.message
                        )
                        raise RuntimeError(
                            f"embedded Qwen-Image request failed: {detail}"
                        )
                    request_output.parent.mkdir(parents=True, exist_ok=True)
                    completion.output.save(request_output)
                    digest = hashlib.sha256(request_output.read_bytes()).hexdigest()
                    results.append(
                        {
                            "request_id": completion.request_id,
                            "output": str(request_output.resolve()),
                            "sha256": digest,
                            "submitted_at": completion.submitted_at,
                            "started_at": completion.started_at,
                            "completed_at": completion.completed_at,
                        }
                    )
                    print(
                        f"Saved embedded Qwen-Image output to {request_output.resolve()}"
                    )
                    print(f"sha256={digest}")
            output.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "world_size": world.world_size,
                        "policy": args.policy,
                        "steps": args.steps,
                        "secondary_steps": args.secondary_steps,
                        "true_cfg_scale": args.true_cfg_scale,
                        "seed": args.seed,
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
