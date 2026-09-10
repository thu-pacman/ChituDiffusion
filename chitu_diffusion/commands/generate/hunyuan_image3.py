from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from chitu_diffusion.commands.generate.common import (
    generation_timestamp,
    write_generation_metadata,
)
from chitu_diffusion.commands.parallel_args import (
    add_vae_parallel_arguments,
    vae_parallel_degree,
    validate_vae_parallel_arguments,
)
from chitu_diffusion.epe.scheduling.planner import EpeSchedulingModule
from chitu_diffusion.models.hunyuan_image3 import (
    ATTENTION_MODES,
    HunyuanImage3Executor,
    HunyuanImage3ParallelPlan,
    HunyuanImage3ParallelRuntime,
    HunyuanImage3Pipeline,
    HunyuanImage3Request,
    load_hunyuan_image3,
    resolve_parallel_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one Hunyuan Image 3 result on a single-node "
            "TPxCFGxCPxEP stage. Launch with torchrun; the degrees default to "
            "TP2xCFG2xCP2xEP4 on eight GPUs."
        )
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("HUNYUAN_IMAGE3_MODEL_PATH"),
        help="Local HunyuanImage-3.0 checkpoint (or set HUNYUAN_IMAGE3_MODEL_PATH).",
    )
    parser.add_argument("--prompt", default="a red cube on a white table")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        dest="images",
        help="Conditioning image for Instruct image-to-image; repeat to add more.",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument(
        "--attention-mode",
        choices=ATTENTION_MODES,
        default="agkv",
        help="Context-parallel attention transport; AGKV is the default.",
    )
    parser.add_argument(
        "--ulysses-transport",
        choices=("auto", "torch", "fast_ulysses"),
        default=None,
        help="Ulysses exchange backend; defaults to CHITU_ULYSSES_TRANSPORT/torch.",
    )
    parser.add_argument(
        "--agkv-transport",
        choices=("auto", "torch", "fast_agkv"),
        default=None,
        help="AGKV exchange backend; defaults to CHITU_AGKV_TRANSPORT/torch.",
    )
    parser.add_argument("--tensor-parallel-degree", type=int, default=2)
    parser.add_argument(
        "--cfg-parallel-degree",
        type=int,
        default=2,
        choices=(1, 2),
        help="2 runs the guided branches on separate lanes, 1 keeps both local.",
    )
    parser.add_argument(
        "--context-parallel-degree",
        type=int,
        default=None,
        help="Defaults to the ranks left in a TP plane after the CFG split.",
    )
    parser.add_argument(
        "--expert-parallel-degree",
        type=int,
        default=None,
        help="Defaults to the whole TP plane; smaller degrees replicate experts.",
    )
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--seed", type=int, default=0)
    add_vae_parallel_arguments(parser, with_degree=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/chitu/hunyuan_image3.png"),
    )
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or HUNYUAN_IMAGE3_MODEL_PATH is required")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    validate_vae_parallel_arguments(parser, args)
    return args


def build_plan(args: argparse.Namespace, world_size: int) -> HunyuanImage3ParallelPlan:
    """Resolve the launch degrees against the world torchrun actually started."""

    try:
        return resolve_parallel_plan(
            world_size=world_size,
            tensor_parallel_degree=args.tensor_parallel_degree,
            cfg_parallel_degree=args.cfg_parallel_degree,
            context_parallel_degree=args.context_parallel_degree,
            expert_parallel_degree=args.expert_parallel_degree,
            attention_mode=args.attention_mode,
        )
    except ValueError as exc:
        raise SystemExit(f"unsupported Hunyuan Image 3 topology: {exc}") from exc


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    plan = build_plan(args, world_size)
    runtime = HunyuanImage3ParallelRuntime.from_torchrun(
        plan=plan,
        num_experts=HunyuanImage3ParallelRuntime.checkpoint_num_experts(
            args.model_path
        ),
        vae_parallel_degree=vae_parallel_degree(args, world_size),
        vae_parallel_halo=args.vae_parallel_halo,
        ulysses_transport=args.ulysses_transport,
        agkv_transport=args.agkv_transport,
    )
    load_started = time.perf_counter()
    loaded = load_hunyuan_image3(args.model_path, runtime=runtime)
    load_ms = (time.perf_counter() - load_started) * 1000
    pipeline = HunyuanImage3Pipeline(
        loaded,
        runtime,
        flow_shift=args.flow_shift,
    )
    executor = HunyuanImage3Executor(
        pipeline,
        EpeSchedulingModule(runtime.context, policy="static_cp"),
        default_height=args.height,
        default_width=args.width,
        default_num_steps=args.steps,
        default_guidance_scale=args.guidance_scale,
    )
    try:
        started = generation_timestamp()
        images = executor.generate(
            HunyuanImage3Request(
                prompt=args.prompt,
                image_paths=tuple(args.images),
                height=args.height,
                width=args.width,
                num_steps=args.steps,
                guidance_scale=args.guidance_scale,
                system_prompt=args.system_prompt,
                seed=args.seed,
            )
        )
        elapsed_s = generation_timestamp() - started
        total_ms = elapsed_s * 1000
        if runtime.rank == 0:
            if not images:
                raise RuntimeError("rank 0 did not receive a decoded image")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(args.output)
            metadata = {
                "prompt": args.prompt,
                "image_paths": list(args.images),
                "requested_size": [args.height, args.width],
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "load_ms": load_ms,
                "generate_ms": total_ms,
                "topology": plan.describe(),
                "tensor_parallel_degree": plan.tensor_parallel_degree,
                "cfg_parallel_degree": plan.cfg_parallel_degree,
                "context_parallel_degree": plan.context_parallel_degree,
                "expert_parallel_degree": plan.expert_parallel_degree,
                "vae_parallel_degree": runtime.vae_placement.degree,
                "attention_mode": plan.attention_mode,
            }
            write_generation_metadata(
                args.output,
                args=args,
                pipeline=pipeline,
                elapsed_s=elapsed_s,
                metadata=metadata,
            )
            print(f"Saved Hunyuan Image 3 result to {args.output.resolve()}")
            print(f"load {load_ms:.0f} ms, generate {total_ms:.0f} ms")
        if dist.is_initialized():
            dist.barrier(
                device_ids=[runtime.context.local_rank]
                if torch.cuda.is_available()
                else None
            )
    finally:
        executor.close()


if __name__ == "__main__":
    main()
