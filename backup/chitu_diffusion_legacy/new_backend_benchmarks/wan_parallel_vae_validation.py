from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.utils import export_to_video

from chitu_diffusers import WanEPACPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Wan leader-only and lane-parallel VAE on one latent"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="A cat walking on grass.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae-parallel-halo", type=int, default=8)
    parser.add_argument("--attention-mode", choices=("agkv", "usp"), default="agkv")
    parser.add_argument("--ulysses-degree", type=int)
    args = parser.parse_args()
    if args.vae_parallel_halo < 0:
        parser.error("--vae-parallel-halo must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    facade = WanEPACPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attention_mode=args.attention_mode,
        ulysses_degree=args.ulysses_degree,
        cfg_parallel=True,
        parallel_vae=True,
        vae_parallel_halo=args.vae_parallel_halo,
        flow_shift=args.flow_shift,
    )
    pipeline = facade.diffusers_pipeline
    parallel = pipeline.parallel_context
    lane = tuple(range(parallel.world_size))
    device = torch.device("cuda", parallel.local_rank)
    try:
        state = pipeline.prepare_request(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
        while not state.complete:
            pipeline.denoise_step(state, lane_ranks=lane)

        with parallel.activate(lane) as topology:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            reference = pipeline.decode_request(
                state,
                topology=topology,
                parallel_vae=False,
            )
            torch.cuda.synchronize(device)
            reference_ms = (time.perf_counter() - started) * 1000.0
            if topology.process_group is not None:
                dist.barrier(group=topology.process_group)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            decoded = pipeline.decode_request(
                state,
                topology=topology,
                parallel_vae=True,
                vae_parallel_halo=args.vae_parallel_halo,
            )
            torch.cuda.synchronize(device)
            parallel_ms = (time.perf_counter() - started) * 1000.0

        if parallel.rank == 0:
            assert reference is not None and decoded is not None
            difference = decoded.float() - reference.float()
            mse = float(difference.square().mean().item())
            metrics = {
                "world_size": parallel.world_size,
                "attention_mode": args.attention_mode,
                "ulysses_degree": args.ulysses_degree,
                "shape": list(decoded.shape),
                "vae_parallel_halo": args.vae_parallel_halo,
                "reference_ms": reference_ms,
                "parallel_ms": parallel_ms,
                "speedup": reference_ms / parallel_ms,
                "mae": float(difference.abs().mean().item()),
                "max_abs": float(difference.abs().max().item()),
                "psnr_db": float("inf") if mse == 0 else 10.0 * math.log10(4.0 / mse),
            }
            args.output_dir.mkdir(parents=True, exist_ok=True)
            reference_video = pipeline.video_processor.postprocess_video(
                reference, output_type="np"
            )[0]
            parallel_video = pipeline.video_processor.postprocess_video(
                decoded, output_type="np"
            )[0]
            export_to_video(
                reference_video,
                str(args.output_dir / "leader-only.mp4"),
                fps=16,
            )
            export_to_video(
                parallel_video,
                str(args.output_dir / "lane-parallel.mp4"),
                fps=16,
            )
            (args.output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            print("WAN_PARALLEL_VAE_VALIDATION", json.dumps(metrics, sort_keys=True))
        if dist.is_initialized():
            dist.barrier()
    finally:
        facade.close()


if __name__ == "__main__":
    main()
