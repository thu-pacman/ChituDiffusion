"""Compare dense and static layer-wise VAE decode on identical latent tensors.

Launch with torchrun. Model loading, transfers, and synchronization between
samples are excluded from the timed region. Optionally compare DistVAE too.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from diffusers import (
    AutoencoderKL,
    AutoencoderKLFlux2,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
)

from chitu_diffusion.parallel.vae import parallel_vae_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=("kl", "flux2", "wan", "qwen"), required=True
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-rmse", type=float, default=0.01)
    parser.add_argument("--distvae", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        cls = {
            "kl": AutoencoderKL,
            "flux2": AutoencoderKLFlux2,
            "wan": AutoencoderKLWan,
            "qwen": AutoencoderKLQwenImage,
        }[args.family]
        vae = (
            cls.from_pretrained(
                args.model_path,
                subfolder="vae",
                local_files_only=True,
                torch_dtype=getattr(torch, args.dtype),
            )
            .to(device)
            .eval()
        )
        video = args.family in {"wan", "qwen"}
        config = vae.config
        spatial_scale = 2 ** (
            len(config.dim_mult if video else config.block_out_channels) - 1
        )
        shape = [1, config.z_dim if video else config.latent_channels]
        if video:
            temporal_scale = 2 ** sum(config.temperal_downsample)
            shape.append((args.frames - 1) // temporal_scale + 1)
        shape.extend((args.height // spatial_scale, args.width // spatial_scale))
        latent = torch.randn(shape, generator=torch.Generator().manual_seed(7)).to(
            device, vae.dtype
        )
        topology = SimpleNamespace(
            rank_in_lane=rank,
            width=world,
            process_group=dist.group.WORLD,
            is_leader=rank == 0,
        )

        def measure(decode):
            output = decode()
            del output
            torch.cuda.synchronize()
            samples = []
            peaks = []
            output = None
            for _ in range(args.repeats):
                del output
                dist.barrier()
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                output = decode()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated() - start_memory
                pair = torch.tensor([elapsed, peak], dtype=torch.float64, device=device)
                dist.all_reduce(pair, op=dist.ReduceOp.MAX)
                elapsed, peak = pair.tolist()
                samples.append(elapsed * 1000)
                peaks.append(peak / 2**20)
            host = output.cpu().float() if rank == 0 else None
            return host, {
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
                "peak_extra_mib": max(peaks),
            }

        with torch.inference_mode():
            expected, dense = measure(lambda: vae.decode(latent, return_dict=False)[0])
            actual, parallel = measure(
                lambda: parallel_vae_decode(vae, latent, topology=topology)
            )
            report = {
                "family": args.family,
                "dtype": args.dtype,
                "allow_tf32": args.allow_tf32,
                "shape": shape,
                "world_size": world,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "dense": dense,
                "parallel": parallel,
            }
            report["speedup"] = dense["median_ms"] / parallel["median_ms"]
            if rank == 0:
                difference = actual - expected
                report["rmse"] = difference.square().mean().sqrt().item()
                report["max_error"] = difference.abs().max().item()
                boundary = ((shape[-2] + world - 1) // world) * spatial_scale
                if 0 < boundary < expected.shape[-2]:
                    for name, image in (("dense", expected), ("parallel", actual)):
                        report[name]["boundary_jump"] = (
                            (image[..., boundary, :] - image[..., boundary - 1, :])
                            .abs()
                            .mean()
                            .item()
                        )
            if args.distvae:
                from distvae.vae import parallelize_decoder

                parallelize_decoder(vae, dist.group.WORLD)
                reference, reference_stats = measure(
                    lambda: vae.decode(latent, return_dict=False)[0]
                )
                if rank == 0:
                    reference_stats["rmse"] = (
                        (reference - expected).square().mean().sqrt().item()
                    )
                    reference_stats["max_error"] = (
                        (reference - expected).abs().max().item()
                    )
                    report["distvae"] = reference_stats
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2), flush=True)
                assert report["rmse"] <= args.max_rmse, (
                    "parallel VAE exceeds the requested RMSE tolerance"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
