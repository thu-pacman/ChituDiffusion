from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from chitu_diffusion.models.minimax_h3.video_vae import load_video_vae
from chitu_diffusion.parallel.vae import create_vae_parallel_group


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/dockerdata/MiniMax-H3/FL2VA/video_vae",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--degree", type=int, default=None)
    return parser.parse_args()


def _timed(device: torch.device, fn):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(device)
    return result, (time.perf_counter() - started) * 1000.0


def main() -> None:
    args = _args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    degree = args.degree or world_size
    if degree != world_size:
        raise ValueError("benchmark torchrun world size must equal --degree")
    topology = create_vae_parallel_group(degree)
    device = torch.device("cuda", local_rank)
    vae = load_video_vae(args.model_path, device=device)

    generator = torch.Generator(device=device).manual_seed(20260826)
    latents = torch.randn(
        1,
        24,
        37,
        48,
        48,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    vae.decode(latents[:, :, :7, :16, :16])
    torch.cuda.synchronize(device)
    dist.barrier()

    serial = None
    serial_ms = None
    serial_peak = None
    if rank == 0:
        torch.cuda.reset_peak_memory_stats(device)
        serial, serial_ms = _timed(device, lambda: vae.decode(latents))
        serial_peak = torch.cuda.max_memory_allocated(device)
    dist.barrier()

    torch.cuda.reset_peak_memory_stats(device)
    parallel, parallel_ms = _timed(
        device,
        lambda: vae.decode_parallel(latents, topology=topology),
    )
    parallel_peak = torch.cuda.max_memory_allocated(device)
    local = {
        "rank": rank,
        "parallel_ms": parallel_ms,
        "parallel_peak_bytes": parallel_peak,
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local)

    if rank == 0:
        assert serial is not None and parallel is not None
        difference = (serial - parallel).float()
        mse = difference.square().mean().item()
        max_abs = difference.abs().max().item()
        psnr = math.inf if mse == 0 else 10 * math.log10(1.0 / mse)
        rank_reports = [report for report in gathered if report is not None]
        parallel_wall_ms = max(report["parallel_ms"] for report in rank_reports)
        result = {
            "shape": list(latents.shape),
            "serial_ms": serial_ms,
            "serial_peak_bytes": serial_peak,
            "parallel_wall_ms": parallel_wall_ms,
            "parallel_rank_reports": rank_reports,
            "speedup": serial_ms / parallel_wall_ms,
            "max_abs": max_abs,
            "mse": mse,
            "psnr_db": psnr,
            "output_shape": list(parallel.shape),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
    topology.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
