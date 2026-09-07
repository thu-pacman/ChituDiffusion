#!/usr/bin/env python3
"""Benchmark native NCCL and Fast CP inside MiniMax-H3 or Hunyuan Image 3."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

from chitu_diffusion.models.hunyuan_image3 import HunyuanImage3Request
from chitu_diffusion.models.minimax_h3 import MiniMaxH3Request
from chitu_diffusion.serve.config import StageServiceConfig
from chitu_diffusion.serve.runtime import DiffusionServiceRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=("minimax-h3", "hunyuan-image3"),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("nccl-ulysses", "fast-ulysses", "nccl-agkv", "fast-agkv"),
    )
    parser.add_argument(
        "--cfg-parallel-degree",
        type=int,
        default=2,
        choices=(1, 2),
        help="Hunyuan Image 3 only: CFG2 halves the TP plane into CP2 lanes",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _raw_config(args: argparse.Namespace) -> dict[str, Any]:
    name = (
        "stage-minimax-h3.yaml"
        if args.model == "minimax-h3"
        else "stage-hunyuan-image3.yaml"
    )
    raw = yaml.safe_load((Path("examples") / name).read_text(encoding="utf-8"))
    cp = raw["parallelism"]["cp"]
    cp["attention_mode"] = "ulysses" if "ulysses" in args.mode else "agkv"
    cp.pop("ulysses_transport", None)
    cp.pop("agkv_transport", None)
    if args.mode == "fast-ulysses":
        cp["ulysses_transport"] = "fast"
    elif args.mode == "fast-agkv":
        cp["agkv_transport"] = "fast"
    elif args.mode == "nccl-ulysses":
        cp["ulysses_transport"] = "nccl"
    else:
        cp["agkv_transport"] = "nccl"

    scheduler = raw["parallelism"]["scheduler"]
    # The service schema requires at least three; this harness performs its own
    # request-level warmup and does not call DiffusionServiceRuntime.warmup().
    scheduler["warmup_steps"] = 3
    scheduler["online_calibration"] = False
    raw["factory_args"]["num_steps"] = args.steps
    if args.model == "minimax-h3":
        raw["factory_args"]["enable_native_media"] = False
        raw["parallelism"]["vae"] = {"enabled": False}
    else:
        # CFG2 is the shipped layout: each TP plane splits into two CP2 lanes.
        # CFG1 leaves the plane whole as one CP4 lane and needs EP4 to fit.
        cfg_degree = args.cfg_parallel_degree
        raw["parallelism"]["model"]["cfg_parallel_degree"] = cfg_degree
        raw["parallelism"]["model"]["expert_parallel_degree"] = 4 // cfg_degree
        if cp["attention_mode"] == "ulysses":
            cp["ulysses_degree"] = 4 // cfg_degree
        raw["parallelism"]["vae"] = {"enabled": False}
    return raw


def _request(args: argparse.Namespace):
    if args.model == "minimax-h3":
        # 31 x 96/2 x 96/2 = 71,424 video tokens, representative of the
        # long-sequence native video path without timing text/VAE components.
        return MiniMaxH3Request(
            synthetic=True,
            height=96,
            width=96,
            latent_t=31,
            audio_t=64,
            audio_channels=2,
            text_length=128,
            num_steps=args.steps,
            output_type="latent",
            seed=1234,
        )
    return HunyuanImage3Request(
        prompt="a red cube on a white table, studio lighting",
        height=1024,
        width=1024,
        num_steps=args.steps,
        guidance_scale=5.0,
        output_type="latent",
        seed=1234,
    )


def _max_elapsed_ms(started: float) -> float:
    elapsed = torch.tensor(
        (time.perf_counter() - started) * 1000.0,
        dtype=torch.float64,
        device=torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))),
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.cpu())


def main() -> None:
    args = _parser().parse_args()
    if args.warmup_runs < 0 or args.measure_runs < 1 or args.steps < 2:
        raise ValueError("runs must be non-negative and steps must be at least two")
    runtime = DiffusionServiceRuntime.from_config(
        StageServiceConfig.from_mapping(_raw_config(args))
    )
    rank = dist.get_rank()
    samples = []
    last_result = None
    try:
        for index in range(args.warmup_runs + args.measure_runs):
            dist.barrier()
            started = time.perf_counter()
            last_result = runtime.executor.generate(_request(args))
            elapsed_ms = _max_elapsed_ms(started)
            if index >= args.warmup_runs:
                samples.append(elapsed_ms)
    finally:
        runtime.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    if rank == 0:
        tensors = getattr(last_result, "tensors", {})
        if isinstance(last_result, torch.Tensor):
            tensors = {"output": last_result}
        elif isinstance(last_result, dict):
            tensors = last_result
        elif isinstance(last_result, (list, tuple)):
            tensors = {
                f"output_{index}": value
                for index, value in enumerate(last_result)
            }
        cpu_tensors = {
            name: tensor.detach().float().cpu()
            for name, tensor in tensors.items()
            if isinstance(tensor, torch.Tensor)
        }
        result = {
            "model": args.model,
            "mode": args.mode,
            "gpus": 8,
            "gpu": "NVIDIA RTX PRO 5000 72GB Blackwell",
            "steps": args.steps,
            "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs,
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "median_step_ms": statistics.median(samples)
            / (args.steps - 1 if args.model == "minimax-h3" else args.steps),
            "outputs": {
                name: {
                    "shape": list(tensor.shape),
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std()),
                }
                for name, tensor in cpu_tensors.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cpu_tensors, args.output.with_suffix(".pt"))
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
