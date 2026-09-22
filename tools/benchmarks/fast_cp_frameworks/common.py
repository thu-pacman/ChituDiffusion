"""Shared measurement and reporting for Fast CP framework comparisons."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def build_parser(*, framework: str, methods: tuple[str, ...]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{framework} dense non-causal QKV-to-O context-parallel benchmark"
    )
    parser.add_argument("--method", choices=methods, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--component",
        choices=("e2e", "communication", "attention"),
        default="e2e",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="bracket measured iterations with the CUDA Profiler API",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.sequence <= 0 or args.sequence % world:
        raise ValueError("global sequence must be positive and divisible by world size")
    if args.batch <= 0 or args.heads <= 0 or args.head_dim <= 0:
        raise ValueError("batch, heads, and head-dim must be positive")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative; iterations and repeats positive")


def init_torch_distributed() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def make_inputs(
    args: argparse.Namespace, *, rank: int, world: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    local_sequence = args.sequence // world
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    shape = (args.batch, local_sequence, args.heads, args.head_dim)
    return tuple(
        torch.randn(
            shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        for _ in range(3)
    )  # type: ignore[return-value]


def cudnn_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).contiguous()


def reference_output(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    world = dist.get_world_size()
    if world == 1:
        return cudnn_attention(query, key, value)
    key_parts = [torch.empty_like(key) for _ in range(world)]
    value_parts = [torch.empty_like(value) for _ in range(world)]
    dist.all_gather(key_parts, key.contiguous())
    dist.all_gather(value_parts, value.contiguous())
    return cudnn_attention(
        query,
        torch.cat(key_parts, dim=1),
        torch.cat(value_parts, dim=1),
    )


def measure(
    call: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    device: torch.device,
    profile: bool = False,
) -> dict[str, Any]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)

    samples: list[float] = []
    repeat_medians: list[float] = []
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        for _ in range(repeats):
            current: list[float] = []
            for _ in range(iterations):
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                call()
                end.record()
                end.synchronize()
                rank_max = torch.tensor(start.elapsed_time(end), device=device)
                dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
                current.append(float(rank_max))
            samples.extend(current)
            repeat_medians.append(statistics.median(current))
    finally:
        if profile:
            torch.cuda.cudart().cudaProfilerStop()

    ordered = sorted(samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return {
        "statistic": "rank-max CUDA-event latency",
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "repeat_medians_ms": repeat_medians,
        "samples_ms": samples,
        "max_peak_memory_bytes": int(peak),
    }


def git_revision(path: Path | None) -> str | None:
    if path is None:
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def git_dirty(path: Path | None) -> bool | None:
    if path is None:
        return None
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    return None if result.returncode else bool(result.stdout.strip())


def run_benchmark(
    *,
    framework: str,
    args: argparse.Namespace,
    call: Callable[[], Any],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    device: torch.device,
    implementation: dict[str, Any],
    verify: bool = True,
    boundary: str | None = None,
) -> dict[str, Any]:
    rank, world = dist.get_rank(), dist.get_world_size()
    if verify:
        actual = call()
        expected = reference_output(query, key, value)
        delta = actual.float() - expected.float()
        accuracy = {
            "checked": True,
            "rtol": args.rtol,
            "atol": args.atol,
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(
                delta.norm() / expected.float().norm().clamp_min(1e-12)
            ),
        }
        torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
    else:
        accuracy = {"checked": False, "reason": "component timing"}
    timing = measure(
        call,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        device=device,
        profile=args.profile,
    )
    hostnames: list[str | None] = [None] * world
    dist.all_gather_object(hostnames, socket.gethostname())
    report = {
        "schema_version": 1,
        "status": "complete",
        "framework": framework,
        "method": args.method,
        "boundary": boundary
        or "pre-projected local Q/K/V through communication and native attention to local O",
        "semantics": "BF16 dense non-causal self-attention",
        "cuda_graph": False,
        "shape": {
            "batch": args.batch,
            "global_sequence": args.sequence,
            "local_sequence": args.sequence // world,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": "bfloat16",
            "cp_degree": world,
        },
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "framework_revision": git_revision(args.source_root),
            "framework_worktree_dirty": git_dirty(args.source_root),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "world_size": world,
            "nodes": sorted({hostname for hostname in hostnames if hostname}),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "implementation": implementation,
        "accuracy": accuracy,
        "timing": timing,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True), flush=True)
    return report
