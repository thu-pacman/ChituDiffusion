#!/usr/bin/env python3
"""Benchmark CE transport, prefetched compute, and online Flash Ring."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from chitu_diffusion.parallel.fast_cp.ring import (
    FastRingKVTransport,
    FastRingAttention,
    RingKVPhase,
)
from chitu_diffusion.parallel.fast_cp._runtime import FastUlyssesAllToAll


def _all_gather_sequence(tensor: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    gathered = torch.empty(
        world_size * tensor.shape[0],
        *tensor.shape[1:],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(gathered, tensor)
    return (
        gathered.unflatten(0, (world_size, tensor.shape[0]))
        .permute(1, 0, 2, 3, 4)
        .flatten(1, 2)
        .contiguous()
    )


@dataclass
class _PrefetchedSession:
    blocks: list[tuple[torch.Tensor, torch.Tensor]]
    rank: int
    released: int = 0

    @property
    def phase_count(self) -> int:
        return len(self.blocks)

    def start(self) -> None:
        """Every shard is already resident, so there is nothing to submit."""

    def acquire(self, phase_index: int) -> RingKVPhase:
        source = (self.rank - phase_index) % self.phase_count
        key, value = self.blocks[source]
        return RingKVPhase(phase_index, source, key, value)

    def release(self, phase: RingKVPhase) -> None:
        self.released += 1

    def close(self) -> None:
        if self.released != self.phase_count:
            raise RuntimeError("incomplete prefetched traversal")


class _PrefetchedTransport:
    def __init__(
        self,
        blocks: list[tuple[torch.Tensor, torch.Tensor]],
        rank: int,
    ) -> None:
        self.blocks = blocks
        self.rank = rank
        self.world_size = len(blocks)

    def begin(self, key: torch.Tensor, value: torch.Tensor) -> _PrefetchedSession:
        return _PrefetchedSession(self.blocks, self.rank)


def _measure_critical_path(
    call,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
    progress=None,
) -> dict[str, float]:
    for index in range(warmup):
        if progress is not None:
            progress(f"warmup {index + 1}/{warmup}")
        call()
    torch.cuda.synchronize(device)
    samples: list[float] = []
    host_samples: list[float] = []
    for index in range(iterations):
        if progress is not None:
            progress(f"iteration {index + 1}/{iterations}")
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        host_start = time.perf_counter()
        call()
        # Host dispatch time excludes any wait on the device.
        host_samples.append((time.perf_counter() - host_start) * 1e3)
        stop.record()
        stop.synchronize()
        critical = torch.tensor(
            start.elapsed_time(stop),
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(critical, op=dist.ReduceOp.MAX)
        samples.append(critical.item())
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "host_dispatch_median_ms": statistics.median(host_samples),
    }


def _capture_cuda_trace(
    call,
    *,
    label: str,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> None:
    """Capture only steady-state calls when launched under nsys."""

    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        # The first GPU activity after cudaProfilerStart carries a large Nsight
        # capture-start penalty. Keep it out of the measured iteration.
        torch.cuda.nvtx.range_push(f"{label}/capture_lead_in")
        try:
            call()
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)
        # The profiled rank alone pays capture instrumentation overhead. Align
        # ranks again before measuring the ticket-driven ring traversal.
        dist.barrier()
        for index in range(iterations):
            torch.cuda.nvtx.range_push(f"{label}/iteration_{index}")
            try:
                call()
            finally:
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize(device)
    finally:
        torch.cuda.cudart().cudaProfilerStop()
    dist.barrier()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-sequence", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=0,
        help="capture this many steady-state online calls via cudaProfilerApi",
    )
    parser.add_argument("--pool-bytes", type=int, default=2 << 30)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="print rank-local stage progress to diagnose distributed stalls",
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local_rank),
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in {2, 4, 8}:
        raise ValueError("benchmark requires CP2, CP4, or CP8")
    if args.global_sequence % world_size:
        raise ValueError("global sequence must be divisible by CP degree")
    local_sequence = args.global_sequence // world_size
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda", local_rank)
    generator = torch.Generator(device=device).manual_seed(20260815 + rank)
    shape = (
        args.batch,
        local_sequence,
        args.heads,
        args.head_dim,
    )
    query, key, value = [
        torch.randn(shape, dtype=dtype, device=device, generator=generator)
        for _ in range(3)
    ]
    global_key = _all_gather_sequence(key)
    global_value = _all_gather_sequence(value)
    blocks = [
        (
            global_key[:, source * local_sequence : (source + 1) * local_sequence],
            global_value[:, source * local_sequence : (source + 1) * local_sequence],
        )
        for source in range(world_size)
    ]
    runtime = FastUlyssesAllToAll(
        dist.group.WORLD,
        device,
        pool_bytes=args.pool_bytes,
    )
    ce_transport = FastRingKVTransport(
        runtime,
        rank=rank,
        world_size=world_size,
        tag_prefix="flash_ring_benchmark",
    )
    online = FastRingAttention(ce_transport)
    prefetched = FastRingAttention(_PrefetchedTransport(blocks, rank))

    def progress(stage: str, detail: str = "") -> None:
        if args.progress:
            suffix = f" {detail}" if detail else ""
            print(f"FLASH_RING_PROGRESS rank={rank} stage={stage}{suffix}", flush=True)

    def transport_only() -> None:
        session = ce_transport.begin(key, value)
        # No attention runs here, so the traversal is kicked immediately; the
        # compute path instead starts it once phase zero is on the stream.
        session.start()
        for phase_index in range(world_size):
            phase = session.acquire(phase_index)
            session.release(phase)
        session.close()

    try:
        progress("correctness", "prefetched")
        expected, expected_lse = prefetched(
            query,
            key,
            value,
            return_lse=True,
        )
        progress("correctness", "online")
        actual, actual_lse = online(
            query,
            key,
            value,
            return_lse=True,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(
            actual_lse,
            expected_lse,
            rtol=2e-3,
            atol=2e-3,
        )
        if args.profile_iterations:
            _capture_cuda_trace(
                lambda: online(query, key, value),
                label=f"fast_ring/S{args.global_sequence}/CP{world_size}",
                warmup=args.warmup,
                iterations=args.profile_iterations,
                device=device,
            )
        progress("correctness", "done")
        progress("transport", "start")
        transport_latency = _measure_critical_path(
            transport_only,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
            progress=lambda detail: progress("transport", detail),
        )
        progress("transport", "done")
        progress("prefetched", "start")
        prefetched_latency = _measure_critical_path(
            lambda: prefetched(query, key, value),
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
            progress=lambda detail: progress("prefetched", detail),
        )
        progress("prefetched", "done")
        progress("online", "start")
        online_latency = _measure_critical_path(
            lambda: online(query, key, value),
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
            progress=lambda detail: progress("online", detail),
        )
        progress("online", "done")
        exposed_ms = max(
            0.0,
            online_latency["median_ms"] - prefetched_latency["median_ms"],
        )
        exposed_fraction = exposed_ms / transport_latency["median_ms"]
        compute_overhead_fraction = (
            online_latency["median_ms"] / prefetched_latency["median_ms"] - 1.0
        )
        local_tensor_bytes = key.numel() * key.element_size()
        result = {
            "device": torch.cuda.get_device_name(device),
            "cp_degree": world_size,
            "dtype": args.dtype,
            "batch": args.batch,
            "global_sequence": args.global_sequence,
            "local_sequence": local_sequence,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "transport_only": transport_latency,
            "prefetched_compute": prefetched_latency,
            "online_flash_ring": online_latency,
            "exposed_communication_ms": exposed_ms,
            "exposed_transport_fraction": exposed_fraction,
            "online_over_prefetched_fraction": compute_overhead_fraction,
            "kv_symmetric_bytes": 4 * local_tensor_bytes,
            "kv_symmetric_local_shards": 2,
            "running_output_lse_bytes": (
                query.numel()
                + args.batch * args.heads * local_sequence
            )
            * torch.float32.itemsize,
            "correctness": "CuTe output/LSE matched prefetched shard traversal",
            "passes_95_percent_overlap_gate": exposed_fraction <= 0.05,
            "passes_3_percent_online_overhead_gate": (
                compute_overhead_fraction <= 0.03
            ),
        }
        if rank == 0:
            print("FLASH_RING_RESULT=" + json.dumps(result, sort_keys=True))
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    finally:
        runtime._group.destroy()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
