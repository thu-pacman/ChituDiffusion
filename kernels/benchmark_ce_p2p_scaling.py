#!/usr/bin/env python3
"""Measure Copy-Engine P2P scaling for pair, fan-out, and bidirectional traffic."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * 64)]


def _load_cudart() -> ctypes.CDLL:
    candidates = [
        "libcudart.so",
        "libcudart.so.13",
        str(
            Path(torch.__file__).parents[1]
            / "nvidia"
            / "cu13"
            / "lib"
            / "libcudart.so.13"
        ),
    ]
    errors: list[str] = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("unable to load CUDA runtime:\n" + "\n".join(errors))


class CudaRuntime:
    MEMCPY_DEVICE_TO_HOST = 2
    IPC_LAZY_ENABLE_PEER_ACCESS = 1

    def __init__(self) -> None:
        self.lib = _load_cudart()
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(_CudaIpcMemHandle),
            ctypes.c_void_p,
        ]
        self.lib.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _CudaIpcMemHandle,
            ctypes.c_uint,
        ]
        self.lib.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]

    def check(self, status: int, operation: str) -> None:
        if status != 0:
            message = self.lib.cudaGetErrorString(status).decode("utf-8")
            raise RuntimeError(f"{operation} failed: CUDA error {status}: {message}")

    def malloc(self, nbytes: int) -> int:
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), nbytes), "cudaMalloc")
        if pointer.value is None:
            raise RuntimeError("cudaMalloc returned a null pointer")
        return pointer.value

    def free(self, pointer: int) -> None:
        self.check(self.lib.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def memset(self, pointer: int, value: int, nbytes: int) -> None:
        self.check(
            self.lib.cudaMemset(ctypes.c_void_p(pointer), value, nbytes),
            "cudaMemset",
        )

    def ipc_handle(self, pointer: int) -> bytes:
        handle = _CudaIpcMemHandle()
        self.check(
            self.lib.cudaIpcGetMemHandle(
                ctypes.byref(handle), ctypes.c_void_p(pointer)
            ),
            "cudaIpcGetMemHandle",
        )
        return bytes(handle.reserved)

    def open_ipc_handle(self, payload: bytes) -> int:
        if len(payload) != 64:
            raise ValueError(f"CUDA IPC handle must be 64 bytes, got {len(payload)}")
        handle = _CudaIpcMemHandle()
        ctypes.memmove(handle.reserved, payload, len(payload))
        pointer = ctypes.c_void_p()
        self.check(
            self.lib.cudaIpcOpenMemHandle(
                ctypes.byref(pointer),
                handle,
                self.IPC_LAZY_ENABLE_PEER_ACCESS,
            ),
            "cudaIpcOpenMemHandle",
        )
        if pointer.value is None:
            raise RuntimeError("cudaIpcOpenMemHandle returned a null pointer")
        return pointer.value

    def close_ipc_handle(self, pointer: int) -> None:
        self.check(
            self.lib.cudaIpcCloseMemHandle(ctypes.c_void_p(pointer)),
            "cudaIpcCloseMemHandle",
        )

    def memcpy_async(
        self,
        destination: int,
        source: int,
        nbytes: int,
        stream: torch.cuda.Stream,
    ) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source),
                nbytes,
                4,  # cudaMemcpyDefault: resolve direction from UVA peer pointers.
                ctypes.c_void_p(stream.cuda_stream),
            ),
            "cudaMemcpyAsync",
        )

    def read_byte(self, pointer: int) -> int:
        value = ctypes.c_ubyte()
        self.check(
            self.lib.cudaMemcpy(
                ctypes.byref(value),
                ctypes.c_void_p(pointer),
                1,
                self.MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy(D2H)",
        )
        return int(value.value)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[int(quantile * (len(ordered) - 1))]


def _measure_iteration(
    runtime: CudaRuntime,
    source: int,
    destinations: list[int],
    streams: list[torch.cuda.Stream],
    nbytes: int,
) -> float:
    if not destinations:
        return 0.0
    control = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    ready = torch.cuda.Event()
    done = [torch.cuda.Event() for _ in destinations]
    start.record(control)
    ready.record(control)
    for stream, destination, completion in zip(
        streams, destinations, done, strict=True
    ):
        stream.wait_event(ready)
        runtime.memcpy_async(destination, source, nbytes, stream)
        completion.record(stream)
        control.wait_event(completion)
    stop.record(control)
    stop.synchronize()
    return start.elapsed_time(stop)


def _run_case(
    *,
    name: str,
    flows: tuple[tuple[int, int], ...],
    runtime: CudaRuntime,
    source: int,
    local_destination: int,
    remote_destinations: dict[int, int],
    nbytes: int,
    warmup: int,
    iterations: int,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    outgoing = [dst for src, dst in flows if src == rank]
    incoming = [src for src, dst in flows if dst == rank]
    destination_pointers = [remote_destinations[dst] for dst in outgoing]
    streams = [torch.cuda.Stream(priority=0) for _ in outgoing]

    runtime.memset(local_destination, 0, nbytes)
    dist.barrier()
    _measure_iteration(runtime, source, destination_pointers, streams, nbytes)
    torch.cuda.synchronize()
    dist.barrier()
    if incoming:
        observed = runtime.read_byte(local_destination)
        expected = incoming[-1] + 1
        if observed != expected:
            raise AssertionError(
                f"{name}: rank {rank} received byte {observed}, expected {expected}"
            )
    dist.barrier()

    local_samples: list[float] = []
    for index in range(warmup + iterations):
        # The barrier is outside the timed GPU interval. It aligns independent
        # processes so bidirectional traffic is genuinely concurrent.
        dist.barrier()
        elapsed = _measure_iteration(
            runtime, source, destination_pointers, streams, nbytes
        )
        dist.barrier()
        if index >= warmup:
            local_samples.append(elapsed)

    gathered: list[list[float] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_samples)
    sender_samples = [
        samples
        for sender, samples in enumerate(gathered)
        if samples is not None and any(src == sender for src, _ in flows)
    ]
    critical_path = [
        max(samples[index] for samples in sender_samples)
        for index in range(iterations)
    ]
    median_ms = statistics.median(critical_path)
    aggregate_bytes = nbytes * len(flows)
    return {
        "name": name,
        "flows": [f"{source_rank}->{destination_rank}" for source_rank, destination_rank in flows],
        "flow_count": len(flows),
        "bytes_per_flow": nbytes,
        "median_ms": median_ms,
        "p95_ms": _percentile(critical_path, 0.95),
        "minimum_ms": min(critical_path),
        "maximum_ms": max(critical_path),
        "per_flow_GBps_using_wall_time": nbytes / median_ms / 1e6,
        "aggregate_payload_GBps": aggregate_bytes / median_ms / 1e6,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--require-world-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size_mib <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("size, warmup, and iterations must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.require_world_size:
        raise ValueError(
            f"expected {args.require_world_size} ranks, got {world_size}"
        )

    runtime = CudaRuntime()
    nbytes = args.size_mib << 20
    source = runtime.malloc(nbytes)
    local_destination = runtime.malloc(nbytes)
    runtime.memset(source, rank + 1, nbytes)
    runtime.memset(local_destination, 0, nbytes)

    local_handle = runtime.ipc_handle(local_destination)
    handles: list[bytes | None] = [None] * world_size
    dist.all_gather_object(handles, local_handle)
    remote_destinations = {
        peer: runtime.open_ipc_handle(handle)
        for peer, handle in enumerate(handles)
        if peer != rank and handle is not None
    }

    fanout_flows = tuple((0, peer) for peer in range(1, world_size))
    ring_flows = tuple(
        (peer, (peer + 1) % world_size) for peer in range(world_size)
    )
    cases = (
        ("single_0_to_1", ((0, 1),)),
        ("fanout_0_to_all_peers", fanout_flows),
        ("bidirectional_0_1", ((0, 1), (1, 0))),
        ("ring_one_hop_all_ranks", ring_flows),
    )
    try:
        results = [
            _run_case(
                name=name,
                flows=flows,
                runtime=runtime,
                source=source,
                local_destination=local_destination,
                remote_destinations=remote_destinations,
                nbytes=nbytes,
                warmup=args.warmup,
                iterations=args.iterations,
                rank=rank,
                world_size=world_size,
            )
            for name, flows in cases
        ]
    finally:
        dist.barrier()
        for pointer in remote_destinations.values():
            runtime.close_ipc_handle(pointer)
        runtime.free(local_destination)
        runtime.free(source)

    single_bw = float(results[0]["aggregate_payload_GBps"])
    fanout_bw = float(results[1]["aggregate_payload_GBps"])
    bidirectional_bw = float(results[2]["aggregate_payload_GBps"])
    result = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "world_size": world_size,
        "size_mib_per_flow": args.size_mib,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timing": "CUDA-event critical path; inter-rank barrier excluded",
        "transport": "cudaMemcpyAsync(cudaMemcpyDefault) to CUDA IPC peer pointer",
        "cases": results,
        "scaling": {
            "fanout_aggregate_over_single": fanout_bw / single_bw,
            "bidirectional_aggregate_over_single": bidirectional_bw / single_bw,
        },
    }
    if rank == 0:
        encoded = json.dumps(result, indent=2, sort_keys=True)
        print("CE_P2P_SCALING=" + json.dumps(result, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
