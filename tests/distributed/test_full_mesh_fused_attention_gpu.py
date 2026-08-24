"""Distributed correctness and latency harness for fused full-mesh attention."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.distributed as dist


def _gather_sequence(tensor: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    """Concatenate every rank's shard. Shards may differ in length, which rules
    out all_gather, so each source broadcasts its own slice."""
    rank = dist.get_rank()
    pieces = []
    for source, length in enumerate(lengths):
        piece = (
            tensor
            if source == rank
            else torch.empty(
                (tensor.shape[0], length, *tensor.shape[2:]),
                device=tensor.device,
                dtype=tensor.dtype,
            )
        )
        dist.broadcast(piece, src=source)
        pieces.append(piece)
    return torch.cat(pieces, dim=1)


def _measure(
    call: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, samples)
    rank_samples = [values for values in gathered if values]
    critical_path = [
        max(values[index] for values in rank_samples) for index in range(iterations)
    ]
    ordered = sorted(critical_path)
    return {
        "median_ms": statistics.median(critical_path),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
    }


def _measure_all_streams(
    call: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    """Measure asynchronous transport through completion on every CUDA stream."""
    for _ in range(warmup):
        call()
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, samples)
    rank_samples = [values for values in gathered if values]
    critical_path = [
        max(values[index] for values in rank_samples) for index in range(iterations)
    ]
    ordered = sorted(critical_path)
    return {
        "median_ms": statistics.median(critical_path),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
    }


def _measure_interleaved(
    calls: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float]]:
    labels = tuple(calls)
    for index in range(warmup):
        for offset in range(len(labels)):
            calls[labels[(index + offset) % len(labels)]]()
    torch.cuda.synchronize()

    local_samples = {label: [] for label in labels}
    for index in range(iterations):
        for offset in range(len(labels)):
            label = labels[(index + offset) % len(labels)]
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            calls[label]()
            stop.record()
            stop.synchronize()
            local_samples[label].append(start.elapsed_time(stop))

    gathered: list[dict[str, list[float]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_samples)
    rank_samples = [samples for samples in gathered if samples]
    report = {}
    for label in labels:
        critical_path = [
            max(samples[label][index] for samples in rank_samples)
            for index in range(iterations)
        ]
        ordered = sorted(critical_path)
        report[label] = {
            "median_ms": statistics.median(critical_path),
            "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        }
    return report


def main() -> None:
    project_root = Path(__file__).parents[2]
    sys.path.insert(0, str(project_root / "refs/kernels/flash-attention"))
    sys.path.insert(0, str(project_root))

    from flash_attn.cute.interface import _flash_attn_fwd

    from chitu_diffusion.parallel.fast_cp.experimental.agkv import (
        FastAgkvAttention,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--local-sequence", type=int)
    parser.add_argument(
        "--global-sequence",
        type=int,
        help="total CP sequence; needs no tile or world-size divisibility",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transport-only", action="store_true")
    parser.add_argument(
        "--cp2-split-compute",
        action="store_true",
        help="measure local/remote partial attention plus exact LSE combine",
    )
    parser.add_argument(
        "--prefetched-kernel",
        action="store_true",
        help="measure the streaming kernel after every K/V segment is ready",
    )
    parser.add_argument(
        "--ab-ready-mask",
        action="store_true",
        help="interleave C1 fixed, C2 fixed, and C2 ready-mask in one process",
    )
    parser.add_argument(
        "--with-transport",
        action="store_true",
        help="run a same-sized transport beside --prefetched-kernel",
    )
    parser.add_argument("--use-sm", action="store_true")
    parser.add_argument("--legacy-stripes", action="store_true")
    parser.add_argument("--chunks", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--scheduler", choices=("fixed", "ready_mask"), default="fixed")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in {2, 4, 8}:
        raise ValueError("world size must be 2, 4, or 8")

    if (args.local_sequence is None) == (args.global_sequence is None):
        raise ValueError("pass exactly one of --local-sequence or --global-sequence")
    global_sequence = args.global_sequence or args.local_sequence * world_size
    lengths = FastAgkvAttention.shard_lengths(global_sequence, world_size)
    local_sequence = lengths[rank]
    if args.batch <= 0:
        raise ValueError("batch must be positive")

    torch.manual_seed(2000 + rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shape = (args.batch, local_sequence, args.heads, args.head_dim)
    query = torch.randn(shape, device="cuda", dtype=dtype)
    key = torch.randn(shape, device="cuda", dtype=dtype)
    value = torch.randn(shape, device="cuda", dtype=dtype)
    full_key = _gather_sequence(key, lengths)
    full_value = _gather_sequence(value, lengths)

    bytes_per_global_tensor = full_key.numel() * full_key.element_size()
    # Two rotated buffers for the fused path, two for the isolated transport
    # measurement, and more for the optional experiment modes.
    layout_copies = 6 + (2 if args.with_transport else 0)
    layout_copies += 4 if args.ab_ready_mask else 0
    pool_bytes = layout_copies * bytes_per_global_tensor + (64 << 20)
    fused = FastAgkvAttention(
        dist.group.WORLD,
        torch.device("cuda", local_rank),
        pool_bytes=pool_bytes,
        use_sm_transport=args.use_sm,
        source_chunk_streaming=not args.legacy_stripes,
        chunk_count=args.chunks,
        scheduler=args.scheduler,
    )
    try:
        if args.transport_only:
            (
                streamed_key,
                streamed_value,
                flags,
                epoch_guard,
                done,
            ) = fused._transport.begin_full_mesh_stream_kv(
                key,
                value,
                lengths,
                key_tag="chitu_full_mesh_debug_key",
                value_tag="chitu_full_mesh_debug_value",
                flag_tag="chitu_full_mesh_debug_flags",
                epoch=1,
                use_sm=args.use_sm,
                source_chunk=not args.legacy_stripes,
                chunk_count=args.chunks,
            )
            torch.cuda.current_stream().wait_event(done)
            torch.cuda.synchronize()
            pieces_key = list(full_key.split(list(lengths), dim=1))
            pieces_value = list(full_value.split(list(lengths), dim=1))
            if args.legacy_stripes:
                stripe = local_sequence // world_size
                expected_key = torch.cat(
                    [
                        piece[:, phase * stripe : (phase + 1) * stripe]
                        for phase in range(world_size)
                        for piece in pieces_key
                    ],
                    dim=1,
                )
                expected_value = torch.cat(
                    [
                        piece[:, phase * stripe : (phase + 1) * stripe]
                        for phase in range(world_size)
                        for piece in pieces_value
                    ],
                    dim=1,
                )
                expected_flags = torch.ones_like(flags)
            else:
                source_order = [
                    (rank - (world_size - 1 - slot)) % world_size
                    for slot in range(world_size)
                ]
                expected_key = torch.cat(
                    [
                        pieces_key[source][
                            :,
                            pieces_key[source].shape[1]
                            * chunk
                            // args.chunks : pieces_key[source].shape[1]
                            * (chunk + 1)
                            // args.chunks,
                        ]
                        for chunk in range(args.chunks)
                        for source in source_order
                    ],
                    dim=1,
                )
                expected_value = torch.cat(
                    [
                        pieces_value[source][
                            :,
                            pieces_value[source].shape[1]
                            * chunk
                            // args.chunks : pieces_value[source].shape[1]
                            * (chunk + 1)
                            // args.chunks,
                        ]
                        for chunk in range(args.chunks)
                        for source in source_order
                    ],
                    dim=1,
                )
                expected_flags = torch.ones_like(flags)
            torch.testing.assert_close(streamed_key, expected_key, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                streamed_value, expected_value, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                flags,
                expected_flags,
                rtol=0.0,
                atol=0.0,
            )
            epoch_guard.record_stream(torch.cuda.current_stream())
            if rank == 0:
                print(json.dumps({"transport_only": "passed", "flags": flags.tolist()}))
            return
        reference = _flash_attn_fwd(query, full_key, full_value)[0]
        if args.cp2_split_compute:
            if world_size != 2:
                raise ValueError("split-compute experiment requires CP2")
            remote = 1 - rank
            remote_start = sum(lengths[:remote])
            remote_stop = remote_start + lengths[remote]
            remote_key = full_key[:, remote_start:remote_stop].contiguous()
            remote_value = full_value[:, remote_start:remote_stop].contiguous()
            partial_out = torch.empty(
                (2, *query.shape), dtype=query.dtype, device=query.device
            )
            partial_lse = torch.empty(
                (2, query.shape[0], query.shape[2], query.shape[1]),
                dtype=torch.float32,
                device=query.device,
            )

            def partial_compute() -> torch.Tensor:
                _flash_attn_fwd(
                    query,
                    key,
                    value,
                    return_lse=True,
                    out=partial_out[0],
                    lse=partial_lse[0],
                )
                _flash_attn_fwd(
                    query,
                    remote_key,
                    remote_value,
                    return_lse=True,
                    out=partial_out[1],
                    lse=partial_lse[1],
                )
                return partial_out[1]

            def merge_partials(
                outputs: torch.Tensor, lses: torch.Tensor
            ) -> torch.Tensor:
                merged_lse = torch.logaddexp(lses[0], lses[1])
                local_weight = torch.exp(lses[0] - merged_lse)
                remote_weight = torch.exp(lses[1] - merged_lse)
                return (
                    outputs[0].float() * local_weight.transpose(-2, -1).unsqueeze(-1)
                    + outputs[1].float() * remote_weight.transpose(-2, -1).unsqueeze(-1)
                ).to(outputs.dtype)

            compiled_merge = torch.compile(merge_partials, dynamic=False)

            def split_compute() -> torch.Tensor:
                partial_compute()
                return compiled_merge(partial_out, partial_lse)

            actual = split_compute()
            torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-3)
            result = {
                "partial_attention_only": _measure(
                    partial_compute,
                    warmup=args.warmup,
                    iterations=args.iterations,
                ),
                "cp2_split_compute": _measure(
                    split_compute,
                    warmup=args.warmup,
                    iterations=args.iterations,
                ),
                "max_abs_error": float((actual - reference).abs().max()),
            }
            if rank == 0:
                encoded = json.dumps(result, indent=2)
                print(encoded)
                if args.output is not None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(encoded + "\n")
            return
        if args.ab_ready_mask:
            if args.legacy_stripes or args.use_sm:
                raise ValueError(
                    "ready-mask A/B requires source chunks and Copy Engines"
                )

            def variant(chunks: int, scheduler: str) -> Callable[[], torch.Tensor]:
                def call() -> torch.Tensor:
                    fused.chunk_count = chunks
                    fused.scheduler = scheduler
                    output = fused(query, key, value, global_sequence=global_sequence)
                    return output

                return call

            calls = {
                "c1_fixed": variant(1, "fixed"),
                "c2_fixed": variant(2, "fixed"),
                "c2_ready_mask": variant(2, "ready_mask"),
            }
            for call in calls.values():
                output = call()
                torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-3)
            timing = _measure_interleaved(
                calls,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            if rank == 0:
                print(json.dumps({"interleaved_ab": timing}, indent=2))
            return
        if args.prefetched_kernel:
            prefetched_epoch = 1
            (
                streamed_key,
                streamed_value,
                flags,
                epoch_guard,
                done,
            ) = fused._transport.begin_full_mesh_stream_kv(
                key,
                value,
                lengths,
                key_tag="chitu_full_mesh_key",
                value_tag="chitu_full_mesh_value",
                flag_tag="chitu_full_mesh_flags",
                epoch=prefetched_epoch,
                use_sm=args.use_sm,
                source_chunk=not args.legacy_stripes,
                chunk_count=args.chunks,
            )
            torch.cuda.current_stream().wait_event(done)
            torch.cuda.synchronize()
            epoch_tensor = torch.tensor(
                [prefetched_epoch, *fused._slot_starts(lengths)],
                dtype=torch.int32,
                device=query.device,
            )

            background_epoch = [0]

            def prefetched_call() -> torch.Tensor:
                # --with-transport moves the same volume over a private tag while
                # the kernel reads an already-complete buffer. The kernel can
                # therefore never wait, so whatever the timing loses is the raw
                # cost of running the copies next to the compute.
                if args.with_transport:
                    background_epoch[0] += 1
                    (
                        _,
                        _,
                        _,
                        background_guard,
                        background_done,
                    ) = fused._transport.begin_full_mesh_stream_kv(
                        key,
                        value,
                        lengths,
                        key_tag="chitu_full_mesh_background_key",
                        value_tag="chitu_full_mesh_background_value",
                        flag_tag="chitu_full_mesh_background_flags",
                        epoch=background_epoch[0],
                        use_sm=args.use_sm,
                        source_chunk=not args.legacy_stripes,
                        chunk_count=args.chunks,
                    )
                output = fused._flash_attn_fwd(
                    query,
                    streamed_key,
                    streamed_value,
                    causal=False,
                    _streaming_kv=True,
                    _kv_ready_flags=flags.view(-1),
                    _kv_slots=world_size * args.chunks,
                    _kv_epoch_tensor=epoch_tensor,
                    _kv_world_size=-1 if args.scheduler == "ready_mask" else 1,
                )[0]
                if args.with_transport:
                    torch.cuda.current_stream().wait_event(background_done)
                    fused._transport.publish_full_mesh_consumed(
                        flag_tag="chitu_full_mesh_background_flags",
                        epoch=background_epoch[0],
                        epoch_guard=background_guard,
                    )
                return output

            actual = prefetched_call()
            torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-3)
            timing = _measure(
                prefetched_call,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            fused._transport.publish_full_mesh_consumed(
                flag_tag="chitu_full_mesh_flags",
                epoch=prefetched_epoch,
                epoch_guard=epoch_guard,
            )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "prefetched_kernel": timing,
                            "chunk_count": args.chunks,
                            "max_abs_error": float((actual - reference).abs().max()),
                        },
                        indent=2,
                    )
                )
            return
        actual = fused(query, key, value, global_sequence=global_sequence)
        torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-3)
        compute_timing = _measure(
            lambda: _flash_attn_fwd(query, full_key, full_value)[0],
            warmup=args.warmup,
            iterations=args.iterations,
        )
        transport_epoch = [0]

        def transport_only() -> torch.Tensor:
            # Private tags and epoch counter: this measures the transport alone,
            # so it must not disturb the buffers the fused path rotates through.
            transport_epoch[0] += 1
            streamed_key, _, _, epoch_guard, done = (
                fused._transport.begin_full_mesh_stream_kv(
                    key,
                    value,
                    lengths,
                    key_tag="chitu_full_mesh_key",
                    value_tag="chitu_full_mesh_value",
                    flag_tag="chitu_full_mesh_flags",
                    epoch=transport_epoch[0],
                    use_sm=args.use_sm,
                    source_chunk=not args.legacy_stripes,
                    chunk_count=args.chunks,
                )
            )
            torch.cuda.current_stream().wait_event(done)
            # No attention kernel consumes this payload, so acknowledge it here.
            # Peers hold their next send until every peer has acknowledged, and
            # an unacknowledged epoch stalls all of them.
            fused._transport.publish_full_mesh_consumed(
                flag_tag="chitu_full_mesh_flags",
                epoch=transport_epoch[0],
                epoch_guard=epoch_guard,
            )
            epoch_guard.record_stream(torch.cuda.current_stream())
            return streamed_key

        transport_timing = _measure_all_streams(
            transport_only,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        timing = _measure(
            lambda: fused(query, key, value, global_sequence=global_sequence),
            warmup=args.warmup,
            iterations=args.iterations,
        )
        result = {
            "device": torch.cuda.get_device_name(),
            "world_size": world_size,
            "local_sequence": local_sequence,
            "global_sequence": global_sequence,
            "shard_lengths": list(lengths),
            "batch": args.batch,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "transport": "sm_direct_write" if args.use_sm else "copy_engine",
            "layout": "phase_stripes" if args.legacy_stripes else "source_chunks",
            "chunk_count": args.chunks,
            "fused_full_mesh": timing,
            "compute_only": compute_timing,
            "transport_only": transport_timing,
            "hidden_percent": (
                1
                - max(
                    0.0,
                    timing["median_ms"] - compute_timing["median_ms"],
                )
                / transport_timing["median_ms"]
            )
            * 100,
            "max_abs_error": float((actual - reference).abs().max()),
        }
        if rank == 0:
            encoded = json.dumps(result, indent=2)
            print(encoded)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(encoded + "\n")
    finally:
        fused.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
