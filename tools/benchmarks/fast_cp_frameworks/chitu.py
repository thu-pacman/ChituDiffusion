"""Benchmark Chitu NCCL and Fast CP on a common QKV-to-O boundary."""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

from chitu_diffusion.parallel.cp import UlyssesTopology
from chitu_diffusion.parallel.cp.agkv_transport import create_agkv_transport
from chitu_diffusion.parallel.cp.image_attention import ImageSelfAttention, _pad_heads
from chitu_diffusion.parallel.cp.ulysses_transport import create_ulysses_transport
from tools.benchmarks.fast_cp_frameworks.common import (
    build_parser,
    cudnn_attention,
    init_torch_distributed,
    make_inputs,
    run_benchmark,
    validate_args,
)

METHODS = ("nccl_agkv", "fast_agkv", "nccl_ulysses", "fast_ulysses")


@torch.inference_mode()
def main() -> None:
    args = build_parser(framework="chitu", methods=METHODS).parse_args()
    validate_args(args)
    rank, world, device = init_torch_distributed()
    query, key, value = make_inputs(args, rank=rank, world=world, device=device)
    mode = "agkv" if args.method.endswith("agkv") else "ulysses"
    transport_name = "fast_" + mode if args.method.startswith("fast") else "torch"
    attention = ImageSelfAttention(mode, planned=args.method.startswith("fast"))
    transport = None
    topology = None
    if mode == "agkv":
        transport = create_agkv_transport(
            transport_name,
            process_group=dist.group.WORLD,
            device=device,
            fast_eligible=True,
        )
    else:
        transport = create_ulysses_transport(
            transport_name,
            process_group=dist.group.WORLD,
            device=device,
            fast_eligible=True,
        )
        topology = UlyssesTopology(
            lane_ranks=tuple(range(world)),
            rank=rank,
            degree=world,
            process_group=dist.group.WORLD,
            transport=transport,
        )

    def call() -> torch.Tensor:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return attention(
                query,
                key,
                value,
                lane_process_group=dist.group.WORLD,
                ulysses_topology=topology,
                agkv_transport=transport if mode == "agkv" else None,
            )

    boundary = None
    if args.component == "communication":
        boundary = "native CP communication only on pre-projected Q/K/V"
        if mode == "agkv":

            def call() -> tuple[torch.Tensor, torch.Tensor]:
                return transport.all_gather_kv(key, value)

        else:
            padded = tuple(
                _pad_heads(tensor, world)[0] for tensor in (query, key, value)
            )

            def call() -> torch.Tensor:
                transport.reset()
                redistributed = transport.all_to_all_qkv(*padded)
                return transport.all_to_all(redistributed[0], 1, 2)

    elif args.component == "attention":
        boundary = "native cuDNN attention only on the post-communication local shape"
        if mode == "agkv":
            full_key, full_value = transport.all_gather_kv(key, value)

            def call() -> torch.Tensor:
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    return attention.finish_agkv(query, full_key, full_value)

        else:
            padded = tuple(
                _pad_heads(tensor, world)[0] for tensor in (query, key, value)
            )
            transport.reset()
            redistributed = transport.all_to_all_qkv(*padded)

            def call() -> torch.Tensor:
                return cudnn_attention(*redistributed)

        torch.cuda.synchronize(device)
        dist.barrier()

    try:
        # Initialize lazy transports, transfer selection, and the planner before
        # constructing the report metadata. run_benchmark still performs its own
        # untimed correctness call before warmup.
        call()
        torch.cuda.synchronize(device)
        dist.barrier()
        attention_plan = attention.attention_plan
        if attention_plan is None:
            attention_backend = "PyTorch cuDNN SDPA"
        elif attention_plan.strategy == "cudnn":
            attention_backend = "Chitu planner: PyTorch cuDNN SDPA"
        else:
            attention_backend = "Chitu planner: CuTe FlashAttention"
        run_benchmark(
            framework="chitu",
            args=args,
            call=call,
            query=query,
            key=key,
            value=value,
            device=device,
            implementation={
                "parallelism": mode,
                "transport": "NVSHMEM Fast CP" if args.method.startswith("fast") else "NCCL",
                "attention_backend": attention_backend,
                "attention_plan": (
                    asdict(attention_plan) if attention_plan is not None else None
                ),
                "attention_cuda_graph": attention_plan is not None,
                "transfer_paths": getattr(transport, "selected_transfers", {}),
                "transfer_tuning_ms": getattr(
                    transport, "transfer_tuning_ms", {}
                ),
                "component": args.component,
            },
            verify=args.component == "e2e",
            boundary=boundary,
        )
    finally:
        if transport is not None:
            transport.close()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
