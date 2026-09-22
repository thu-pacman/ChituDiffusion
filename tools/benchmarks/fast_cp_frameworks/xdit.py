"""Benchmark xDiT Ulysses on the common QKV-to-O boundary."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from tools.benchmarks.fast_cp_frameworks.common import (
    build_parser,
    make_inputs,
    run_benchmark,
    validate_args,
)


@torch.inference_mode()
def main() -> None:
    args = build_parser(framework="xdit", methods=("ulysses", "usp")).parse_args()
    validate_args(args)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if args.method == "ulysses":
        ulysses_degree, ring_degree = world, 1
    else:
        # xDiT's own hybrid-attention test uses U=2 and assigns the remaining
        # ranks to ring parallelism. At CP2 this validly degenerates to Ulysses.
        ulysses_degree = min(2, world)
        ring_degree = world // ulysses_degree
    if args.heads % ulysses_degree:
        raise ValueError("xDiT USP requires heads divisible by Ulysses degree")

    import xfuser.core.distributed.runtime_state as runtime_state_module
    import xfuser.model_executor.layers.usp as usp_module
    from xfuser.config import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.attention_backend import AttentionBackendType
    from xfuser.model_executor.layers.usp import USP

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(
        sequence_parallel_degree=world,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        backend="nccl",
    )
    # xDiT's ExternalRuntimeState builds a CP1 config by default. Supply the
    # groups already initialized above when benchmarking USP as a component.
    requested_backend = os.environ.get("FAST_CP_XDIT_ATTENTION_BACKEND")
    runtime_args = xFuserArgs(
        dit_parallel_size=world,
        vae_parallel_size=0,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        attention_backend=requested_backend,
    )
    runtime_state_module.xFuserArgs = lambda: runtime_args
    initialize_runtime_state()
    if requested_backend:
        get_runtime_state().set_attention_backend(
            AttentionBackendType[requested_backend.upper()]
        )
    attention_backend = get_runtime_state().attention_backend.name
    device = torch.device("cuda", local_rank)
    query, key, value = make_inputs(args, rank=rank, world=world, device=device)

    def call() -> torch.Tensor:
        return USP(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).contiguous()

    boundary = None
    if args.component == "communication":
        boundary = "native xDiT packed QKV input and output all-to-all only"

        def call() -> torch.Tensor:
            redistributed = usp_module._combined_qkv_all_to_all(
                query.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
            )
            return usp_module._ft_c_output_all_to_all(redistributed[0]).transpose(
                1, 2
            ).contiguous()

    elif args.component == "attention":
        boundary = "native xDiT attention only on the post-all-to-all shape"
        redistributed = usp_module._combined_qkv_all_to_all(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
        )
        attention_function = usp_module._get_attention_function()
        torch.cuda.synchronize(device)
        dist.barrier()

        def call() -> torch.Tensor:
            output, _ = attention_function(
                *redistributed,
                dropout_p=0.0,
                is_causal=False,
                attention_kwargs=None,
            )
            return output

    run_benchmark(
        framework="xdit",
        args=args,
        call=call,
        query=query,
        key=key,
        value=value,
        device=device,
        implementation={
            "parallelism": "Ulysses" if args.method == "ulysses" else "USP",
            "ulysses_degree": ulysses_degree,
            "ring_degree": ring_degree,
            "transport": (
                "xDiT functional NCCL all-to-all"
                if ring_degree == 1
                else "xDiT functional NCCL all-to-all + PyTorch templated ring attention"
            ),
            "attention_backend": f"xDiT {attention_backend}",
            "component": args.component,
        },
        verify=args.component == "e2e",
        boundary=boundary,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
