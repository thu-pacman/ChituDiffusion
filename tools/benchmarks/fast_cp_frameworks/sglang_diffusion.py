"""Benchmark SGLang Diffusion Ulysses and K/V gather on a common boundary."""

from __future__ import annotations

import os
from types import SimpleNamespace

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
    parser = build_parser(framework="sglang", methods=("ulysses", "kv_gather"))
    parser.add_argument("--force-nccl", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    if args.force_nccl:
        os.environ["SGLANG_DIFFUSION_IPC_A2A"] = "0"
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.method == "ulysses" and args.heads % world:
        raise ValueError("SGLang Ulysses requires heads divisible by CP degree")

    from sglang.multimodal_gen.runtime.distributed.communication_op import (
        sequence_model_parallel_all_gather,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_input_all_to_all_qkv,
        _usp_output_all_to_all,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.multimodal_gen.runtime.utils.precision import (
        set_mixed_precision_policy,
    )

    set_global_server_args(
        SimpleNamespace(
            attention_backend=None,
            kv_gather_degree=world if args.method == "kv_gather" else 1,
            sp_split_auto=False,
        )
    )
    set_mixed_precision_policy(torch.bfloat16, torch.float32)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=world,
        ulysses_degree=world,
        ring_degree=1,
    )
    device = torch.device("cuda", local_rank)
    query, key, value = make_inputs(args, rank=rank, world=world, device=device)
    requested_backend = os.environ.get("FAST_CP_SGLANG_ATTENTION_BACKEND")
    required_backend = (
        AttentionBackendEnum[requested_backend.upper()]
        if requested_backend
        else None
    )
    attention = USPAttention(
        num_heads=args.heads,
        head_size=args.head_dim,
        dropout_rate=0.0,
        causal=False,
        required_attention_backend=required_backend,
    )

    def call() -> torch.Tensor:
        with set_forward_context(current_timestep=0, attn_metadata=None):
            return attention(query, key, value)

    boundary = None
    if args.component == "communication":
        if args.method == "kv_gather":
            boundary = "native SGLang K/V all-gather only"

            def call() -> tuple[torch.Tensor, torch.Tensor]:
                return (
                    sequence_model_parallel_all_gather(key, dim=1),
                    sequence_model_parallel_all_gather(value, dim=1),
                )

        else:
            boundary = "native SGLang packed QKV input and output all-to-all only"

            def call() -> torch.Tensor:
                redistributed = _usp_input_all_to_all_qkv(query, key, value)
                return _usp_output_all_to_all(redistributed[0], head_dim=2)

    elif args.component == "attention":
        if args.method == "kv_gather":
            compute_q = query
            compute_k = sequence_model_parallel_all_gather(key, dim=1)
            compute_v = sequence_model_parallel_all_gather(value, dim=1)
        else:
            compute_q, compute_k, compute_v = _usp_input_all_to_all_qkv(
                query, key, value
            )
        boundary = "native SGLang attention backend only on the post-communication shape"
        torch.cuda.synchronize(device)
        dist.barrier()

        def call() -> torch.Tensor:
            with set_forward_context(current_timestep=0, attn_metadata=None):
                return attention.attn_impl.forward(
                    compute_q, compute_k, compute_v, None
                )

    # IPC communicators are initialized lazily by the first attention call.
    call()
    transport = "NCCL"
    if args.method == "ulysses" and world == 2 and not args.force_nccl:
        try:
            from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
                IPC_A2A,
            )

            if IPC_A2A.inited:
                transport = "SGLang packed QKV NCCL input + CUDA IPC output"
        except ImportError:
            pass
    elif args.method == "ulysses":
        transport = "SGLang packed QKV NCCL input + NCCL output"
    run_benchmark(
        framework="sglang",
        args=args,
        call=call,
        query=query,
        key=key,
        value=value,
        device=device,
        implementation={
            "parallelism": "K/V gather" if args.method == "kv_gather" else "Ulysses",
            "transport": transport,
            "attention_backend": attention.backend.name,
            "component": args.component,
        },
        verify=args.component == "e2e",
        boundary=boundary,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
