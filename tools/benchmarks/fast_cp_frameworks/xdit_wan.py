"""Weight-free Wan attention benchmark through xDiT's native processor."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

from tools.benchmarks.fast_cp_frameworks.common import git_revision, measure

MODELS = {
    "wan1.3b": {"heads": 12, "head_dim": 128, "hidden": 1536},
    "wan14b": {"heads": 40, "head_dim": 128, "hidden": 5120},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rank_max(value: float, device: torch.device) -> float:
    result = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result)


def initialize_xdit(world: int, rank: int, local_rank: int) -> str:
    import xfuser.core.distributed.runtime_state as runtime_state_module
    from xfuser.config import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.attention_backend import AttentionBackendType

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(
        sequence_parallel_degree=world,
        ulysses_degree=world,
        ring_degree=1,
        backend="nccl",
    )
    requested_backend = os.environ.get("FAST_CP_XDIT_ATTENTION_BACKEND")
    runtime_args = xFuserArgs(
        dit_parallel_size=world,
        vae_parallel_size=0,
        ulysses_degree=world,
        ring_degree=1,
        attention_backend=requested_backend,
    )
    runtime_state_module.xFuserArgs = lambda: runtime_args
    initialize_runtime_state()
    if requested_backend:
        get_runtime_state().set_attention_backend(
            AttentionBackendType[requested_backend.upper()]
        )
    return get_runtime_state().attention_backend.name


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    attention_backend = initialize_xdit(world, rank, local_rank)

    config = MODELS[args.model]
    if args.sequence <= 0 or args.sequence % world:
        raise ValueError("global sequence must be positive and divisible by CP degree")
    if config["heads"] % world:
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "schema_version": 1,
                "status": "unsupported",
                "framework": "xdit",
                "model": args.model,
                "method": "ulysses",
                "reason": (
                    f"xDiT Ulysses requires heads divisible by CP degree: "
                    f"{config['heads']} % {world} != 0"
                ),
                "shape": {
                    "global_sequence": args.sequence,
                    "heads": config["heads"],
                    "head_dim": config["head_dim"],
                    "hidden": config["hidden"],
                    "cp_degree": world,
                },
            }
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(json.dumps(report, sort_keys=True), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    from xfuser.model_executor.models.transformers.transformer_wan import (
        xFuserWanAttnProcessor,
    )

    torch.manual_seed(args.seed)
    processor = xFuserWanAttnProcessor()
    attention = WanAttention(
        dim=config["hidden"],
        heads=config["heads"],
        dim_head=config["head_dim"],
        processor=processor,
    ).to(device=device, dtype=torch.bfloat16)
    attention.register_buffer(
        "head_perm",
        torch.arange(config["heads"], dtype=torch.long, device=device),
        persistent=False,
    )
    attention.eval()

    local_sequence = args.sequence // world
    generator = torch.Generator(device=device).manual_seed(args.seed + 1 + rank)
    hidden = torch.randn(
        1,
        local_sequence,
        config["hidden"],
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    positions = torch.arange(args.sequence, dtype=torch.float32, device=device)
    inverse_frequency = 1.0 / (
        10000
        ** (
            torch.arange(0, config["head_dim"], 2, device=device).float()
            / config["head_dim"]
        )
    )
    angles = torch.outer(positions, inverse_frequency).repeat_interleave(2, dim=-1)
    full_rotary = tuple(
        tensor.to(torch.bfloat16).view(1, args.sequence, 1, config["head_dim"])
        for tensor in (angles.cos(), angles.sin())
    )
    token_start = rank * local_sequence
    local_rotary = tuple(
        tensor[:, token_start : token_start + local_sequence]
        for tensor in full_rotary
    )

    def call() -> torch.Tensor:
        return attention(hidden, rotary_emb=local_rotary)

    hidden_parts = [torch.empty_like(hidden) for _ in range(world)]
    dist.all_gather(hidden_parts, hidden.contiguous())
    full_hidden = torch.cat(hidden_parts, dim=1)
    attention.set_processor(WanAttnProcessor())
    expected_full = attention(full_hidden, rotary_emb=full_rotary)
    expected = expected_full[:, token_start : token_start + local_sequence]
    attention.set_processor(processor)
    actual = call()
    delta = actual.float() - expected.float()
    accuracy = {
        "rtol": args.rtol,
        "atol": args.atol,
        "max_abs": rank_max(float(delta.abs().max()), device),
        "relative_l2": rank_max(
            float(delta.norm() / expected.float().norm().clamp_min(1e-12)), device
        ),
    }
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
    timing = measure(
        call,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        device=device,
    )

    hostnames: list[str | None] = [None] * world
    dist.all_gather_object(hostnames, socket.gethostname())
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "framework": "xdit",
        "model": args.model,
        "method": "ulysses",
        "boundary": "native Wan attention input through output projection",
        "semantics": "BF16 dense non-causal Wan self-attention",
        "shape": {
            "batch": 1,
            "global_sequence": args.sequence,
            "local_sequence": local_sequence,
            "heads": config["heads"],
            "head_dim": config["head_dim"],
            "hidden": config["hidden"],
            "dtype": "bfloat16",
            "cp_degree": world,
        },
        "implementation": {
            "parallelism": "ulysses",
            "transport": "xDiT/yunchang NCCL all-to-all",
            "attention_backend": f"xDiT {attention_backend}",
            "rotary": True,
        },
        "accuracy": accuracy,
        "timing": timing,
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "framework_revision": git_revision(args.source_root),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "world_size": world,
            "nodes": sorted({hostname for hostname in hostnames if hostname}),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "arguments": vars(args) | {"output": str(args.output)},
    }
    report["arguments"]["source_root"] = (
        str(args.source_root) if args.source_root is not None else None
    )
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True), flush=True)

    del expected_full, full_hidden
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
