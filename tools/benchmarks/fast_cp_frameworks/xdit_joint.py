"""Weight-free Qwen-Image and Z-Image attention through xDiT processors."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
)
from diffusers.models.transformers.transformer_z_image import (
    ZSingleStreamAttnProcessor,
)

from tools.benchmarks.fast_cp_frameworks.common import git_revision, measure

MODELS = {
    "qwenimage": {"heads": 24, "head_dim": 128, "hidden": 3072},
    "zimage": {"heads": 30, "head_dim": 128, "hidden": 3840},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--text-sequence", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def complex_rope(sequence: int, head_dim: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(sequence, dtype=torch.float32, device=device)
    inverse_frequency = 1.0 / (
        10000
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )
    return torch.polar(
        torch.ones((sequence, head_dim // 2), dtype=torch.float32, device=device),
        torch.outer(positions, inverse_frequency),
    )


def rank_max(value: float, device: torch.device) -> float:
    result = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result)


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
        raise ValueError("global image sequence must be divisible by CP degree")
    if args.text_sequence <= 0 or args.text_sequence % world:
        raise ValueError("text sequence must be divisible by CP degree")
    if args.model == "qwenimage" and config["heads"] % world:
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "schema_version": 1,
                "status": "unsupported",
                "framework": "xdit",
                "model": args.model,
                "method": "ulysses",
                "reason": (
                    "xDiT Qwen Ulysses requires heads divisible by CP degree: "
                    f"{config['heads']} % {world} != 0"
                ),
                "shape": {
                    "global_image_sequence": args.sequence,
                    "text_sequence": args.text_sequence,
                    "heads": config["heads"],
                    "head_dim": config["head_dim"],
                    "hidden": config["hidden"],
                    "cp_degree": world,
                },
            }
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        dist.barrier()
        dist.destroy_process_group()
        return

    from xfuser.model_executor.models.transformers.transformer_qwen import (
        xFuserQwenDoubleStreamAttnProcessor,
    )
    from xfuser.model_executor.models.transformers.transformer_z_image import (
        xFuserZSingleStreamAttnProcessor,
    )

    torch.manual_seed(args.seed)
    if args.model == "qwenimage":
        cp_processor = xFuserQwenDoubleStreamAttnProcessor()
        native_processor = QwenDoubleStreamAttnProcessor2_0()
        attention = Attention(
            query_dim=config["hidden"],
            added_kv_proj_dim=config["hidden"],
            dim_head=config["head_dim"],
            heads=config["heads"],
            out_dim=config["hidden"],
            context_pre_only=False,
            bias=True,
            processor=cp_processor,
            qk_norm="rms_norm",
            eps=1e-6,
        )
    else:
        cp_processor = xFuserZSingleStreamAttnProcessor()
        native_processor = ZSingleStreamAttnProcessor()
        attention = Attention(
            query_dim=config["hidden"],
            dim_head=config["head_dim"],
            heads=config["heads"],
            qk_norm="rms_norm",
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=cp_processor,
        )
    attention = attention.to(device=device, dtype=torch.bfloat16).eval()

    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    full_image = torch.randn(
        1,
        args.sequence,
        config["hidden"],
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    full_text = torch.randn(
        1,
        args.text_sequence,
        config["hidden"],
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    image_rope = complex_rope(args.sequence, config["head_dim"], device)
    text_rope = complex_rope(args.text_sequence, config["head_dim"], device)
    image_start = rank * (args.sequence // world)
    image_stop = (rank + 1) * (args.sequence // world)
    text_start = rank * (args.text_sequence // world)
    text_stop = (rank + 1) * (args.text_sequence // world)

    if args.model == "qwenimage":
        local_image = full_image[:, image_start:image_stop].clone()
        local_text = full_text[:, text_start:text_stop].clone()
        local_rope = (
            image_rope[image_start:image_stop],
            text_rope[text_start:text_stop],
        )

        def call() -> tuple[torch.Tensor, torch.Tensor]:
            return attention(
                local_image,
                encoder_hidden_states=local_text,
                image_rotary_emb=local_rope,
            )

        attention.set_processor(native_processor)
        expected_image, expected_text = attention(
            full_image,
            encoder_hidden_states=full_text,
            image_rotary_emb=(image_rope, text_rope),
        )
        attention.set_processor(cp_processor)
        actual_image, actual_text = call()
        expected_image = expected_image[:, image_start:image_stop]
        expected_text = expected_text[:, text_start:text_stop]
        actual_flat = torch.cat(
            (actual_image.float().flatten(), actual_text.float().flatten())
        )
        expected_flat = torch.cat(
            (expected_image.float().flatten(), expected_text.float().flatten())
        )
        torch.testing.assert_close(
            actual_image, expected_image, rtol=args.rtol, atol=args.atol
        )
        torch.testing.assert_close(
            actual_text, expected_text, rtol=args.rtol, atol=args.atol
        )
        partition = "image and text streams sharded independently"
    else:
        full_hidden = torch.cat((full_image, full_text), dim=1)
        full_rope = torch.cat((image_rope, text_rope), dim=0).unsqueeze(0)
        global_sequence = args.sequence + args.text_sequence
        if global_sequence % world:
            raise ValueError("unified image/text sequence must be divisible by CP degree")
        local_sequence = global_sequence // world
        token_start = rank * local_sequence
        token_stop = (rank + 1) * local_sequence
        local_hidden = full_hidden[:, token_start:token_stop].clone()
        local_rope = full_rope[:, token_start:token_stop].clone()

        def call() -> torch.Tensor:
            return attention(local_hidden, freqs_cis=local_rope)

        attention.set_processor(native_processor)
        expected_full = attention(full_hidden, freqs_cis=full_rope)
        attention.set_processor(cp_processor)
        actual = call()
        expected = expected_full[:, token_start:token_stop]
        actual_flat = actual.float().flatten()
        expected_flat = expected.float().flatten()
        torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
        partition = "unified image/text sequence sharded contiguously"

    delta = actual_flat - expected_flat
    accuracy = {
        "rtol": args.rtol,
        "atol": args.atol,
        "max_abs": rank_max(float(delta.abs().max()), device),
        "relative_l2": rank_max(
            float(delta.norm() / expected_flat.norm().clamp_min(1e-12)), device
        ),
    }
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
        "boundary": "native model joint-attention input through output projections",
        "semantics": "BF16 dense non-causal image/text joint attention",
        "shape": {
            "batch": 1,
            "global_image_sequence": args.sequence,
            "text_sequence": args.text_sequence,
            "heads": config["heads"],
            "head_dim": config["head_dim"],
            "hidden": config["hidden"],
            "dtype": "bfloat16",
            "cp_degree": world,
        },
        "implementation": {
            "parallelism": "xDiT Ulysses",
            "transport": "xDiT/yunchang NCCL all-to-all",
            "attention_backend": f"xDiT {attention_backend}",
            "rotary": True,
            "partition": partition,
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

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
