"""Weight-free Qwen-Image and Z-Image native Chitu CP attention benchmark."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from diffusers.models.attention_processor import Attention

from chitu_diffusion.models.qwen_image.attention import QwenImageCpAttnProcessor
from chitu_diffusion.models.zimage.attention import ZImageCpAttnProcessor
from chitu_diffusion.parallel.cp import UlyssesTopology
from chitu_diffusion.parallel.cp.agkv_transport import create_agkv_transport
from chitu_diffusion.parallel.cp.fast.adapters.qwen_image import (
    prepare_qwen_image_sharded_agkv,
)
from chitu_diffusion.parallel.cp.fast.adapters.zimage import (
    PreparedZImageAgkv,
    PreparedZImageShardedAgkv,
)
from chitu_diffusion.parallel.cp.fast.runtime.backend import FastUlyssesAllToAll
from chitu_diffusion.parallel.cp.ulysses_transport import create_ulysses_transport
from tools.benchmarks.fast_cp_frameworks.common import git_revision, measure

MODELS = {
    "qwenimage": {"heads": 24, "head_dim": 128, "hidden": 3072},
    "zimage": {"heads": 30, "head_dim": 128, "hidden": 3840},
}
METHODS = ("nccl_agkv", "fast_agkv", "nccl_ulysses", "fast_ulysses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--text-sequence", type=int, default=512)
    parser.add_argument(
        "--z-sharded-text",
        action="store_true",
        help="Shard Z-Image caption tokens across CP ranks instead of replicating them",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def complex_rope(sequence: int, head_dim: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(sequence, dtype=torch.float32, device=device)
    inverse_frequency = 1.0 / (
        10000
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )
    angles = torch.outer(positions, inverse_frequency)
    return torch.polar(torch.ones_like(angles), angles)


def rank_max(value: float, device: torch.device) -> float:
    result = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if args.sequence <= 0 or args.sequence % world:
        raise ValueError("global image sequence must be divisible by CP degree")
    if args.model == "qwenimage" and args.text_sequence % world:
        raise ValueError("Qwen text sequence must be divisible by CP degree")
    if args.z_sharded_text and args.model != "zimage":
        raise ValueError("--z-sharded-text is only valid for Z-Image")
    if args.z_sharded_text and args.text_sequence % world:
        raise ValueError("sharded Z-Image text sequence must be divisible by CP degree")

    config = MODELS[args.model]
    mode = "agkv" if args.method.endswith("agkv") else "ulysses"
    transport_name = "fast_" + mode if args.method.startswith("fast") else "torch"
    transport = None
    topology = None
    if mode == "agkv" and args.method != "fast_agkv":
        transport = create_agkv_transport(
            transport_name,
            process_group=dist.group.WORLD,
            device=device,
            fast_eligible=True,
        )
    elif mode == "ulysses":
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
    parallel = SimpleNamespace(
        active=SimpleNamespace(
            width=world,
            process_group=dist.group.WORLD,
            rank_in_lane=rank,
        ),
        active_ulysses=topology,
        active_agkv_transport=transport if mode == "agkv" else None,
    )

    torch.manual_seed(args.seed)
    if args.model == "qwenimage":
        processor = QwenImageCpAttnProcessor(
            parallel,
            mode=mode,
            planned_attention=args.method == "fast_ulysses",
            sharded_text=True,
        )
        attention = Attention(
            query_dim=config["hidden"],
            added_kv_proj_dim=config["hidden"],
            dim_head=config["head_dim"],
            heads=config["heads"],
            out_dim=config["hidden"],
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
        )
    else:
        processor = ZImageCpAttnProcessor(
            parallel,
            mode=mode,
            planned_attention=args.method == "fast_ulysses",
            sharded_text=args.z_sharded_text,
        )
        attention = Attention(
            query_dim=config["hidden"],
            dim_head=config["head_dim"],
            heads=config["heads"],
            qk_norm="rms_norm",
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=processor,
        )
    attention = attention.to(device=device, dtype=torch.bfloat16).eval()

    local_sequence = args.sequence // world
    image_generator = torch.Generator(device=device).manual_seed(args.seed + 1 + rank)
    text_generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    image = torch.randn(
        1,
        local_sequence,
        config["hidden"],
        dtype=torch.bfloat16,
        device=device,
        generator=image_generator,
    )
    text = torch.randn(
        1,
        args.text_sequence,
        config["hidden"],
        dtype=torch.bfloat16,
        device=device,
        generator=text_generator,
    )
    image_rope = complex_rope(args.sequence, config["head_dim"], device)
    text_rope = complex_rope(args.text_sequence, config["head_dim"], device)
    token_start = rank * local_sequence
    local_image_rope = image_rope[token_start : token_start + local_sequence]
    text_start = rank * (args.text_sequence // world)
    text_stop = (rank + 1) * (args.text_sequence // world)
    local_text = text[:, text_start:text_stop].contiguous()
    local_text_rope = text_rope[text_start:text_stop].contiguous()

    image_parts = [torch.empty_like(image) for _ in range(world)]
    dist.all_gather(image_parts, image.contiguous())
    full_image = torch.cat(image_parts, dim=1)
    processor.disable()
    if args.model == "qwenimage":
        expected_image, expected_text = attention(
            full_image,
            encoder_hidden_states=text,
            image_rotary_emb=(image_rope, text_rope),
        )
        expected_image = expected_image[
            :, token_start : token_start + local_sequence
        ]
    else:
        z_text = local_text if args.z_sharded_text else text
        z_text_rope = local_text_rope if args.z_sharded_text else text_rope
        local_hidden = torch.cat([image, z_text], dim=1)
        local_rope = torch.cat([local_image_rope, z_text_rope], dim=0).unsqueeze(0)
        full_hidden = torch.cat([full_image, text], dim=1)
        full_rope = torch.cat([image_rope, text_rope], dim=0).unsqueeze(0)
        expected = attention(full_hidden, freqs_cis=full_rope)
        expected = torch.cat(
            [
                expected[:, token_start : token_start + local_sequence],
                (
                    expected[:, args.sequence + text_start : args.sequence + text_stop]
                    if args.z_sharded_text
                    else expected[:, args.sequence :]
                ),
            ],
            dim=1,
        )

    prepared_runtime = None
    prepared_slot = None
    prepare_ms = None
    if args.method == "fast_agkv":
        pool_bytes = (
            14 * 2 * (args.sequence + args.text_sequence) * config["hidden"] * 2
            + (64 << 20)
        )
        prepared_runtime = FastUlyssesAllToAll(
            dist.group.WORLD,
            device,
            pool_bytes=pool_bytes,
        )
        if args.model == "qwenimage":
            schedule = os.environ.get("CHITU_QWEN_AGKV_SCHEDULE", "auto")
            prepare_start = time.perf_counter()
            prepared_slot = prepare_qwen_image_sharded_agkv(
                prepared_runtime,
                attention,
                image,
                local_text,
                (local_image_rope, local_text_rope),
                schedule=schedule,
                tag=f"qwen_s{args.sequence}_cp{world}",
            )
            torch.cuda.synchronize(device)
            prepare_ms = rank_max((time.perf_counter() - prepare_start) * 1000, device)
        else:
            if args.z_sharded_text:
                prepared_slot = PreparedZImageShardedAgkv(
                    prepared_runtime,
                    attention,
                    local_hidden,
                    local_rope,
                    projection_backend="cublas",
                    tag=f"zimage_sharded_s{args.sequence}_cp{world}",
                )
            else:
                prepared_slot = PreparedZImageAgkv(
                    prepared_runtime,
                    attention,
                    local_hidden,
                    local_sequence,
                    local_rope,
                    projection_backend="cublas",
                    tag=f"zimage_s{args.sequence}_cp{world}",
                )
        attention.set_processor(prepared_slot)
    elif args.model == "qwenimage":
        processor.enable()
    else:
        processor.enable(local_sequence)

    if args.model == "qwenimage":

        def call():
            return attention(
                image,
                encoder_hidden_states=local_text,
                image_rotary_emb=(local_image_rope, local_text_rope),
            )

        actual_image, actual_text = call()
        actual_flat = torch.cat(
            [actual_image.float().flatten(), actual_text.float().flatten()]
        )
        expected_flat = torch.cat(
            [
                expected_image.float().flatten(),
                expected_text[:, text_start:text_stop].float().flatten(),
            ]
        )
        torch.testing.assert_close(
            actual_image, expected_image, rtol=args.rtol, atol=args.atol
        )
        torch.testing.assert_close(
            actual_text,
            expected_text[:, text_start:text_stop],
            rtol=args.rtol,
            atol=args.atol,
        )
    else:

        def call():
            return attention(local_hidden, freqs_cis=local_rope)

        actual = call()
        actual_flat = actual.float().flatten()
        expected_flat = expected.float().flatten()
        torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)

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
        "framework": "chitu",
        "model": args.model,
        "method": args.method,
        "boundary": "native model joint-attention input through output projections",
        "semantics": "BF16 dense non-causal image/text joint attention",
        "shape": {
            "batch": 1,
            "global_image_sequence": args.sequence,
            "local_image_sequence": local_sequence,
            "text_sequence": args.text_sequence,
            "heads": config["heads"],
            "head_dim": config["head_dim"],
            "hidden": config["hidden"],
            "dtype": "bfloat16",
            "cp_degree": world,
        },
        "implementation": {
            "parallelism": mode,
            "transport": (
                "NVSHMEM Fast CP" if args.method.startswith("fast") else "NCCL"
            ),
            "attention_backend": (
                "Chitu KV-split attention planner"
                if args.method.startswith("fast")
                else "Diffusers/PyTorch native SDPA dispatch"
            ),
            "rotary": True,
            "replicated_text": args.model != "qwenimage" and not args.z_sharded_text,
            "partition": (
                "image and text sequence-sharded"
                if args.model == "qwenimage" or args.z_sharded_text
                else "image-sharded, replicated text"
            ),
            "optimized_components": (
                [
                    "NVSHMEM direct-slot image KV",
                    (
                        "CE all-gather/Q and text projection overlap"
                        if prepared_slot.metadata["kv_transport_q_overlap"]
                        else "packed QKV before CE all-gather"
                    ),
                    "KV-split attention planner",
                    "CUDA graph compute segments",
                ]
                if prepared_slot is not None
                else [
                    "serial NVSHMEM",
                    "SM/TMA/CE transfer selection",
                    "KV-split attention planner",
                ]
                if args.method == "fast_ulysses"
                else []
            ),
            "attention_plan": (
                prepared_slot.metadata["attention_plan"]
                if prepared_slot is not None
                else asdict(processor.attention.attention_plan)
                if args.model == "qwenimage"
                and processor.attention.attention_plan is not None
                else asdict(processor.context_parallel.attention_plan)
                if args.model == "zimage"
                and processor.context_parallel.attention_plan is not None
                else None
            ),
            "transfer_paths": (
                transport.selected_transfers
                if args.method == "fast_ulysses"
                else {"all_gather": "ce"}
                if prepared_slot is not None
                else {}
            ),
            "transfer_tuning_ms": (
                transport.transfer_tuning_ms
                if args.method == "fast_ulysses"
                else {}
            ),
            "agkv_schedule": (
                prepared_slot.metadata.get("schedule", "overlap")
                if prepared_slot is not None
                else None
            ),
            "agkv_schedule_tuning_ms": (
                prepared_slot.metadata.get("schedule_tuning_ms", {})
                if prepared_slot is not None
                else {}
            ),
            "agkv_prepare_wall_ms": prepare_ms,
            "qkv_epilogue": (
                "paired QK norm/RoPE with joint QKV layout"
                if args.model == "qwenimage" and (
                    prepared_slot is None
                    or prepared_slot.metadata.get("schedule") == "packed"
                )
                else None
            ),
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

    if prepared_slot is not None:
        prepared_slot.close()
    if transport is not None:
        transport.close()
    if prepared_runtime is not None:
        prepared_runtime.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
