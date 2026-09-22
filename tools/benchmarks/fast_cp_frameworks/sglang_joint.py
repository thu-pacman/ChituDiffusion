"""Weight-free Qwen-Image and Z-Image attention through SGLang modules."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from tools.benchmarks.fast_cp_frameworks.common import git_revision, measure

MODELS = {
    "wan1.3b": {"heads": 12, "head_dim": 128, "hidden": 1536},
    "wan14b": {"heads": 40, "head_dim": 128, "hidden": 5120},
    "qwenimage": {"heads": 24, "head_dim": 128, "hidden": 3072},
    "zimage": {"heads": 30, "head_dim": 128, "hidden": 3840},
}
METHODS = ("ulysses", "kv_gather")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
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


def rope_cache(sequence: int, head_dim: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(sequence, dtype=torch.float32, device=device)
    inverse_frequency = 1.0 / (
        10000
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )
    angles = torch.outer(positions, inverse_frequency)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()


def initialize_parameters(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.ndim >= 2:
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
            elif name.endswith("bias"):
                parameter.zero_()
            else:
                parameter.fill_(1.0)


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
    config = MODELS[args.model]
    if args.sequence <= 0 or args.sequence % world:
        raise ValueError("global image sequence must be divisible by CP degree")
    if args.model in {"qwenimage", "zimage"} and (
        args.text_sequence <= 0 or args.text_sequence % world
    ):
        raise ValueError("text sequence must be divisible by CP degree")
    if args.method == "ulysses" and config["heads"] % world:
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "schema_version": 1,
                "status": "unsupported",
                "framework": "sglang",
                "model": args.model,
                "method": args.method,
                "reason": (
                    "SGLang Ulysses requires heads divisible by CP degree: "
                    f"{config['heads']} % {world} != 0"
                ),
                "shape": {
                    "global_image_sequence": args.sequence,
                    "text_sequence": (
                        0 if args.model.startswith("wan") else args.text_sequence
                    ),
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
        return

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
    from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
    from sglang.multimodal_gen.runtime.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
        apply_flashinfer_rope_qk_inplace,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
        QwenImageCrossAttention,
    )
    from sglang.multimodal_gen.runtime.models.dits.zimage import ZImageAttention
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

    if args.model.startswith("wan"):

        class WanSelfAttentionPath(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                hidden = config["hidden"]
                prefix = "benchmark.attention"
                self.heads = config["heads"]
                self.head_dim = config["head_dim"]
                self.to_q = ColumnParallelLinear(
                    hidden,
                    hidden,
                    bias=True,
                    gather_output=False,
                    prefix=f"{prefix}.to_q",
                )
                self.to_k = ColumnParallelLinear(
                    hidden,
                    hidden,
                    bias=True,
                    gather_output=False,
                    prefix=f"{prefix}.to_k",
                )
                self.to_v = ColumnParallelLinear(
                    hidden,
                    hidden,
                    bias=True,
                    gather_output=False,
                    prefix=f"{prefix}.to_v",
                )
                self.norm_q = RMSNorm(hidden, eps=1e-6)
                self.norm_k = RMSNorm(hidden, eps=1e-6)
                self.attn = USPAttention(
                    num_heads=self.heads,
                    head_size=self.head_dim,
                    dropout_rate=0.0,
                    causal=False,
                    prefix=f"{prefix}.attn",
                )
                self.to_out = RowParallelLinear(
                    hidden,
                    hidden,
                    bias=True,
                    input_is_parallel=True,
                    prefix=f"{prefix}.to_out",
                )

            def forward(
                self,
                hidden_states: torch.Tensor,
                cache: torch.Tensor,
                positions: torch.Tensor,
            ) -> torch.Tensor:
                query, _ = self.to_q(hidden_states)
                key, _ = self.to_k(hidden_states)
                value, _ = self.to_v(hidden_states)
                query = self.norm_q(query).unflatten(
                    2, (self.heads, self.head_dim)
                )
                key = self.norm_k(key).unflatten(2, (self.heads, self.head_dim))
                value = value.unflatten(2, (self.heads, self.head_dim))
                query, key = apply_flashinfer_rope_qk_inplace(
                    query,
                    key,
                    cache,
                    is_neox=False,
                    positions=positions,
                )
                output = self.attn(query, key, value).flatten(2)
                output, _ = self.to_out(output)
                return output

        attention = WanSelfAttentionPath()
    elif args.model == "qwenimage":
        attention = QwenImageCrossAttention(
            dim=config["hidden"],
            num_heads=config["heads"],
            head_dim=config["head_dim"],
            added_kv_proj_dim=config["hidden"],
            out_dim=config["hidden"],
            context_pre_only=False,
            qk_norm=True,
            eps=1e-6,
            prefix="benchmark.attention",
        )
    else:
        attention = ZImageAttention(
            dim=config["hidden"],
            num_heads=config["heads"],
            num_kv_heads=config["heads"],
            qk_norm=True,
            eps=1e-5,
            prefix="benchmark.attention",
        )
    attention = attention.to(device=device, dtype=torch.bfloat16).eval()
    initialize_parameters(attention, args.seed)

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
    image_start = rank * (args.sequence // world)
    image_stop = (rank + 1) * (args.sequence // world)
    text_start = rank * (args.text_sequence // world)
    text_stop = (rank + 1) * (args.text_sequence // world)
    image_cache = rope_cache(args.sequence, config["head_dim"], device)
    text_cache = rope_cache(args.text_sequence, config["head_dim"], device)

    if args.model.startswith("wan"):
        local_hidden = full_image[:, image_start:image_stop].clone()
        local_positions = torch.arange(image_start, image_stop, device=device).long()

        def call() -> torch.Tensor:
            with set_forward_context(current_timestep=0, attn_metadata=None):
                return attention(local_hidden, image_cache, local_positions)

        attention.attn.skip_sequence_parallel = True
        with set_forward_context(current_timestep=0, attn_metadata=None):
            expected_full = attention(
                full_image,
                image_cache,
                torch.arange(args.sequence, device=device).long(),
            )
        attention.attn.skip_sequence_parallel = False
        actual = call()
        expected = expected_full[:, image_start:image_stop]
        actual_flat = actual.float().flatten()
        expected_flat = expected.float().flatten()
        torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
        partition = "image sequence sharded contiguously"
        boundary = "extracted native Wan block self-attention through output projection"
        semantics = "BF16 dense non-causal Wan self-attention"
    elif args.model == "qwenimage":
        local_image = full_image[:, image_start:image_stop].clone()
        local_text = full_text[:, text_start:text_stop].clone()
        local_caches = (
            image_cache[image_start:image_stop],
            text_cache[text_start:text_stop],
        )

        def call() -> tuple[torch.Tensor, ...]:
            with set_forward_context(current_timestep=0, attn_metadata=None):
                return attention(
                    local_image,
                    local_text,
                    local_caches,
                    sp_text_sharded=True,
                )

        attention.attn.skip_sequence_parallel = True
        with set_forward_context(current_timestep=0, attn_metadata=None):
            expected_image, expected_text, _, _ = attention(
                full_image,
                full_text,
                (image_cache, text_cache),
                sp_text_sharded=False,
            )
        attention.attn.skip_sequence_parallel = False
        actual_image, actual_text, _, _ = call()
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
        boundary = "native model joint-attention input through output projections"
        semantics = "BF16 dense non-causal image/text joint attention"
    else:
        local_image = full_image[:, image_start:image_stop].clone()
        local_hidden = torch.cat((local_image, full_text), dim=1)
        global_cache = torch.cat((image_cache, text_cache), dim=0)
        local_positions = torch.cat(
            (
                torch.arange(image_start, image_stop, device=device),
                torch.arange(
                    args.sequence,
                    args.sequence + args.text_sequence,
                    device=device,
                ),
            )
        ).long()

        def call() -> torch.Tensor:
            with set_forward_context(current_timestep=0, attn_metadata=None):
                return attention(
                    local_hidden,
                    rope_cos_sin_cache=global_cache,
                    rope_positions=local_positions,
                    num_replicated_suffix=args.text_sequence,
                )

        attention.attn.skip_sequence_parallel = True
        full_hidden = torch.cat((full_image, full_text), dim=1)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            expected_full = attention(
                full_hidden,
                rope_cos_sin_cache=global_cache,
                rope_positions=torch.arange(
                    args.sequence + args.text_sequence, device=device
                ).long(),
            )
        attention.attn.skip_sequence_parallel = False
        actual = call()
        expected = torch.cat(
            (
                expected_full[:, image_start:image_stop],
                expected_full[:, args.sequence :],
            ),
            dim=1,
        )
        actual_flat = actual.float().flatten()
        expected_flat = expected.float().flatten()
        torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
        partition = "image sharded with replicated caption suffix"
        boundary = "native model joint-attention input through output projections"
        semantics = "BF16 dense non-causal image/text joint attention"

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
        "framework": "sglang",
        "model": args.model,
        "method": args.method,
        "boundary": boundary,
        "semantics": semantics,
        "shape": {
            "batch": 1,
            "global_image_sequence": args.sequence,
            "text_sequence": 0 if args.model.startswith("wan") else args.text_sequence,
            "heads": config["heads"],
            "head_dim": config["head_dim"],
            "hidden": config["hidden"],
            "dtype": "bfloat16",
            "cp_degree": world,
        },
        "implementation": {
            "parallelism": args.method,
            "transport": "SGLang native",
            "attention_backend": attention.attn.backend.name,
            "rotary": True,
            "partition": partition,
            "fixture": (
                "extracted native WanTransformerBlock self-attention path"
                if args.model.startswith("wan")
                else "native model attention module"
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

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
