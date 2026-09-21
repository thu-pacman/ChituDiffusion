"""Weight-free distributed correctness and wrapper overhead regression.

Run with torchrun on one H20 NVLink node. JSON includes all timing samples.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

from chitu_diffusion.models.qwen_image.attention import QwenImageCpAttnProcessor
from chitu_diffusion.models.zimage.attention import ZImageCpAttnProcessor
from chitu_diffusion.parallel.cp.fast import (
    QwenImageAdapter,
    WanAdapter,
    ZImageAdapter,
    init_fast_cp,
)
from chitu_diffusion.parallel.cp.fast.api import _map
from tools.benchmarks.fast_cp_frameworks.chitu_joint import complex_rope
from tools.benchmarks.fast_cp_frameworks.common import measure

WAN_CASES = {
    "wan1.3b": {"sequence": 32760, "heads": 12},
    "wan14b": {"sequence": 75600, "heads": 40},
    # Small internal fixture used only for collective contract checks.
    "wan-unit": {"sequence": 4096, "heads": 12},
}


def gather(x):
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts, 1)


def compare(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0.01, rtol=0.03)
    else:
        for x, y in zip(a, b, strict=True):
            compare(x, y)


def make_case(model, sequence, device, replicated=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    if sequence <= 0 or sequence % world:
        raise ValueError("global sequence must be positive and divisible by CP degree")
    torch.manual_seed(20260921)
    heads = (
        WAN_CASES[model]["heads"]
        if model in WAN_CASES
        else {
            "qwen": 24,
            "zimage": 30,
        }[model]
    )
    width, local = heads * 128, sequence // world
    if model in WAN_CASES:
        module = (
            WanAttention(
                dim=width, heads=heads, dim_head=128, processor=WanAttnProcessor()
            )
            .to(device=device, dtype=torch.bfloat16)
            .eval()
        )
        adapter = WanAdapter()
    else:
        processor = (
            QwenImageCpAttnProcessor(None)
            if model == "qwen"
            else ZImageCpAttnProcessor(None)
        )
        module = (
            Attention(
                query_dim=width,
                heads=heads,
                dim_head=128,
                bias=model == "qwen",
                added_kv_proj_dim=width if model == "qwen" else None,
                context_pre_only=False if model == "qwen" else None,
                qk_norm="rms_norm",
                eps=1e-6,
                processor=processor,
            )
            .to(device=device, dtype=torch.bfloat16)
            .eval()
        )
        adapter = (
            QwenImageAdapter()
            if model == "qwen"
            else ZImageAdapter(local, "replicated" if replicated else "sharded")
        )
    image = torch.randn(
        1,
        local,
        width,
        device=device,
        dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(100 + rank),
    )
    rope = complex_rope(sequence, 128, device)
    sl = slice(rank * local, (rank + 1) * local)
    if model in WAN_CASES:
        full_rope = tuple(
            t.repeat_interleave(2, -1).view(1, sequence, 1, 128).to(torch.bfloat16)
            for t in (rope.real, rope.imag)
        )
        inputs = {
            "hidden_states": image,
            "rotary_emb": tuple(t[:, sl].contiguous() for t in full_rope),
        }

        def reference(module, data):
            return module(gather(data["hidden_states"]), rotary_emb=full_rope)[:, sl]
    else:
        text_count = 512 if replicated else 512 // world
        text = torch.randn(
            1,
            text_count,
            width,
            device=device,
            dtype=torch.bfloat16,
            generator=torch.Generator(device=device).manual_seed(
                700 if replicated else 700 + rank
            ),
        )
        text_rope = complex_rope(512, 128, device)
        ts = (
            slice(0, 512)
            if replicated
            else slice(rank * text_count, (rank + 1) * text_count)
        )
        if model == "qwen":
            inputs = {
                "hidden_states": image,
                "encoder_hidden_states": text,
                "image_rotary_emb": (rope[sl].contiguous(), text_rope[ts].contiguous()),
            }

            def reference(module, data):
                out = module(
                    gather(data["hidden_states"]),
                    encoder_hidden_states=gather(data["encoder_hidden_states"]),
                    image_rotary_emb=(rope, text_rope),
                )
                return out[0][:, sl], out[1][:, ts]
        else:
            inputs = {
                "hidden_states": torch.cat([image, text], 1),
                "freqs_cis": torch.cat([rope[sl], text_rope[ts]], 0).unsqueeze(0),
            }

            def reference(module, data):
                images = gather(data["hidden_states"][:, :local].contiguous())
                texts = data["hidden_states"][:, local:].contiguous()
                if not replicated:
                    texts = gather(texts)
                out = module(
                    torch.cat([images, texts], 1),
                    freqs_cis=torch.cat([rope, text_rope], 0).unsqueeze(0),
                )
                return torch.cat(
                    [out[:, sl], out[:, sequence + ts.start : sequence + ts.stop]], 1
                )

    return module, adapter, inputs, reference


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("wan1.3b", "wan14b", "qwen", "zimage"),
        default=["wan1.3b", "wan14b", "qwen", "zimage"],
    )
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    hostnames = [None] * world
    dist.all_gather_object(hostnames, socket.gethostname())
    report = {
        "status": "running",
        "cp": world,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostnames": hostnames,
        "cases": [],
    }
    # Match the production Wan benchmark's conservative slot estimate. Workspaces
    # are monotonic within one runtime, so reserve the sum for all selected shapes.
    wan_pool = sum(
        14 * 2 * WAN_CASES[model]["sequence"] * WAN_CASES[model]["heads"] * 128 * 2
        for model in args.models
        if model in WAN_CASES
    )
    pool_bytes = (4 << 30) + wan_pool
    report["pool_bytes"] = pool_bytes
    with init_fast_cp(
        dist.group.WORLD, device=device, pool_bytes=pool_bytes, max_workspaces=24
    ) as runtime:
        for model in args.models:
            sequence = (
                WAN_CASES[model]["sequence"] if model in WAN_CASES else args.sequence
            )
            for algorithm in ("agkv", "ulysses"):
                module, adapter, inputs, reference = make_case(model, sequence, device)
                region = runtime.wrap_attention(
                    module,
                    adapter=adapter,
                    algorithm=algorithm,
                    origin=lambda **data: reference(module, data),
                )
                bound = region.bind(inputs=inputs)
                assert bound.metadata["path"] == "fast", bound.metadata
                compare(bound.run(), reference(module, inputs))
                first_count = runtime.workspace_count
                # A second layer with distinct weights must share communication
                # slots, not packed weights, output scratch, or graph executables.
                second = copy.deepcopy(module)
                second.to_q.weight.add_(0.002)
                region2 = runtime.wrap_attention(
                    second, adapter=adapter, algorithm=algorithm
                )
                bound2 = region2.bind(inputs=inputs)
                assert runtime.workspace_count == first_count
                assert bound2.metadata["workspace"] == bound.metadata["workspace"]
                for name in ("projection", "image_projection", "text_projection"):
                    first_projection = getattr(bound.execution.processor, name, None)
                    if first_projection is not None:
                        second_projection = getattr(bound2.execution.processor, name)
                        assert (
                            first_projection.key.data_ptr()
                            == second_projection.key.data_ptr()
                        )
                        assert (
                            first_projection.value.data_ptr()
                            == second_projection.value.data_ptr()
                        )
                compare(bound2.run(), reference(second, inputs))
                compare(bound.run(), reference(module, inputs))
                for _ in range(3):
                    inputs["hidden_states"].mul_(0.97)
                    bound2.run()
                    bound.run()
                compare(bound.run(), reference(module, inputs))
                # Compare the exact same prepared executable with/without wrapper.
                calls = {
                    "direct": lambda: bound.execution(bound.inputs),
                    "bound": bound.run,
                }
                timing = {name: [] for name in calls}
                for repetition in range(4):
                    for name in (
                        ("direct", "bound")
                        if repetition % 2 == 0
                        else ("bound", "direct")
                    ):
                        timing[name].append(
                            measure(
                                calls[name],
                                warmup=3,
                                iterations=args.iterations,
                                repeats=1,
                                device=device,
                            )
                        )
                region.prepare(inputs)
                new_inputs = _map(inputs, lambda t: t.clone())
                owned = region(**new_inputs)
                saved = _map(owned, lambda t: t.clone())
                compare(owned, reference(module, new_inputs))
                new_inputs["hidden_states"].mul_(0.91)
                compare(region(**new_inputs), reference(module, new_inputs))
                compare(owned, saved)
                assert runtime.workspace_count == first_count
                if model in WAN_CASES and algorithm == "agkv":
                    short_sequence = (sequence // 2 // world) * world
                    _, _, shorter, short_reference = make_case(
                        model, short_sequence, device
                    )
                    region.prepare(shorter)
                    compare(region(**shorter), short_reference(module, shorter))
                    assert runtime.workspace_count == first_count + 1
                    region.prepare(inputs)
                    compare(region(**inputs), reference(module, inputs))
                    assert runtime.workspace_count == first_count + 1
                # Local input storage can vary; shapes require collective prepare.
                larger = dict(
                    new_inputs,
                    hidden_states=new_inputs["hidden_states"][:, :-1].contiguous(),
                )
                try:
                    region(**larger)
                except ValueError as exc:
                    assert "metadata changed" in str(exc)
                else:
                    raise AssertionError("changed shape was accepted")
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    try:
                        bound.run()
                    except RuntimeError as exc:
                        assert "construction stream" in str(exc)
                    else:
                        raise AssertionError("wrong stream was accepted")
                # A closed borrower must not tear down the owner runtime.
                with init_fast_cp(
                    dist.group.WORLD, device=device, backend=runtime.backend
                ) as borrower:
                    assert borrower.backend is runtime.backend
                compare(bound.run(), reference(module, inputs))
                region.invalidate()
                try:
                    bound.run()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("invalidated binding was accepted")
                region.close()
                region2.close()
                record = {
                    "model": model,
                    "algorithm": algorithm,
                    "sequence": sequence,
                    "heads": module.heads,
                    "head_dim": 128,
                    "correct": True,
                    "metadata": bound.metadata,
                    "workspace_count": first_count,
                    "timings": timing,
                }
                report["cases"].append(record)
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in record.items()
                                if k not in {"timings", "metadata"}
                            }
                        ),
                        flush=True,
                    )
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
        if "zimage" in args.models:
            module, adapter, inputs, reference = make_case(
                "zimage", args.sequence, device, replicated=True
            )
            for algorithm in ("agkv", "ulysses"):
                region = runtime.wrap_attention(
                    module, adapter=adapter, algorithm=algorithm
                )
                bound = region.bind(inputs=inputs)
                compare(bound.run(), reference(module, inputs))
                region.close()
            report["zimage_replicated_text_correct"] = True
        # Different signatures on ranks must fail before a collective builder.
        unit_sequence = WAN_CASES["wan-unit"]["sequence"]
        module, adapter, inputs, reference = make_case(
            "wan-unit", unit_sequence, device
        )
        region = runtime.wrap_attention(module, adapter=adapter)
        mismatched = dict(inputs)
        if rank == 0:
            mismatched["attention_mask"] = torch.ones(
                1, device=device, dtype=torch.bool
            )
        try:
            region.prepare(mismatched)
        except ValueError as exc:
            assert "signatures differ" in str(exc)
        else:
            raise AssertionError("rank disagreement accepted")
        # Capability failure on one rank selects origin on every rank before
        # any fast builder/collective is submitted.
        from chitu_diffusion.parallel.cp.fast.adapters.base import UnsupportedFastCP

        class RankLimitedWan(WanAdapter):
            def validate(self, module, inputs, algorithm):
                super().validate(module, inputs, algorithm)
                if rank == 0:
                    raise UnsupportedFastCP("injected preparation capability miss")

        fallback = runtime.wrap_attention(
            module,
            adapter=RankLimitedWan(),
            origin=lambda **data: reference(module, data),
        )
        fallback.prepare(inputs)
        assert fallback.metadata["path"] == "origin"
        compare(fallback(**inputs), reference(module, inputs))
        report["collective_origin_fallback_correct"] = True
    report["status"] = "complete"
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
