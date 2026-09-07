from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from diffusers.models.attention_processor import Attention

from chitu_diffusion.models.wan.attention import WanCpAttnProcessor
from chitu_diffusion.parallel.cp import create_agkv_transport


def _measure(call, *, warmup: int = 5, iterations: int = 20):
    for _ in range(warmup):
        output = call()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        output = call()
        stops[index].record()
    torch.cuda.synchronize()
    return output, [
        starts[index].elapsed_time(stops[index]) for index in range(iterations)
    ]


class _SyncAgkvProxy:
    def __init__(self, transport) -> None:
        self._transport = transport

    def all_gather_kv(self, key, value):
        return self._transport.all_gather_kv(key, value)


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.benchmark
@torch.inference_mode()
def test_wan_fast_agkv_overlap_benchmark() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {2, 4, 8} or not torch.cuda.is_available():
        pytest.skip("Wan Fast AGKV benchmark requires CP2, CP4, or CP8")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    owned_world = not dist.is_initialized()
    if owned_world:
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local_rank),
        )
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    os.environ["CHITU_FAST_AGKV_ASYNC"] = "on"
    sequence = int(os.environ.get("CHITU_WAN_FAST_AGKV_SEQUENCE", "4096"))
    if sequence % world_size:
        pytest.skip("sequence length must be divisible by world size")
    local_sequence = sequence // world_size
    model_dim, heads, head_dim = 1536, 12, 128
    generator = torch.Generator(device=device).manual_seed(20260806 + rank)
    hidden_states = torch.randn(
        1,
        local_sequence,
        model_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    rotary_cos = torch.ones(
        1,
        local_sequence,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    rotary_sin = torch.zeros_like(rotary_cos)
    attention = (
        Attention(
            query_dim=model_dim,
            heads=heads,
            kv_heads=heads,
            dim_head=head_dim,
            bias=True,
            out_bias=True,
            qk_norm="rms_norm_across_heads",
            eps=1e-6,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
        .requires_grad_(False)
    )
    torch_transport = create_agkv_transport(
        "torch",
        process_group=dist.group.WORLD,
        device=device,
        fast_eligible=True,
    )
    fast_transport = create_agkv_transport(
        "fast_agkv",
        process_group=dist.group.WORLD,
        device=device,
        fast_eligible=True,
    )
    active = SimpleNamespace(width=world_size, process_group=dist.group.WORLD)

    def processor(transport):
        return WanCpAttnProcessor(
            SimpleNamespace(
                active=active,
                active_agkv_transport=transport,
            ),
            mode="agkv",
        )

    methods = {
        "nccl_agkv": processor(torch_transport),
        "fast_agkv_sync": processor(_SyncAgkvProxy(fast_transport)),
        "fast_agkv_overlap": processor(fast_transport),
    }
    results = {}
    outputs = {}
    try:
        for name, method in methods.items():
            dist.barrier()
            output, local_samples = _measure(
                lambda method=method: method(
                    attention,
                    hidden_states,
                    rotary_emb=(rotary_cos, rotary_sin),
                )
            )
            outputs[name] = output.clone()
            gathered: list[list[float] | None] = [None] * world_size
            dist.all_gather_object(gathered, local_samples)
            if rank == 0:
                samples = [
                    sample
                    for rank_samples in gathered
                    if rank_samples is not None
                    for sample in rank_samples
                ]
                results[name] = {
                    "median_ms": statistics.median(samples),
                    "p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))],
                }
        for name in ("fast_agkv_sync", "fast_agkv_overlap"):
            torch.testing.assert_close(
                outputs[name],
                outputs["nccl_agkv"],
                rtol=0,
                atol=0,
            )
        if rank == 0:
            report = {
                "world_size": world_size,
                "global_sequence": sequence,
                "model_dim": model_dim,
                "heads": heads,
                "head_dim": head_dim,
                "methods": results,
            }
            output_path = os.environ.get("CHITU_WAN_FAST_AGKV_OUTPUT")
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print("WAN_FAST_AGKV_RESULT=" + json.dumps(report, sort_keys=True))
    finally:
        fast_transport.close()
        torch_transport.close()
        dist.barrier()
        if owned_world:
            dist.destroy_process_group()
