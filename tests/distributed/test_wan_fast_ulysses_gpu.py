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
from chitu_diffusion.parallel.cp import UlyssesTopology, create_ulysses_transport


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return ordered[index]


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


def _topology(world_size: int, rank: int, transport) -> UlyssesTopology:
    return UlyssesTopology(
        lane_ranks=tuple(range(world_size)),
        rank=rank,
        degree=world_size,
        process_group=dist.group.WORLD,
        transport=transport,
    )


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.benchmark
@torch.inference_mode()
def test_wan_1_3b_full_fast_ulysses_benchmark() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {2, 4} or not torch.cuda.is_available():
        pytest.skip("Wan 1.3B full-Ulysses benchmark requires CP2 or CP4")
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
    sequence = int(os.environ.get("CHITU_WAN_FAST_BENCH_SEQUENCE", "32760"))
    model_dim, heads, head_dim = 1536, 12, 128
    if sequence % world_size:
        pytest.skip("sequence length must be divisible by the CP world size")
    local_sequence = sequence // world_size
    generator = torch.Generator(device=device).manual_seed(20260805 + rank)
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
    torch_transport = create_ulysses_transport(
        "torch",
        process_group=dist.group.WORLD,
        device=device,
        fast_eligible=True,
    )
    fast_transport = create_ulysses_transport(
        "fast_ulysses",
        process_group=dist.group.WORLD,
        device=device,
        fast_eligible=True,
    )
    active = SimpleNamespace(width=world_size, process_group=dist.group.WORLD)
    results = {}
    outputs = {}
    try:
        methods = (
            (
                "agkv",
                WanCpAttnProcessor(
                    SimpleNamespace(
                        active=active,
                        active_agkv_transport=None,
                    ),
                    mode="agkv",
                ),
                None,
            ),
            (
                "torch_ulysses",
                WanCpAttnProcessor(
                    SimpleNamespace(
                        active=active,
                        active_ulysses=_topology(world_size, rank, torch_transport),
                    ),
                    mode="ulysses",
                ),
                None,
            ),
            (
                "fast_ulysses_sync",
                WanCpAttnProcessor(
                    SimpleNamespace(
                        active=active,
                        active_ulysses=_topology(world_size, rank, fast_transport),
                    ),
                    mode="ulysses",
                ),
                "0",
            ),
            (
                "fast_ulysses_full",
                WanCpAttnProcessor(
                    SimpleNamespace(
                        active=active,
                        active_ulysses=_topology(world_size, rank, fast_transport),
                    ),
                    mode="ulysses",
                ),
                "1",
            ),
        )
        for name, processor, async_ce in methods:
            if async_ce is not None:
                os.environ["CHITU_FAST_ULYSSES_ASYNC_CE"] = async_ce
            dist.barrier()
            output, local_samples = _measure(
                lambda: processor(
                    attention,
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=None,
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
                    "p95_ms": _percentile(samples, 0.95),
                }

        torch.testing.assert_close(
            outputs["torch_ulysses"],
            outputs["agkv"],
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            outputs["fast_ulysses_sync"],
            outputs["torch_ulysses"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            outputs["fast_ulysses_full"],
            outputs["fast_ulysses_sync"],
            rtol=0,
            atol=0,
        )
        if rank == 0:
            report = {
                "model": "Wan 1.3B",
                "world_size": world_size,
                "shape": [1, sequence, model_dim],
                "heads": heads,
                "head_dim": head_dim,
                "dtype": "bfloat16",
                "methods": results,
            }
            output_path = os.environ.get("CHITU_WAN_FAST_BENCH_OUTPUT")
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print("WAN_FAST_ULYSSES_RESULT=" + json.dumps(report, sort_keys=True))
    finally:
        fast_transport.close()
        torch_transport.close()
        dist.barrier()
        if owned_world:
            dist.destroy_process_group()
