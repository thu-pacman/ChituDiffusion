from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


def _measure(call, *, warmup: int = 10, iterations: int = 30):
    for _ in range(warmup):
        call()
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


def _nccl_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    gathered = torch.empty(
        world_size * tensor.shape[0],
        *tensor.shape[1:],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(gathered, tensor.contiguous())
    return (
        gathered.unflatten(0, (world_size, tensor.shape[0]))
        .permute(1, 0, 2, 3, 4)
        .reshape(
            tensor.shape[0],
            world_size * tensor.shape[1],
            *tensor.shape[2:],
        )
        .contiguous()
    )


def _nccl_all_gather_kv(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _nccl_all_gather(key), _nccl_all_gather(value)


def _fast_agkv(
    group,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    key_tag: str,
    value_tag: str,
    use_ce: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.fast_ulysses.all_gather_kv_4d(
        group._group,
        key.contiguous(),
        value.contiguous(),
        key_tag,
        value_tag,
        use_ce,
    )


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.benchmark
@torch.inference_mode()
def test_fast_agkv_correctness_and_benchmark() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {2, 4, 8} or not torch.cuda.is_available():
        pytest.skip("Fast AGKV benchmark requires 2, 4, or 8 GPUs")
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

    from fast_ulysses import UlyssesGroup

    if not hasattr(torch.ops.fast_ulysses, "all_gather_kv_4d"):
        pytest.skip("installed fast_ulysses does not include Fast AGKV")

    sequence = int(os.environ.get("CHITU_FAST_AGKV_SEQUENCE", "4096"))
    heads = int(os.environ.get("CHITU_FAST_AGKV_HEADS", "24"))
    head_dim = int(os.environ.get("CHITU_FAST_AGKV_HEAD_DIM", "128"))
    if sequence % world_size:
        pytest.skip("sequence must be divisible by world size")
    local_sequence = sequence // world_size
    output_bytes = sequence * heads * head_dim * 2
    required_pool = 4 * output_bytes + (16 << 20)
    pool_bytes = ((required_pool + (64 << 20) - 1) // (64 << 20)) * (64 << 20)
    group = UlyssesGroup(
        process_group=dist.group.WORLD,
        device=device,
        initial_pool_bytes=pool_bytes,
    )
    try:
        # Multi-batch correctness also exercises nontrivial batch offsets.
        generator = torch.Generator(device=device).manual_seed(20260805 + rank)
        small_key, small_value = [
            torch.randn(
                2,
                8,
                heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            for _ in range(2)
        ]
        expected_key, expected_value = _nccl_all_gather_kv(small_key, small_value)
        fast_key, fast_value = _fast_agkv(
            group,
            small_key,
            small_value,
            key_tag="fast_agkv_small_key",
            value_tag="fast_agkv_small_value",
        )
        ce_key, ce_value = _fast_agkv(
            group,
            small_key,
            small_value,
            key_tag="fast_agkv_small_ce_key",
            value_tag="fast_agkv_small_ce_value",
            use_ce=True,
        )
        torch.testing.assert_close(fast_key, expected_key, rtol=0, atol=0)
        torch.testing.assert_close(fast_value, expected_value, rtol=0, atol=0)
        torch.testing.assert_close(ce_key, expected_key, rtol=0, atol=0)
        torch.testing.assert_close(ce_value, expected_value, rtol=0, atol=0)

        query, key, value = [
            torch.randn(
                1,
                local_sequence,
                heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            for _ in range(3)
        ]

        def attention(gather):
            full_key, full_value = gather()
            return F.scaled_dot_product_attention(
                query.transpose(1, 2),
                full_key.transpose(1, 2),
                full_value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        calls = {
            "nccl": lambda: _nccl_all_gather_kv(key, value),
            "fast_agkv": lambda: _fast_agkv(
                group,
                key,
                value,
                key_tag="fast_agkv_key",
                value_tag="fast_agkv_value",
            ),
            "fast_agkv_ce": lambda: _fast_agkv(
                group,
                key,
                value,
                key_tag="fast_agkv_ce_key",
                value_tag="fast_agkv_ce_value",
                use_ce=True,
            ),
            "nccl_attention": lambda: attention(
                lambda: _nccl_all_gather_kv(key, value)
            ),
            "fast_agkv_attention": lambda: attention(
                lambda: _fast_agkv(
                    group,
                    key,
                    value,
                    key_tag="fast_agkv_key",
                    value_tag="fast_agkv_value",
                )
            ),
            "fast_agkv_ce_attention": lambda: attention(
                lambda: _fast_agkv(
                    group,
                    key,
                    value,
                    key_tag="fast_agkv_ce_key",
                    value_tag="fast_agkv_ce_value",
                    use_ce=True,
                )
            ),
        }
        results = {}
        outputs = {}
        for name, call in calls.items():
            dist.barrier()
            output, local_samples = _measure(call)
            outputs[name] = (
                tuple(tensor.clone() for tensor in output)
                if isinstance(output, tuple)
                else output.clone()
            )
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
        for name in ("fast_agkv", "fast_agkv_ce"):
            torch.testing.assert_close(
                outputs[name][0],
                outputs["nccl"][0],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                outputs[name][1],
                outputs["nccl"][1],
                rtol=0,
                atol=0,
            )
        for name in ("fast_agkv_attention", "fast_agkv_ce_attention"):
            torch.testing.assert_close(
                outputs[name],
                outputs["nccl_attention"],
                rtol=0,
                atol=0,
            )
        if rank == 0:
            report = {
                "world_size": world_size,
                "global_shape": [1, sequence, heads, head_dim],
                "dtype": "bfloat16",
                "methods": results,
            }
            output_path = os.environ.get("CHITU_FAST_AGKV_OUTPUT")
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print("FAST_AGKV_RESULT=" + json.dumps(report, sort_keys=True))
    finally:
        group.destroy()
        dist.barrier()
        if owned_world:
            dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.distributed
@torch.inference_mode()
def test_fast_agkv_image_attention_integration() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {2, 4, 8} or not torch.cuda.is_available():
        pytest.skip("Fast AGKV integration requires 2, 4, or 8 GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    owned_world = not dist.is_initialized()
    if owned_world:
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local_rank),
        )
    device = torch.device("cuda", local_rank)

    from chitu_diffusion.parallel.cp import (
        ImageSelfAttention,
        create_agkv_transport,
    )

    transport = create_agkv_transport(
        "fast_agkv",
        process_group=dist.group.WORLD,
        device=device,
        fast_eligible=True,
    )
    try:
        generator = torch.Generator(device=device).manual_seed(
            20260806 + dist.get_rank()
        )
        query, key, value = [
            torch.randn(
                2,
                64,
                24,
                128,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            for _ in range(3)
        ]
        output = ImageSelfAttention(mode="agkv")(
            query,
            key,
            value,
            lane_process_group=dist.group.WORLD,
            agkv_transport=transport,
        )
        full_key, full_value = _nccl_all_gather_kv(key, value)
        expected = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            full_key.transpose(1, 2),
            full_value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        transport.close()
        dist.barrier()
        if owned_world:
            dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.distributed
@torch.inference_mode()
def test_a_cfg_halved_lane_owns_the_fast_transport() -> None:
    """CFG parallelism cuts the plane in two, and both halves stay on Fast AGKV.

    Each half is its own NVSHMEM runtime, which is legal because a rank belongs
    to exactly one of them.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {4, 8} or not torch.cuda.is_available():
        pytest.skip("a CFG-halved lane needs 4 or 8 GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    owned_world = not dist.is_initialized()
    if owned_world:
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local_rank),
        )
    device = torch.device("cuda", local_rank)

    from chitu_diffusion.parallel.cp.context import (
        EpeParallelContext,
        cfg_parallel_rank_groups,
    )
    from chitu_diffusion.parallel.cp.nccl.agkv import all_gather_sequence_pair

    half = world_size // 2
    context = EpeParallelContext.from_torchrun(
        allowed_widths=(1, half, world_size),
        ulysses_degree=half,
        owns_process_group=False,
        agkv_transport="fast_agkv",
        fast_lane_width=half,
    )
    try:
        rank = dist.get_rank()
        cp_groups, _ = cfg_parallel_rank_groups(tuple(range(world_size)))
        lane = next(group for group in cp_groups if rank in group)
        with context.activate(lane) as topology:
            transport = context.active_agkv_transport
            assert transport.name == "fast_agkv"
            generator = torch.Generator(device=device).manual_seed(20260902 + rank)
            key, value = [
                torch.randn(
                    1,
                    128,
                    8,
                    128,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                for _ in range(2)
            ]
            gathered_key, gathered_value = transport.all_gather_kv(key, value)
            expected_key, expected_value = all_gather_sequence_pair(
                key, value, topology.process_group
            )
            assert gathered_key.shape[1] == half * key.shape[1]
            torch.testing.assert_close(gathered_key, expected_key, rtol=0, atol=0)
            torch.testing.assert_close(gathered_value, expected_value, rtol=0, atol=0)
        # The plane the CFG branches were cut from is not the fast lane.
        with context.activate(tuple(range(world_size))):
            assert context.active_agkv_transport.name != "fast_agkv"
    finally:
        context.close()
        if owned_world and dist.is_initialized():
            dist.destroy_process_group()
