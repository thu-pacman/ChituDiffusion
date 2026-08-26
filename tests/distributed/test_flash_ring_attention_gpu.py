from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from chitu_diffusion.parallel.cp.fast._runtime import FastUlyssesAllToAll
from chitu_diffusion.parallel.cp.fast.experimental.ring import (
    FastRingAttention,
    FastRingKVTransport,
)


def _all_gather_sequence(tensor: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    gathered = torch.empty(
        world_size * tensor.shape[0],
        *tensor.shape[1:],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(gathered, tensor)
    return (
        gathered.unflatten(0, (world_size, tensor.shape[0]))
        .permute(1, 0, 2, 3, 4)
        .flatten(1, 2)
        .contiguous()
    )


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = query.shape[-1] ** -0.5
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * scale
    return (
        torch.einsum(
            "bhqk,bkhd->bqhd",
            torch.softmax(scores, dim=-1),
            value.float(),
        ).to(query.dtype),
        torch.logsumexp(scores, dim=-1),
    )


@pytest.mark.gpu
@pytest.mark.distributed
@torch.inference_mode()
def test_flash_ring_ce_correctness_dynamic_shapes_and_epoch_reset() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in {2, 4, 8} or not torch.cuda.is_available():
        pytest.skip("Flash Ring CE requires torchrun with CP2, CP4, or CP8")
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
    runtime = FastUlyssesAllToAll(
        dist.group.WORLD,
        device,
        pool_bytes=512 << 20,
    )
    transport = FastRingKVTransport(
        runtime,
        rank=rank,
        world_size=world_size,
        tag_prefix="flash_ring_gpu_test",
    )
    attention = FastRingAttention(transport)

    try:
        for call_index, shape in enumerate(
            ((1, 32, 4, 64), (2, 17, 6, 128), (1, 32, 4, 64))
        ):
            generator = torch.Generator(device=device).manual_seed(
                20260815 + rank * 100 + call_index
            )
            query = torch.randn(
                shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            key = torch.randn(
                shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            value = torch.randn(
                shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            global_key = _all_gather_sequence(key)
            global_value = _all_gather_sequence(value)
            reference_output, reference_lse = _reference(
                query,
                global_key,
                global_value,
            )

            actual_output, actual_lse = attention(
                query,
                key,
                value,
                return_lse=True,
            )
            torch.testing.assert_close(
                actual_output.float(),
                reference_output.float(),
                rtol=2e-2,
                atol=2e-2,
            )
            torch.testing.assert_close(
                actual_lse,
                reference_lse,
                rtol=2e-3,
                atol=2e-3,
            )
            # Two symmetric landing slots per tensor, independent of CP degree.
            state = runtime.begin_ring_ce_transport(
                key,
                value,
                key_tag="flash_ring_memory_gate_key",
                value_tag="flash_ring_memory_gate_value",
                state_tag="flash_ring_memory_gate_state",
                epoch=1,
            )
            assert state.local_key is key
            assert state.local_value is value
            assert state.landing_key.numel() == 2 * key.numel()
            assert state.landing_value.numel() == 2 * value.numel()
            # Drain this additional traversal so its CE streams cannot outlive
            # the test or block a future slot reuse.
            for phase in range(1, world_size):
                runtime.wait_ring_ce_ready(state, phase=phase)
                runtime.publish_ring_ce_consumed(state, phase=phase)
            torch.cuda.synchronize()
            dist.barrier()
    finally:
        runtime._group.destroy()
        dist.barrier()
        if owned_world:
            dist.destroy_process_group()
