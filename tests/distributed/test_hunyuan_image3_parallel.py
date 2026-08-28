from __future__ import annotations

import os
import socket
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from chitu_diffusion.models.hunyuan_image3 import (
    HunyuanImage3ParallelRuntime,
    resolve_parallel_plan,
)
from chitu_diffusion.parallel.cp import (
    SequencePartition,
    heads_to_sequence,
    sequence_to_heads,
)
from chitu_diffusion.parallel.ep import (
    build_expert_parallel_topology,
    expert_parallel_moe,
)
from chitu_diffusion.parallel.ep import dispatch as ep_dispatch


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _expert_scales(num_experts: int) -> torch.Tensor:
    return torch.arange(1, num_experts + 1, dtype=torch.float32) * 0.5


def _dense_reference(
    states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    expected = torch.zeros_like(states)
    for token in range(states.shape[0]):
        for slot in range(indices.shape[1]):
            expert = int(indices[token, slot])
            expected[token] += states[token] * scales[expert] * weights[token, slot]
    return expected


def _expert_parallel_worker(rank: int, world_size: int, port: int) -> None:
    _init(rank, world_size, port)
    num_experts, tokens, hidden, topk = 8, 11, 6, 3
    topology = build_expert_parallel_topology(
        num_experts=num_experts,
        ranks=tuple(range(world_size)),
        rank=rank,
        process_group=dist.group.WORLD,
    )
    # Every rank routes the same replicated tokens, matching a context-parallel
    # shard that is mirrored across the tensor-parallel axis.
    generator = torch.Generator().manual_seed(1234)
    states = torch.randn(tokens, hidden, generator=generator)
    logits = torch.randn(tokens, num_experts, generator=generator)
    weights, indices = torch.topk(torch.softmax(logits, dim=-1), topk)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    scales = _expert_scales(num_experts)

    visited: list[int] = []

    def run_expert(expert_index: int, rows: torch.Tensor) -> torch.Tensor:
        assert topology.owns(expert_index)
        visited.append(expert_index)
        return rows * scales[expert_index]

    result = expert_parallel_moe(
        states,
        weights,
        indices,
        topology=topology,
        run_expert=run_expert,
    )
    torch.testing.assert_close(
        result, _dense_reference(states, weights, indices, scales)
    )
    assert all(topology.owns(index) for index in visited)
    dist.destroy_process_group()


def _uneven_token_worker(rank: int, world_size: int, port: int) -> None:
    _init(rank, world_size, port)
    num_experts, hidden, topk = 8, 4, 2
    topology = build_expert_parallel_topology(
        num_experts=num_experts,
        ranks=tuple(range(world_size)),
        rank=rank,
        process_group=dist.group.WORLD,
    )
    # Rank 0 owns no tokens at all; the collectives must still line up.
    tokens = 0 if rank == 0 else rank * 3
    generator = torch.Generator().manual_seed(50 + rank)
    states = torch.randn(tokens, hidden, generator=generator)
    if tokens:
        logits = torch.randn(tokens, num_experts, generator=generator)
        weights, indices = torch.topk(torch.softmax(logits, dim=-1), topk)
        weights = weights / weights.sum(dim=-1, keepdim=True)
    else:
        weights = torch.zeros(0, topk)
        indices = torch.zeros(0, topk, dtype=torch.long)
    scales = _expert_scales(num_experts)

    result = expert_parallel_moe(
        states,
        weights,
        indices,
        topology=topology,
        run_expert=lambda index, rows: rows * scales[index],
    )
    assert result.shape == states.shape
    torch.testing.assert_close(
        result, _dense_reference(states, weights, indices, scales)
    )
    dist.destroy_process_group()


def _shared_destination_worker(rank: int, world_size: int, port: int) -> None:
    _init(rank, world_size, port)
    num_experts, tokens, hidden, topk = 16, 5, 3, 4
    topology = build_expert_parallel_topology(
        num_experts=num_experts,
        ranks=tuple(range(world_size)),
        rank=rank,
        process_group=dist.group.WORLD,
    )
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(tokens, hidden, generator=generator)
    # Three of every token's experts live on rank 0 and one on rank 2, so the
    # dispatch must send two rows per token rather than one per routed expert.
    indices = torch.tensor([[0, 1, 2, 8]]).repeat(tokens, 1)
    weights = torch.full((tokens, topk), 0.25)
    scales = _expert_scales(num_experts)

    rows_sent: list[int] = []
    original = dist.all_to_all_single

    def recording(output, payload, **kwargs):
        if payload.dim() == 2 and payload.shape[-1] == hidden:
            rows_sent.append(int(payload.shape[0]))
        return original(output, payload, **kwargs)

    with patch.object(ep_dispatch.dist, "all_to_all_single", recording):
        result = expert_parallel_moe(
            states,
            weights,
            indices,
            topology=topology,
            run_expert=lambda index, rows: rows * scales[index],
        )

    # Two rows leave per token instead of four, and rank 0 receives one row per
    # token per peer for its three experts instead of three.
    owning_ranks = {index // topology.experts_per_rank for index in (0, 1, 2, 8)}
    received = tokens * world_size if rank in owning_ranks else 0
    assert rows_sent == [tokens * 2, received]
    torch.testing.assert_close(
        result, _dense_reference(states, weights, indices, scales)
    )
    dist.destroy_process_group()


def _runtime_placement_worker(
    rank: int,
    world_size: int,
    port: int,
    tensor: int,
    cfg: int,
    expert: int,
) -> None:
    # The placement is pure rank arithmetic, so keep the workers off the GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _init(rank, world_size, port)
    num_experts = 8
    plan = resolve_parallel_plan(
        world_size=world_size,
        tensor_parallel_degree=tensor,
        cfg_parallel_degree=cfg,
        expert_parallel_degree=expert,
    )
    runtime = HunyuanImage3ParallelRuntime.from_torchrun(
        plan=plan, num_experts=num_experts
    )

    plane_width = world_size // tensor
    plane_start = (rank // plane_width) * plane_width
    cp_width = plane_width // cfg
    branch = 0 if cfg == 1 else int(rank - plane_start >= cp_width)
    cp_start = plane_start + branch * cp_width

    assert runtime.runs_both_cfg_branches == (cfg == 1)
    assert runtime.cfg_branch_index == branch
    assert runtime.context_parallel_ranks == tuple(
        range(cp_start, cp_start + cp_width)
    )
    assert runtime.context_parallel_rank == rank - cp_start
    assert runtime.context_parallel_degree == cp_width
    if cfg == 2:
        # The two branches pair up shard by shard so the guided combination
        # exchanges predictions for the same sequence positions.
        assert runtime.cfg is not None
        assert runtime.cfg.cfg_ranks == (
            plane_start + runtime.context_parallel_rank,
            plane_start + cp_width + runtime.context_parallel_rank,
        )

    expert_start = plane_start + ((rank - plane_start) // expert) * expert
    assert runtime.experts.ranks == tuple(
        range(expert_start, expert_start + expert)
    )
    assert runtime.experts.experts_per_rank == num_experts // expert
    owned = [
        index for index in range(num_experts) if runtime.owns_checkpoint_expert(index)
    ]
    first = (rank - expert_start) * runtime.experts.experts_per_rank
    assert owned == list(range(first, first + runtime.experts.experts_per_rank))

    partition = runtime.sequence_partition(2 * cp_width + 1)
    assert partition.degree == cp_width
    assert partition.total_length == 2 * cp_width + 1

    runtime.close()
    dist.destroy_process_group()


def _dense_ulysses_worker(rank: int, world_size: int, port: int) -> None:
    _init(rank, world_size, port)
    batch, heads, kv_heads, head_dim = 1, 8, 4, 6
    total_length = 4 * world_size + 3  # deliberately indivisible
    partition = SequencePartition.balanced(
        total_length, degree=world_size, rank=rank
    )
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(batch, heads, total_length, head_dim, generator=generator)
    key = torch.randn(batch, kv_heads, total_length, head_dim, generator=generator)
    value = torch.randn(batch, kv_heads, total_length, head_dim, generator=generator)
    mask = (
        torch.ones(total_length, total_length, dtype=torch.bool)
        .tril()
        .reshape(1, 1, total_length, total_length)
    )
    groups = heads // kv_heads
    reference = F.scaled_dot_product_attention(
        query,
        key.repeat_interleave(groups, dim=1),
        value.repeat_interleave(groups, dim=1),
        attn_mask=mask,
        dropout_p=0.0,
    )

    local_query = partition.shard(query, 2).contiguous()
    local_key = partition.shard(key, 2).contiguous()
    local_value = partition.shard(value, 2).contiguous()
    exchanged_query = heads_to_sequence(
        local_query, partition=partition, process_group=dist.group.WORLD
    )
    exchanged_key = heads_to_sequence(
        local_key, partition=partition, process_group=dist.group.WORLD
    )
    exchanged_value = heads_to_sequence(
        local_value, partition=partition, process_group=dist.group.WORLD
    )
    assert exchanged_query.shape == (
        batch,
        heads // world_size,
        total_length,
        head_dim,
    )
    local_groups = exchanged_query.shape[1] // exchanged_key.shape[1]
    output = F.scaled_dot_product_attention(
        exchanged_query,
        exchanged_key.repeat_interleave(local_groups, dim=1),
        exchanged_value.repeat_interleave(local_groups, dim=1),
        attn_mask=mask,
        dropout_p=0.0,
    )
    restored = sequence_to_heads(
        output, partition=partition, process_group=dist.group.WORLD
    )
    assert restored.shape == (batch, heads, partition.local_length, head_dim)
    torch.testing.assert_close(
        restored, partition.shard(reference, 2), atol=2e-5, rtol=2e-5
    )
    dist.destroy_process_group()


def _uneven_gather_worker(rank: int, world_size: int, port: int) -> None:
    _init(rank, world_size, port)
    total_length = 2 * world_size + 1
    partition = SequencePartition.balanced(
        total_length, degree=world_size, rank=rank
    )
    full = torch.arange(total_length, dtype=torch.float32).reshape(1, total_length, 1)
    gathered = partition.gather(
        partition.shard(full, 1).contiguous(),
        1,
        process_group=dist.group.WORLD,
    )
    torch.testing.assert_close(gathered, full)
    dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", (2, 4))
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_expert_parallel_dispatch_matches_dense_routing(world_size: int) -> None:
    mp.spawn(
        _expert_parallel_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_expert_parallel_dispatch_tolerates_ranks_without_tokens() -> None:
    mp.spawn(
        _uneven_token_worker,
        args=(4, _free_port()),
        nprocs=4,
        join=True,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_experts_sharing_a_rank_send_the_token_once() -> None:
    mp.spawn(
        _shared_destination_worker,
        args=(4, _free_port()),
        nprocs=4,
        join=True,
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("tensor", "cfg", "expert"),
    (
        (2, 2, 4),
        (2, 2, 2),
        (2, 1, 4),
        (2, 1, 1),
        (4, 2, 2),
        (1, 2, 8),
    ),
)
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_the_runtime_places_every_axis_for_a_configured_topology(
    tensor: int, cfg: int, expert: int
) -> None:
    mp.spawn(
        _runtime_placement_worker,
        args=(8, _free_port(), tensor, cfg, expert),
        nprocs=8,
        join=True,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", (2, 4))
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_dense_ulysses_reproduces_full_masked_attention(world_size: int) -> None:
    mp.spawn(
        _dense_ulysses_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", (2, 4))
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_uneven_sequence_gather_restores_rank_order(world_size: int) -> None:
    mp.spawn(
        _uneven_gather_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )
