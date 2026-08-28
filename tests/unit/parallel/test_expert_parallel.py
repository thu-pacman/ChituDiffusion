import pytest
import torch

from chitu_diffusion.parallel.ep import (
    ExpertParallelTopology,
    build_expert_parallel_topology,
    expert_parallel_moe,
)


def _single_rank_topology(num_experts: int = 8) -> ExpertParallelTopology:
    return build_expert_parallel_topology(
        num_experts=num_experts,
        ranks=(0,),
        rank=0,
        process_group=None,
    )


def test_contiguous_ownership_splits_experts_evenly():
    topology = ExpertParallelTopology(
        ranks=(0, 1, 2, 3),
        rank=2,
        rank_in_group=2,
        degree=4,
        num_experts=64,
        process_group=object(),
    )
    assert topology.experts_per_rank == 16
    assert topology.local_expert_start == 32
    assert topology.local_expert_end == 48
    assert topology.owns(32) and topology.owns(47)
    assert not topology.owns(31) and not topology.owns(48)
    assert topology.owner_of(0) == 0
    assert topology.owner_of(63) == 3


def test_owner_of_tensor_matches_scalar_mapping():
    topology = ExpertParallelTopology(
        ranks=(0, 1),
        rank=0,
        rank_in_group=0,
        degree=2,
        num_experts=6,
        process_group=object(),
    )
    indices = torch.arange(6)
    expected = torch.tensor([topology.owner_of(int(i)) for i in indices])
    assert torch.equal(topology.owner_of_tensor(indices), expected)


def test_indivisible_expert_count_is_rejected():
    with pytest.raises(ValueError, match="not divisible"):
        ExpertParallelTopology(
            ranks=(0, 1, 2),
            rank=0,
            rank_in_group=0,
            degree=3,
            num_experts=64,
            process_group=object(),
        )


def test_multi_rank_topology_requires_a_process_group():
    with pytest.raises(ValueError, match="requires a process group"):
        build_expert_parallel_topology(
            num_experts=8,
            ranks=(0, 1),
            rank=0,
            process_group=None,
        )


def test_topology_must_contain_the_local_rank():
    with pytest.raises(ValueError, match="must contain the current rank"):
        build_expert_parallel_topology(
            num_experts=8,
            ranks=(1, 2),
            rank=0,
            process_group=object(),
        )


def test_single_rank_dispatch_matches_a_dense_reference():
    torch.manual_seed(0)
    tokens, hidden, num_experts, topk = 7, 5, 8, 3
    states = torch.randn(tokens, hidden)
    logits = torch.randn(tokens, num_experts)
    weights, indices = torch.topk(torch.softmax(logits, dim=-1), topk)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    scales = torch.arange(1, num_experts + 1, dtype=torch.float32)

    def run_expert(expert_index: int, rows: torch.Tensor) -> torch.Tensor:
        return rows * scales[expert_index]

    result = expert_parallel_moe(
        states,
        weights,
        indices,
        topology=_single_rank_topology(num_experts),
        run_expert=run_expert,
    )

    expected = torch.zeros_like(states)
    for token in range(tokens):
        for slot in range(topk):
            expert = int(indices[token, slot])
            expected[token] += states[token] * scales[expert] * weights[token, slot]
    torch.testing.assert_close(result, expected)


def test_dispatch_rejects_expert_outside_local_ownership():
    topology = ExpertParallelTopology(
        ranks=(0,),
        rank=0,
        rank_in_group=0,
        degree=1,
        num_experts=4,
        process_group=None,
    )
    states = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="does not own"):
        expert_parallel_moe(
            states,
            torch.ones(1, 1),
            torch.tensor([[9]]),
            topology=topology,
            run_expert=lambda index, rows: rows,
        )


def test_dispatch_requires_flat_states():
    with pytest.raises(ValueError, match="flat"):
        expert_parallel_moe(
            torch.zeros(1, 2, 3),
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.long),
            topology=_single_rank_topology(),
            run_expert=lambda index, rows: rows,
        )


def test_dispatch_requires_router_shapes_to_agree():
    with pytest.raises(ValueError, match="same shape"):
        expert_parallel_moe(
            torch.zeros(2, 3),
            torch.ones(2, 2),
            torch.zeros(2, 1, dtype=torch.long),
            topology=_single_rank_topology(),
            run_expert=lambda index, rows: rows,
        )


def test_expert_receiving_no_tokens_is_never_invoked():
    topology = _single_rank_topology(num_experts=4)
    invoked: list[int] = []

    def run_expert(expert_index: int, rows: torch.Tensor) -> torch.Tensor:
        invoked.append(expert_index)
        return rows

    expert_parallel_moe(
        torch.randn(3, 2),
        torch.ones(3, 1),
        torch.zeros(3, 1, dtype=torch.long),
        topology=topology,
        run_expert=run_expert,
    )
    assert invoked == [0]
