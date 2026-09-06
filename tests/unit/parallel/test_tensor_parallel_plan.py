from __future__ import annotations

import pytest
import torch
from torch import nn

from chitu_diffusion.parallel.tp import (
    PlainColumnParallelLinear,
    PlainRowParallelLinear,
    TensorParallelPlan,
    TensorParallelTopology,
    apply_tensor_parallel_plan,
    use_tensor_parallel_topology,
)


class Block(nn.Module):
    def __init__(self, dim: int = 8, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True), nn.Dropout(0.0)])
        self.modulation = nn.Linear(dim, dim)


class Tiny(nn.Module):
    def __init__(self, dim: int = 8, heads: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(2)])


PLAN = TensorParallelPlan(
    column=("blocks.*.to_q",),
    row=("blocks.*.to_out.0",),
    replicated=("blocks.*.modulation",),
    head_counts=(("blocks.*", "heads"),),
)


def test_a_segment_glob_stops_at_the_dot():
    from chitu_diffusion.parallel.tp.plan import matches

    assert matches("blocks.0.to_q", "blocks.*.to_q")
    assert not matches("blocks.0.attention.to_q", "blocks.*.to_q")
    assert matches("blocks.0.attention", "**.attention")
    assert not matches("blocks.0.attention.to_q", "**.attention")
    assert matches("attention", "**.attention")
    assert matches("a.b.c.linear", "**.linear")


def _topology(degree: int) -> TensorParallelTopology:
    return TensorParallelTopology(
        ranks=tuple(range(degree)),
        rank=0,
        rank_in_group=0,
        degree=degree,
        process_group=None,
    )


def test_a_plan_classifies_every_linear_without_touching_the_graph():
    model = Tiny()
    rewrite = apply_tensor_parallel_plan(model, PLAN, degree=1)

    assert rewrite.column == ("blocks.0.to_q", "blocks.1.to_q")
    assert rewrite.row == ("blocks.0.to_out.0", "blocks.1.to_out.0")
    assert rewrite.replicated == ("blocks.0.modulation", "blocks.1.modulation")
    assert isinstance(model.blocks[0].to_q, nn.Linear)
    assert not isinstance(model.blocks[0].to_q, PlainColumnParallelLinear)
    assert model.blocks[0].heads == 4


def test_sharding_replaces_the_named_layers_and_shrinks_the_head_count():
    model = Tiny()
    with use_tensor_parallel_topology(_topology(2)):
        rewrite = apply_tensor_parallel_plan(model, PLAN, degree=2)

    assert rewrite.degree == 2
    assert rewrite.sharded == 4
    column = model.blocks[0].to_q
    row = model.blocks[0].to_out[0]
    assert isinstance(column, PlainColumnParallelLinear)
    assert isinstance(row, PlainRowParallelLinear)
    assert column.weight.shape == (4, 8)
    assert column.bias is None
    assert row.weight.shape == (8, 4)
    assert row.bias is not None and row.bias.shape == (8,)
    assert isinstance(model.blocks[0].modulation, nn.Linear)
    assert model.blocks[0].heads == 2


def test_an_unnamed_linear_is_an_error_rather_than_a_silent_replication():
    model = Tiny()
    model.blocks[0].extra = nn.Linear(8, 8)

    with pytest.raises(ValueError, match="does not name these linear layers"):
        apply_tensor_parallel_plan(model, PLAN, degree=1)


def test_a_layer_claimed_by_two_roles_is_rejected():
    ambiguous = TensorParallelPlan(
        column=("blocks.*.to_q",),
        row=("blocks.*.to_q", "blocks.*.to_out.0"),
        replicated=("blocks.*.modulation",),
    )

    with pytest.raises(ValueError, match="more than one tensor-parallel role"):
        apply_tensor_parallel_plan(Tiny(), ambiguous, degree=1)


def test_an_indivisible_projection_names_itself():
    with pytest.raises(ValueError, match=r"blocks\.0\.to_q\.out_features=8"):
        apply_tensor_parallel_plan(Tiny(dim=8), PLAN, degree=3)


def test_an_indivisible_head_count_names_itself():
    with pytest.raises(ValueError, match=r"blocks\.0\.heads=6"):
        apply_tensor_parallel_plan(Tiny(dim=8, heads=6), PLAN, degree=4)


def test_a_rewrite_refuses_to_build_shards_for_an_inactive_degree():
    with pytest.raises(ValueError, match="does not match the active topology"):
        apply_tensor_parallel_plan(Tiny(), PLAN, degree=2)


def test_a_head_pattern_that_matches_nothing_is_reported():
    stale = TensorParallelPlan(
        column=("blocks.*.to_q",),
        row=("blocks.*.to_out.0",),
        replicated=("blocks.*.modulation",),
        head_counts=(("attention.*", "heads"),),
    )

    with pytest.raises(ValueError, match="no module matches the head-count pattern"):
        apply_tensor_parallel_plan(Tiny(), stale, degree=1)


def test_column_then_row_reproduces_the_dense_result():
    torch.manual_seed(0)
    dense = Tiny().eval().blocks[0]
    inputs = torch.randn(2, 3, 8)
    with torch.no_grad():
        expected = dense.to_out[0](dense.to_q(inputs))

    weight_q = dense.to_q.weight.detach().clone()
    weight_o = dense.to_out[0].weight.detach().clone()
    bias_o = dense.to_out[0].bias.detach().clone()

    partials = []
    for rank in range(2):
        model = Tiny()
        with use_tensor_parallel_topology(
            TensorParallelTopology(
                ranks=(0, 1),
                rank=rank,
                rank_in_group=rank,
                degree=2,
                process_group=None,
            )
        ):
            apply_tensor_parallel_plan(model, PLAN, degree=2)
        shard = model.blocks[0]
        with torch.no_grad():
            shard.to_q.weight.copy_(weight_q[rank * 4 : (rank + 1) * 4])
            shard.to_out[0].weight.copy_(weight_o[:, rank * 4 : (rank + 1) * 4])
            shard.to_out[0].bias.copy_(bias_o)
            # reduce_results would all-reduce; with no group we sum by hand.
            partials.append(
                torch.nn.functional.linear(
                    shard.to_q(inputs), shard.to_out[0].weight, None
                )
            )

    torch.testing.assert_close(partials[0] + partials[1] + bias_o, expected)
