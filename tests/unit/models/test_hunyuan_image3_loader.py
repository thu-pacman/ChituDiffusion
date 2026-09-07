from __future__ import annotations

import json

import pytest

from chitu_diffusion.models.hunyuan_image3.loader import checkpoint_expert_index
from chitu_diffusion.models.hunyuan_image3.parallel import (
    HunyuanImage3ParallelPlan,
    HunyuanImage3ParallelRuntime,
    resolve_parallel_plan,
)
from chitu_diffusion.parallel.ep import (
    build_expert_parallel_topology,
    expert_parallel_ranks,
)


def test_the_default_plan_is_eight_ranks_of_tp2_by_cfg2_by_cp2_by_ep4() -> None:
    plan = HunyuanImage3ParallelPlan()

    assert (
        plan.tensor_parallel_degree,
        plan.cfg_parallel_degree,
        plan.context_parallel_degree,
    ) == (2, 2, 2)
    assert plan.expert_parallel_degree == 4
    assert plan.plane_width == 4
    assert plan.world_size == 8
    assert plan.describe() == "TP2xCFG2xCP2xEP4"


@pytest.mark.parametrize(
    ("fields", "world_size", "plane_width", "lane_widths"),
    (
        ({}, 8, 4, (1, 2, 4)),
        (
            {"cfg_parallel_degree": 1, "context_parallel_degree": 4},
            8,
            4,
            (1, 4),
        ),
        (
            {
                "cfg_parallel_degree": 1,
                "context_parallel_degree": 4,
                "expert_parallel_degree": 2,
            },
            8,
            4,
            (1, 2, 4),
        ),
        (
            {
                "tensor_parallel_degree": 4,
                "context_parallel_degree": 1,
                "expert_parallel_degree": 2,
            },
            8,
            2,
            (1, 2),
        ),
        (
            {
                "tensor_parallel_degree": 1,
                "context_parallel_degree": 4,
                "expert_parallel_degree": 8,
            },
            8,
            8,
            (1, 4, 8),
        ),
        (
            {
                "tensor_parallel_degree": 2,
                "cfg_parallel_degree": 1,
                "context_parallel_degree": 1,
                "expert_parallel_degree": 1,
            },
            2,
            1,
            (1,),
        ),
    ),
)
def test_the_plan_accepts_any_topology_whose_degrees_multiply_to_the_world(
    fields: dict[str, int],
    world_size: int,
    plane_width: int,
    lane_widths: tuple[int, ...],
) -> None:
    plan = HunyuanImage3ParallelPlan(**fields)

    assert plan.world_size == world_size
    assert plan.plane_width == plane_width
    assert plan.lane_widths == lane_widths


@pytest.mark.parametrize(
    ("fields", "message"),
    (
        ({"cfg_parallel_degree": 4}, "cfg_parallel_degree"),
        ({"tensor_parallel_degree": 0}, "tensor_parallel_degree"),
        ({"context_parallel_degree": 0}, "context_parallel_degree"),
        ({"expert_parallel_degree": 3}, "expert_parallel_degree"),
        ({"expert_parallel_degree": 8}, "expert_parallel_degree"),
        ({"attention_mode": "ring"}, "attention_mode"),
    ),
)
def test_the_plan_rejects_geometry_it_cannot_place(
    fields: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        HunyuanImage3ParallelPlan(**fields)


def test_the_plan_rejects_heads_it_cannot_split_across_the_stage() -> None:
    plan = HunyuanImage3ParallelPlan()
    plan.validate_model_shape(
        num_attention_heads=32,
        num_key_value_heads=8,
        num_experts=64,
        mlp_widths=(4096, 8192),
    )
    with pytest.raises(ValueError, match="num_key_value_heads"):
        plan.validate_model_shape(
            num_attention_heads=32, num_key_value_heads=2, num_experts=64
        )
    with pytest.raises(ValueError, match="num_experts"):
        plan.validate_model_shape(
            num_attention_heads=32, num_key_value_heads=8, num_experts=5
        )
    with pytest.raises(ValueError, match="feed-forward width"):
        plan.validate_model_shape(
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=64,
            mlp_widths=(4096, 999),
        )


def test_a_wider_context_lane_needs_more_key_value_heads() -> None:
    plan = HunyuanImage3ParallelPlan(
        cfg_parallel_degree=1, context_parallel_degree=4
    )
    plan.validate_model_shape(
        num_attention_heads=32, num_key_value_heads=8, num_experts=64
    )
    with pytest.raises(ValueError, match="num_key_value_heads"):
        plan.validate_model_shape(
            num_attention_heads=32, num_key_value_heads=4, num_experts=64
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        ({"tensor_parallel_degree": 2, "cfg_parallel_degree": 2}, (2, 2, 2, 4)),
        ({"tensor_parallel_degree": 2, "cfg_parallel_degree": 1}, (2, 1, 4, 4)),
        ({"tensor_parallel_degree": 4, "cfg_parallel_degree": 2}, (4, 2, 1, 2)),
        ({"tensor_parallel_degree": 1, "cfg_parallel_degree": 2}, (1, 2, 4, 8)),
        (
            {
                "tensor_parallel_degree": 2,
                "cfg_parallel_degree": 2,
                "expert_parallel_degree": 2,
            },
            (2, 2, 2, 2),
        ),
    ),
)
def test_resolving_a_plan_derives_the_degrees_the_world_determines(
    kwargs: dict[str, int], expected: tuple[int, int, int, int]
) -> None:
    plan = resolve_parallel_plan(world_size=8, **kwargs)

    assert (
        plan.tensor_parallel_degree,
        plan.cfg_parallel_degree,
        plan.context_parallel_degree,
        plan.expert_parallel_degree,
    ) == expected


def test_resolving_a_plan_rejects_a_world_the_degrees_cannot_fill() -> None:
    with pytest.raises(ValueError, match="must divide the stage world size"):
        resolve_parallel_plan(
            world_size=8, tensor_parallel_degree=3, cfg_parallel_degree=2
        )
    with pytest.raises(ValueError, match="needs 4 ranks, got 8"):
        resolve_parallel_plan(
            world_size=8,
            tensor_parallel_degree=2,
            cfg_parallel_degree=2,
            context_parallel_degree=1,
        )


def test_expert_index_is_read_from_the_checkpoint_path() -> None:
    assert (
        checkpoint_expert_index("model.layers.3.mlp.experts.17.down_proj.weight") == 17
    )
    for shared in (
        "model.layers.3.mlp.shared_mlp.down_proj.weight",
        "model.layers.3.self_attn.qkv_proj.weight",
    ):
        assert checkpoint_expert_index(shared) is None


def test_owned_experts_are_a_contiguous_quarter_of_the_routing_table() -> None:
    group = object()
    topologies = [
        build_expert_parallel_topology(
            num_experts=64,
            ranks=(0, 1, 2, 3),
            rank=rank,
            process_group=group,
        )
        for rank in range(4)
    ]

    assert [topology.experts_per_rank for topology in topologies] == [16] * 4
    assert [topology.local_expert_start for topology in topologies] == [0, 16, 32, 48]
    assert topologies[2].owns(32) and topologies[2].owns(47)
    assert not topologies[2].owns(48)
    assert sum(topology.owns(63) for topology in topologies) == 1


def test_a_narrow_expert_degree_replicates_the_routing_table_in_the_plane() -> None:
    plane = (4, 5, 6, 7)

    slices = [
        expert_parallel_ranks(plane, rank=rank, degree=2) for rank in plane
    ]

    assert slices == [(4, 5), (4, 5), (6, 7), (6, 7)]
    assert expert_parallel_ranks(plane, rank=6, degree=4) == plane
    assert expert_parallel_ranks(plane, rank=6, degree=1) == (6,)


def test_expert_count_comes_from_the_checkpoint_config(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"num_experts": [8, 8, 8]}))
    assert HunyuanImage3ParallelRuntime.checkpoint_num_experts(tmp_path) == 8

    (tmp_path / "config.json").write_text(json.dumps({"num_experts": 64}))
    assert HunyuanImage3ParallelRuntime.checkpoint_num_experts(tmp_path) == 64

    (tmp_path / "config.json").write_text(json.dumps({"num_experts": [8, 4]}))
    with pytest.raises(ValueError, match="every layer"):
        HunyuanImage3ParallelRuntime.checkpoint_num_experts(tmp_path)
