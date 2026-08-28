from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from chitu_diffusion.parallel.vae import (
    VaeParallelPlacement,
    build_spatial_tile_plan,
    create_vae_parallel_group,
    create_vae_parallel_placement,
    parallel_spatial_vae_decode,
)


@dataclass
class _SingleRankTopology:
    rank_in_lane: int = 0
    width: int = 1
    process_group: object | None = None

    @property
    def is_leader(self) -> bool:
        return True


def test_h3_768p_tile_plan_matches_release_split_rule() -> None:
    plan = build_spatial_tile_plan(48, 48)

    assert (plan.rows, plan.columns) == (4, 4)
    assert plan.y_overlaps == (96, 80, 80)
    assert plan.x_overlaps == (96, 80, 80)
    assert [tile.index for tile in plan.tiles] == list(range(16))
    assert plan.tiles[-1].latent_y + plan.tiles[-1].latent_height == 48
    assert plan.tiles[-1].latent_x + plan.tiles[-1].latent_width == 48


def test_single_rank_spatial_tiles_match_full_nearest_decode() -> None:
    latents = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    decoded = parallel_spatial_vae_decode(
        latents,
        lambda tile: tile.repeat_interleave(2, -2).repeat_interleave(2, -1),
        topology=_SingleRankTopology(),
        scale=2,
        tile_size=4,
        overlap_min=2,
    )

    assert decoded is not None
    expected = latents.repeat_interleave(2, -2).repeat_interleave(2, -1)
    torch.testing.assert_close(decoded, expected)


def test_scale16_1024_plan_covers_the_canvas_without_gaps() -> None:
    # 1024px at a 16x spatial scale is a 64x64 latent. Exercise 384px tiles with
    # at least 48px of overlap, independent of any particular VAE implementation.
    plan = build_spatial_tile_plan(64, 64, tile_size=384, overlap_min=48)

    assert (plan.rows, plan.columns) == (3, 3)
    assert plan.y_overlaps == plan.x_overlaps == (64, 64)
    assert len(plan.tiles) == 9
    for tile in plan.tiles:
        assert tile.latent_height == tile.latent_width == 24
        assert tile.latent_y + tile.latent_height <= 64
        assert tile.latent_x + tile.latent_width <= 64
    covered = torch.zeros(64, 64, dtype=torch.bool)
    for tile in plan.tiles:
        covered[
            tile.latent_y : tile.latent_y + tile.latent_height,
            tile.latent_x : tile.latent_x + tile.latent_width,
        ] = True
    assert bool(covered.all())


def test_scale16_tiles_reassemble_into_the_full_decode() -> None:
    latents = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64)

    decoded = parallel_spatial_vae_decode(
        latents,
        lambda tile: tile.repeat_interleave(16, -2).repeat_interleave(16, -1),
        topology=_SingleRankTopology(),
        scale=16,
        tile_size=384,
        overlap_min=48,
    )

    assert decoded is not None
    torch.testing.assert_close(
        decoded, latents.repeat_interleave(16, -2).repeat_interleave(16, -1)
    )


def test_small_canvas_uses_one_release_tile() -> None:
    plan = build_spatial_tile_plan(8, 12)
    assert (plan.rows, plan.columns) == (1, 1)
    assert plan.y_overlaps == ()
    assert plan.x_overlaps == ()


def test_single_rank_vaep_can_be_a_subset_of_stage_world() -> None:
    leader = create_vae_parallel_group(1, world_size=8, rank=0)
    nonmember = create_vae_parallel_group(1, world_size=8, rank=7)

    assert leader.is_member and leader.is_leader
    assert not nonmember.is_member and not nonmember.is_leader
    assert leader.ranks == nonmember.ranks == (0,)


def test_an_unset_degree_keeps_decode_on_the_lane_that_denoised() -> None:
    placement = create_vae_parallel_placement(None, halo=4)
    lane = _SingleRankTopology(width=2)

    assert placement.degree is None
    assert not placement.is_stage_scoped
    assert placement.resolve(lane) is lane
    assert placement.halo == 4


def test_degree_one_decodes_the_whole_canvas_on_the_lane_leader() -> None:
    placement = create_vae_parallel_placement(1, world_size=8, rank=3)
    lane = _SingleRankTopology(width=2)

    assert placement.degree == 1
    assert not placement.sharded
    assert placement.resolve(lane) is lane


def test_a_fixed_group_replaces_the_lane_and_excludes_outsiders() -> None:
    lane = _SingleRankTopology(width=2)
    member = VaeParallelPlacement(
        group=create_vae_parallel_group(1, world_size=8, rank=0)
    )
    outsider = VaeParallelPlacement(
        group=create_vae_parallel_group(1, world_size=8, rank=7)
    )

    assert member.is_stage_scoped and member.resolve(lane) is member.group
    assert member.degree == 1
    assert outsider.resolve(lane) is None


def test_a_negative_halo_is_refused_before_any_group_is_created() -> None:
    with pytest.raises(ValueError, match="halo must be non-negative"):
        create_vae_parallel_placement(None, halo=-1)
