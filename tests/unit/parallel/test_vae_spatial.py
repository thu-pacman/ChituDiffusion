from __future__ import annotations

from dataclasses import dataclass

import torch

from chitu_diffusion.parallel.vae import (
    build_spatial_tile_plan,
    create_vae_parallel_group,
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
