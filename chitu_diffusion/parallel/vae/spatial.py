"""Dynamic lane-parallel VAE decode primitives."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .topology import VaeParallelTopology


@dataclass(frozen=True)
class SpatialTile:
    index: int
    row: int
    column: int
    latent_y: int
    latent_x: int
    latent_height: int
    latent_width: int


@dataclass(frozen=True)
class SpatialTilePlan:
    rows: int
    columns: int
    tiles: tuple[SpatialTile, ...]
    y_overlaps: tuple[int, ...]
    x_overlaps: tuple[int, ...]


def _split_tile_axis(
    pixel_extent: int,
    *,
    tile_size: int,
    overlap_min: int,
    scale: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if pixel_extent <= 0:
        raise ValueError("pixel extent must be positive")
    if tile_size <= 0 or scale <= 0:
        raise ValueError("tile size and scale must be positive")
    if tile_size % scale or overlap_min % scale:
        raise ValueError("tile size and overlap must be divisible by scale")
    if overlap_min < 0 or overlap_min >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")
    if tile_size >= pixel_extent:
        return (0,), (pixel_extent,), ()

    tile_count = math.ceil(pixel_extent / tile_size)
    while True:
        overlaps = [overlap_min] * (tile_count - 1)
        remaining = tile_size * tile_count - sum(overlaps) - pixel_extent
        if remaining >= 0:
            break
        tile_count += 1
    if remaining % scale:
        raise ValueError("pixel extent must preserve the VAE scale grid")
    for index in range(remaining // scale):
        overlaps[index % (tile_count - 1)] += scale

    starts = [0]
    for index in range(tile_count - 1):
        starts.append(starts[-1] + tile_size - overlaps[index])
    return tuple(starts), (tile_size,) * tile_count, tuple(overlaps)


def build_spatial_tile_plan(
    latent_height: int,
    latent_width: int,
    *,
    scale: int = 16,
    tile_size: int = 256,
    overlap_min: int = 64,
) -> SpatialTilePlan:
    """Build the release-compatible row-major H/W tile plan."""

    pixel_height = latent_height * scale
    pixel_width = latent_width * scale
    y_starts, y_lengths, y_overlaps = _split_tile_axis(
        pixel_height,
        tile_size=tile_size,
        overlap_min=overlap_min,
        scale=scale,
    )
    x_starts, x_lengths, x_overlaps = _split_tile_axis(
        pixel_width,
        tile_size=tile_size,
        overlap_min=overlap_min,
        scale=scale,
    )
    tiles = []
    columns = len(x_starts)
    for row, (y_start, y_length) in enumerate(zip(y_starts, y_lengths)):
        for column, (x_start, x_length) in enumerate(zip(x_starts, x_lengths)):
            tiles.append(
                SpatialTile(
                    index=row * columns + column,
                    row=row,
                    column=column,
                    latent_y=y_start // scale,
                    latent_x=x_start // scale,
                    latent_height=y_length // scale,
                    latent_width=x_length // scale,
                )
            )
    return SpatialTilePlan(
        rows=len(y_starts),
        columns=columns,
        tiles=tuple(tiles),
        y_overlaps=y_overlaps,
        x_overlaps=x_overlaps,
    )


def _blend_tile(
    previous: torch.Tensor,
    current: torch.Tensor,
    extent: int,
    *,
    dim: int,
) -> torch.Tensor:
    extent = min(previous.shape[dim], current.shape[dim], extent)
    if extent <= 0:
        return current
    positions = torch.arange(extent, device=current.device, dtype=current.dtype)
    shape = [1] * current.ndim
    shape[dim] = extent
    current_weight = (positions / extent).view(shape)
    previous_weight = (1 - positions / extent).view(shape)
    previous_slice = [slice(None)] * previous.ndim
    previous_slice[dim] = slice(-extent, None)
    current_slice = [slice(None)] * current.ndim
    current_slice[dim] = slice(0, extent)
    blended = (
        previous[tuple(previous_slice)] * previous_weight
        + current[tuple(current_slice)] * current_weight
    )
    if extent >= current.shape[dim]:
        return blended
    current_rest = [slice(None)] * current.ndim
    current_rest[dim] = slice(extent, None)
    return torch.cat((blended, current[tuple(current_rest)]), dim=dim)


def assemble_spatial_tiles(
    tiles: list[torch.Tensor],
    plan: SpatialTilePlan,
) -> torch.Tensor:
    """Blend decoded tiles in the release VAE's vertical-then-horizontal order."""

    if len(tiles) != len(plan.tiles):
        raise ValueError(f"expected {len(plan.tiles)} tiles, got {len(tiles)}")
    rows = [
        tiles[row * plan.columns : (row + 1) * plan.columns]
        for row in range(plan.rows)
    ]
    output_height = sum(
        row[0].shape[-2] - (plan.y_overlaps[index] if index < plan.rows - 1 else 0)
        for index, row in enumerate(rows)
    )
    output_width = sum(
        tile.shape[-1]
        - (plan.x_overlaps[index] if index < plan.columns - 1 else 0)
        for index, tile in enumerate(rows[0])
    )
    output = rows[0][0].new_empty((*rows[0][0].shape[:-2], output_height, output_width))
    y_offset = 0
    for row_index, row in enumerate(rows):
        row_height = row[0].shape[-2] - (
            plan.y_overlaps[row_index] if row_index < plan.rows - 1 else 0
        )
        x_offset = 0
        for column_index, tile in enumerate(row):
            if row_index > 0:
                tile = _blend_tile(
                    rows[row_index - 1][column_index],
                    tile,
                    plan.y_overlaps[row_index - 1],
                    dim=-2,
                )
            if column_index > 0:
                tile = _blend_tile(
                    row[column_index - 1],
                    tile,
                    plan.x_overlaps[column_index - 1],
                    dim=-1,
                )
            if row_index < plan.rows - 1 and plan.y_overlaps[row_index] > 0:
                tile = tile[..., : -plan.y_overlaps[row_index], :]
            if column_index < plan.columns - 1 and plan.x_overlaps[column_index] > 0:
                tile = tile[..., :, : -plan.x_overlaps[column_index]]
            tile_height, tile_width = tile.shape[-2:]
            output[
                ...,
                y_offset : y_offset + tile_height,
                x_offset : x_offset + tile_width,
            ].copy_(tile)
            x_offset += tile_width
        y_offset += row_height
    return output


def parallel_spatial_vae_decode(
    latents: torch.Tensor,
    decode_tile: Callable[[torch.Tensor], torch.Tensor],
    *,
    topology: VaeParallelTopology,
    scale: int = 16,
    tile_size: int = 256,
    overlap_min: int = 64,
    enabled: bool = True,
) -> torch.Tensor | None:
    """Decode release-compatible spatial tiles across a lane.

    All ranks own the complete latent. Tile tasks are assigned round-robin and
    padded along the task dimension for one equal-shape all-gather. Only the
    lane leader blends and returns the complete video.
    """

    if latents.ndim < 4:
        raise ValueError("VAE latents must have spatial dimensions")
    if not enabled:
        return decode_tile(latents) if topology.is_leader else None
    plan = build_spatial_tile_plan(
        int(latents.shape[-2]),
        int(latents.shape[-1]),
        scale=scale,
        tile_size=tile_size,
        overlap_min=overlap_min,
    )

    def run(indices: list[int]) -> list[torch.Tensor]:
        outputs = []
        for index in indices:
            tile = plan.tiles[index]
            latent_tile = latents[
                ...,
                tile.latent_y : tile.latent_y + tile.latent_height,
                tile.latent_x : tile.latent_x + tile.latent_width,
            ].contiguous()
            outputs.append(decode_tile(latent_tile))
        return outputs

    tile_count = len(plan.tiles)
    if topology.width <= 1 or tile_count < topology.width:
        if not topology.is_leader:
            return None
        return assemble_spatial_tiles(run(list(range(tile_count))), plan)
    if topology.process_group is None:
        raise RuntimeError("multi-rank VAE decode requires a process group")

    local_indices = list(range(topology.rank_in_lane, tile_count, topology.width))
    local_outputs = run(local_indices)
    max_tasks = (tile_count + topology.width - 1) // topology.width
    stacked = local_outputs[0].new_zeros((max_tasks, *local_outputs[0].shape))
    torch.stack(local_outputs, dim=0, out=stacked[: len(local_outputs)])
    gathered = [torch.empty_like(stacked) for _ in range(topology.width)]
    dist.all_gather(gathered, stacked, group=topology.process_group)
    if not topology.is_leader:
        return None

    ordered: list[torch.Tensor | None] = [None] * tile_count
    for rank, rank_outputs in enumerate(gathered):
        rank_tasks = (tile_count - rank + topology.width - 1) // topology.width
        for local_index in range(rank_tasks):
            ordered[local_index * topology.width + rank] = rank_outputs[local_index]
    if any(tile is None for tile in ordered):
        raise RuntimeError("VAE tile gather produced an incomplete result")
    return assemble_spatial_tiles([tile for tile in ordered if tile is not None], plan)


def parallel_tiled_vae_decode(
    latents: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    topology: VaeParallelTopology,
    latent_split_dim: int,
    pixel_split_dim: int,
    scale: int,
    halo: int = 8,
    enabled: bool = True,
) -> torch.Tensor | None:
    """Decode one spatial tile per active lane rank and gather on the leader.

    The full normalized latent is replicated across the lane. Each rank decodes
    its core plus an interior halo, trims the halo pixels, and participates in a
    lane-local gather. Only the lane leader materializes the full decoded image.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    if halo < 0:
        raise ValueError("halo must be non-negative")
    if topology.width <= 1 or not enabled:
        return decode_fn(latents) if topology.is_leader else None
    if topology.process_group is None:
        raise RuntimeError("a multi-rank VAE lane requires a process group")

    latent_extent = int(latents.shape[latent_split_dim])
    if latent_extent < topology.width:
        return decode_fn(latents) if topology.is_leader else None

    core_extent = (latent_extent + topology.width - 1) // topology.width
    core_start = topology.rank_in_lane * core_extent
    core_end = min(core_start + core_extent, latent_extent)
    tile_start = max(0, core_start - halo)
    tile_end = min(latent_extent, core_end + halo)

    tile = latents.narrow(
        latent_split_dim, tile_start, tile_end - tile_start
    ).contiguous()
    decoded_tile = decode_fn(tile)
    pixel_offset = (core_start - tile_start) * scale
    pixel_extent = (core_end - core_start) * scale
    core = decoded_tile.narrow(pixel_split_dim, pixel_offset, pixel_extent)

    padded_extent = core_extent * scale
    if pixel_extent < padded_extent:
        padding_shape = list(core.shape)
        padding_shape[pixel_split_dim] = padded_extent - pixel_extent
        core = torch.cat([core, core.new_zeros(padding_shape)], dim=pixel_split_dim)
    core = core.contiguous()

    gathered = [torch.empty_like(core) for _ in range(topology.width)]
    dist.all_gather(gathered, core, group=topology.process_group)
    if not topology.is_leader:
        return None
    pieces = []
    for lane_rank, value in enumerate(gathered):
        true_extent = (
            min((lane_rank + 1) * core_extent, latent_extent) - lane_rank * core_extent
        ) * scale
        if true_extent > 0:
            pieces.append(value.narrow(pixel_split_dim, 0, true_extent))
    return torch.cat(pieces, dim=pixel_split_dim).contiguous()
