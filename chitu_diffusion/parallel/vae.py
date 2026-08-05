"""Dynamic lane-parallel VAE decode primitives."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from .groups import ActiveLaneTopology


def parallel_tiled_vae_decode(
    latents: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    topology: ActiveLaneTopology,
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
