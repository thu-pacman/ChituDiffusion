"""VAE-parallel topology and spatial decode primitives."""

from .spatial import (
    SpatialTile,
    SpatialTilePlan,
    assemble_spatial_tiles,
    build_spatial_tile_plan,
    parallel_spatial_vae_decode,
    parallel_tiled_vae_decode,
)
from .topology import (
    VaeParallelGroup,
    VaeParallelTopology,
    create_vae_parallel_group,
)

__all__ = [
    "SpatialTile",
    "SpatialTilePlan",
    "VaeParallelGroup",
    "VaeParallelTopology",
    "assemble_spatial_tiles",
    "build_spatial_tile_plan",
    "create_vae_parallel_group",
    "parallel_spatial_vae_decode",
    "parallel_tiled_vae_decode",
]
