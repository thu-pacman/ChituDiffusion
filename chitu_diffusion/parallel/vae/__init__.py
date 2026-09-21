"""VAE-parallel topology and spatial decode primitives."""

from .exact import parallel_vae_decode
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
    VaeParallelPlacement,
    VaeParallelTopology,
    create_vae_parallel_group,
    create_vae_parallel_placement,
)

__all__ = [
    "SpatialTile",
    "SpatialTilePlan",
    "VaeParallelGroup",
    "VaeParallelPlacement",
    "VaeParallelTopology",
    "assemble_spatial_tiles",
    "build_spatial_tile_plan",
    "create_vae_parallel_group",
    "create_vae_parallel_placement",
    "parallel_spatial_vae_decode",
    "parallel_tiled_vae_decode",
    "parallel_vae_decode",
]
