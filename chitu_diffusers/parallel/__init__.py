"""Model-independent parallel topology and communication primitives."""

from .groups import (
    ActiveLaneTopology,
    CfgParallelTopology,
    EpeParallelContext,
    cfg_parallel_rank_groups,
)
from .image_attention import (
    CONTEXT_PARALLEL_MODES,
    ImageContextParallelAttention,
    ImageSelfAttention,
    resolve_context_parallel_config,
)
from .topology import UspTopology
from .usp import DynamicUspAttention
from .vae import parallel_tiled_vae_decode

__all__ = [
    "ActiveLaneTopology",
    "CONTEXT_PARALLEL_MODES",
    "CfgParallelTopology",
    "DynamicUspAttention",
    "EpeParallelContext",
    "ImageContextParallelAttention",
    "ImageSelfAttention",
    "UspTopology",
    "cfg_parallel_rank_groups",
    "parallel_tiled_vae_decode",
    "resolve_context_parallel_config",
]
