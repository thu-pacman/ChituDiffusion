"""Model-independent parallel topology and communication primitives."""

from .agkv_transport import (
    AGKV_TRANSPORTS,
    AgkvTransport,
    AsyncAgkvTransport,
    create_agkv_transport,
    resolve_agkv_transport,
)
from .fast_cp import FastAgkvTransport, FastUlyssesTransport
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
from .nccl import (
    TorchAgkvTransport,
    TorchUlyssesTransport,
)
from .topology import UspTopology, select_ulysses_degree
from .ulysses_transport import (
    ULYSSES_TRANSPORTS,
    AsyncTaggedUlyssesTransport,
    UlyssesTransport,
    create_ulysses_transport,
    resolve_ulysses_transport,
)
from .usp import DynamicUspAttention
from .vae import parallel_tiled_vae_decode

__all__ = [
    "AGKV_TRANSPORTS",
    "CONTEXT_PARALLEL_MODES",
    "ULYSSES_TRANSPORTS",
    "ActiveLaneTopology",
    "AgkvTransport",
    "AsyncAgkvTransport",
    "AsyncTaggedUlyssesTransport",
    "CfgParallelTopology",
    "DynamicUspAttention",
    "EpeParallelContext",
    "FastAgkvTransport",
    "FastUlyssesTransport",
    "ImageContextParallelAttention",
    "ImageSelfAttention",
    "TorchAgkvTransport",
    "TorchUlyssesTransport",
    "UlyssesTransport",
    "UspTopology",
    "cfg_parallel_rank_groups",
    "create_agkv_transport",
    "create_ulysses_transport",
    "parallel_tiled_vae_decode",
    "resolve_agkv_transport",
    "resolve_context_parallel_config",
    "resolve_ulysses_transport",
    "select_ulysses_degree",
]
