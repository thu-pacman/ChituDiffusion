"""Model-independent parallel topology and communication primitives."""

from .agkv_transport import (
    AGKV_TRANSPORTS,
    AgkvTransport,
    AsyncAgkvTransport,
    create_agkv_transport,
    resolve_agkv_transport,
)
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
from .attention_backend import (
    Fa4VarlenBackend,
    FlexVarlenBackend,
    SdpaVarlenBackend,
    create_varlen_attention_backend,
)
from .linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from .packed_usp import (
    all_to_all_packed_output,
    all_to_all_packed_qkv,
    validate_tp_ulysses_heads,
)
from .tensor_parallel import (
    TensorParallelTopology,
    adopt_external_tensor_parallel_group,
    get_tp_group,
    get_tp_rank,
    get_tp_topology,
    get_tp_world_size,
    use_tensor_parallel_topology,
)
from .tp_loader import load_tensor_parallel_checkpoint
from .usp import DynamicUspAttention
from .nccl import (
    TorchAgkvTransport,
    TorchUlyssesTransport,
)
from .topology import UlyssesTopology, UspTopology
from .ulysses_transport import (
    ULYSSES_TRANSPORTS,
    AsyncTaggedUlyssesTransport,
    UlyssesTransport,
    create_ulysses_transport,
    resolve_ulysses_transport,
)
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
    "ColumnParallelLinear",
    "DynamicUspAttention",
    "EpeParallelContext",
    "ImageContextParallelAttention",
    "ImageSelfAttention",
    "Fa4VarlenBackend",
    "FlexVarlenBackend",
    "MergedColumnParallelLinear",
    "ReplicatedLinear",
    "RowParallelLinear",
    "SdpaVarlenBackend",
    "TensorParallelTopology",
    "UspTopology",
    "cfg_parallel_rank_groups",
    "all_to_all_packed_output",
    "all_to_all_packed_qkv",
    "adopt_external_tensor_parallel_group",
    "create_varlen_attention_backend",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_topology",
    "get_tp_world_size",
    "load_tensor_parallel_checkpoint",
    "TorchAgkvTransport",
    "TorchUlyssesTransport",
    "UlyssesTransport",
    "UlyssesTopology",
    "cfg_parallel_rank_groups",
    "create_agkv_transport",
    "create_ulysses_transport",
    "parallel_tiled_vae_decode",
    "resolve_agkv_transport",
    "resolve_context_parallel_config",
    "validate_tp_ulysses_heads",
    "use_tensor_parallel_topology",
    "resolve_ulysses_transport",
]
