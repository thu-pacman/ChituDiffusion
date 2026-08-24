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
from .attention_backend import (
    Fa4VarlenBackend,
    FlexVarlenBackend,
    SdpaVarlenBackend,
    create_varlen_attention_backend,
)
from .topology import UspTopology
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
from .vae import parallel_tiled_vae_decode

__all__ = [
    "ActiveLaneTopology",
    "CONTEXT_PARALLEL_MODES",
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
    "parallel_tiled_vae_decode",
    "resolve_context_parallel_config",
    "validate_tp_ulysses_heads",
    "use_tensor_parallel_topology",
]
