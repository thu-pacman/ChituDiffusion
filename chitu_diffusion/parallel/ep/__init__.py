"""Expert-parallel topology and routed-expert dispatch."""

from .dispatch import expert_parallel_moe
from .topology import (
    ExpertParallelTopology,
    build_expert_parallel_topology,
    expert_parallel_ranks,
)

__all__ = [
    "ExpertParallelTopology",
    "build_expert_parallel_topology",
    "expert_parallel_moe",
    "expert_parallel_ranks",
]
