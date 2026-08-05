"""End-to-end parallel runtime feature layer.

This package owns planning and task-local state for Chitu's parallel runtime.
Low-level collectives remain in ``core.distributed`` and math kernels remain in
``modules``; this layer decides how VAE, DiT, and sampler parallelism compose.
"""

from chitu_diffusion.parallel.planner import ParallelPlanner, build_parallel_plan
from chitu_diffusion.parallel.dit import (
    should_wrap_model_compute_for_context_parallel,
    wrap_attention_backend_for_dit_parallel,
)
from chitu_diffusion.parallel.state import (
    ContextParallelLatentState,
    DitParallelPlan,
    ParallelPlan,
    ParallelTaskState,
    SamplerParallelPlan,
    VaeParallelPlan,
)

__all__ = [
    "ContextParallelLatentState",
    "DitParallelPlan",
    "ParallelPlan",
    "ParallelPlanner",
    "ParallelTaskState",
    "SamplerParallelPlan",
    "VaeParallelPlan",
    "build_parallel_plan",
    "should_wrap_model_compute_for_context_parallel",
    "wrap_attention_backend_for_dit_parallel",
]
