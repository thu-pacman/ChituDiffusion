from __future__ import annotations

from typing import Any

from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttention_with_CP
from chitu_diffusion.parallel.state import ParallelPlan


def wrap_attention_backend_for_dit_parallel(attn: Any, args: Any, plan: ParallelPlan) -> Any:
    """Wrap a base attention backend according to the resolved DiT parallel plan."""

    if plan.dit.cp_size <= 1:
        return attn
    return DiffusionAttention_with_CP(
        attn,
        plan.dit.up,
        ring_cudagraph=getattr(args.infer.diffusion, "ring_cudagraph", False),
        cp_backend=plan.dit.cp_backend,
    )


def should_wrap_model_compute_for_context_parallel(args: Any, model_adapter: Any, plan: ParallelPlan) -> bool:
    """Whether the runtime should add the generic CP input/output model wrapper."""

    if plan.dit.cp_size <= 1:
        return False
    return not bool(model_adapter.handles_context_parallel(args))
