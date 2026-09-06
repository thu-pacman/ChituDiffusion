"""Z-Image's tensor-parallel table and its rank-local checkpoint load.

Z-Image is a plain single-stream DiT: every block projects Q/K/V, attends,
projects out, then runs a SwiGLU feed-forward. Only those seven projections
carry the hidden dimension, so only they shard. The modulation, embedding, and
output-head linears stay replicated, which is what lets the blocks hand each
other complete activations without a gather.

Its qk-norm is per head -- the processor reshapes to heads before normalizing
-- so the norms need no reduction across shards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...parallel.tp import (
    TensorParallelPlan,
    TensorParallelRewrite,
    load_tensor_parallel_diffusers_model,
)

ZIMAGE_TENSOR_PARALLEL_PLAN = TensorParallelPlan(
    column=(
        "**.attention.to_q",
        "**.attention.to_k",
        "**.attention.to_v",
        "**.feed_forward.w1",
        "**.feed_forward.w3",
    ),
    row=(
        "**.attention.to_out.0",
        "**.feed_forward.w2",
    ),
    replicated=(
        "**.adaLN_modulation.*",
        "all_x_embedder.*",
        "all_final_layer.*.linear",
        "cap_embedder.*",
        "t_embedder.mlp.*",
    ),
    head_counts=(("**.attention", "heads"),),
)


def load_tensor_parallel_transformer(
    cls: type,
    pretrained_model_name_or_path: str | Path,
    *,
    degree: int,
    **kwargs: Any,
) -> tuple[Any, TensorParallelRewrite]:
    return load_tensor_parallel_diffusers_model(
        cls,
        pretrained_model_name_or_path,
        ZIMAGE_TENSOR_PARALLEL_PLAN,
        degree=degree,
        **kwargs,
    )


__all__ = [
    "ZIMAGE_TENSOR_PARALLEL_PLAN",
    "load_tensor_parallel_transformer",
]
