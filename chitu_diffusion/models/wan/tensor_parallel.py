"""Tensor-parallel plan for the Wan 2.1 video transformer.

One table covers the Wan 2.1 checkpoints this package loads, T2V and I2V at
either size: they differ only in widths and depth, which the plan reads off the
model instead of hard-coding.

Wan is the reason the shared plan grew a ``norm`` role. Its qk-norm is
``rms_norm_across_heads``: ``RMSNorm(heads * head_dim)`` applied to the whole
projection before it is split into heads. Sharding the projections therefore
leaves no rank holding the vector the scale is computed from, so those norms
have to reduce across ranks rather than be replicated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...parallel.tp import (
    TensorParallelPlan,
    TensorParallelRewrite,
    load_tensor_parallel_diffusers_model,
)

WAN_TENSOR_PARALLEL_PLAN = TensorParallelPlan(
    column=(
        "blocks.*.attn1.to_q",
        "blocks.*.attn1.to_k",
        "blocks.*.attn1.to_v",
        "blocks.*.attn2.to_q",
        "blocks.*.attn2.to_k",
        "blocks.*.attn2.to_v",
        # I2V only: the image cross-attention K/V share attn2's head split.
        "blocks.*.attn2.add_k_proj",
        "blocks.*.attn2.add_v_proj",
        "blocks.*.ffn.net.0.proj",
    ),
    row=(
        "blocks.*.attn1.to_out.0",
        "blocks.*.attn2.to_out.0",
        "blocks.*.ffn.net.2",
    ),
    replicated=(
        # Timestep and text conditioning are tiny next to the blocks, and the
        # modulation they produce is consumed at full width by the layer norms.
        "condition_embedder.**",
        # Unpatchify needs the whole channel vector on the rank that writes it.
        "proj_out",
    ),
    norm=(
        "blocks.*.attn1.norm_q",
        "blocks.*.attn1.norm_k",
        "blocks.*.attn2.norm_q",
        "blocks.*.attn2.norm_k",
        "blocks.*.attn2.norm_added_k",
    ),
    head_counts=(
        ("blocks.*.attn1", "heads"),
        ("blocks.*.attn2", "heads"),
    ),
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
        WAN_TENSOR_PARALLEL_PLAN,
        degree=degree,
        **kwargs,
    )


__all__ = [
    "WAN_TENSOR_PARALLEL_PLAN",
    "load_tensor_parallel_transformer",
]
