from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from chitu_diffusion.parallel.tp.linear import MergedColumnParallelLinear
from chitu_diffusion.parallel.tp.loader import load_tensor_parallel_checkpoint

from .config import MiniMaxH3DiTConfig
from .transformer import MiniMaxH3DiTModel


def reorder_grouped_qkv_to_qkv(
    loaded_weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    trailing = loaded_weight.shape[1:]
    grouped = loaded_weight.reshape(num_heads, 3, head_dim, *trailing)
    return grouped.permute(1, 0, 2, *range(3, grouped.ndim)).reshape(
        3 * num_heads * head_dim, *trailing
    )


def _h3_tensor_transform(
    name: str,
    tensor_slice: Any,
    module: nn.Module,
    parameter_name: str,
) -> torch.Tensor | None:
    if not (
        name.endswith(".attn.qkv_proj.weight")
        and parameter_name == "weight"
        and isinstance(module, MergedColumnParallelLinear)
    ):
        return None
    shape = tuple(tensor_slice.get_shape())
    if len(shape) != 2:
        raise ValueError(f"{name}: expected a matrix, got {shape}")
    total_q_width = module.output_sizes[0]
    num_heads = int(module.num_attention_heads)
    head_dim = int(module.attention_head_dim)
    if total_q_width % num_heads:
        raise ValueError(f"{name}: unsupported H3 Q width {total_q_width}")
    local_heads = num_heads // module.tp_size
    group_width = 3 * head_dim
    start = module.tp_rank * local_heads * group_width
    stop = start + local_heads * group_width
    local_grouped = tensor_slice[start:stop, :]
    return reorder_grouped_qkv_to_qkv(
        local_grouped,
        num_heads=local_heads,
        head_dim=head_dim,
    )


def load_minimax_h3_transformer(
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str = "cuda",
    attention_backend: str = "auto",
) -> MiniMaxH3DiTModel:
    checkpoint = Path(checkpoint_dir)
    config = MiniMaxH3DiTConfig.from_pretrained(checkpoint)
    model = MiniMaxH3DiTModel(
        config,
        attention_backend=attention_backend,
        device="meta",
    )
    load_tensor_parallel_checkpoint(
        model,
        checkpoint,
        device=device,
        tensor_transform=_h3_tensor_transform,
    )
    model.eval()
    return model

