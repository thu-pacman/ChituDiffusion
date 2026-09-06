"""Build a diffusers model straight into its tensor-parallel shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights

from .loader import load_tensor_parallel_checkpoint
from .plan import TensorParallelPlan, TensorParallelRewrite, apply_tensor_parallel_plan


def load_tensor_parallel_diffusers_model(
    cls: type,
    pretrained_model_name_or_path: str | Path,
    plan: TensorParallelPlan,
    *,
    degree: int,
    subfolder: str | None = "transformer",
    torch_dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
    **ignored: Any,
) -> tuple[Any, TensorParallelRewrite]:
    """Build the model on meta, shard it, then read only the local weights.

    Materializing the full checkpoint on every rank and slicing afterwards
    would peak at the whole model per rank, so the shards are read straight out
    of the safetensors files into the parameters that keep them.
    """

    for accepted in ("variant", "revision", "local_files_only"):
        ignored.pop(accepted, None)
    if ignored:
        raise TypeError(
            f"tensor-parallel {cls.__name__} loading does not accept "
            + ", ".join(sorted(ignored))
        )

    root = Path(pretrained_model_name_or_path)
    if subfolder:
        root = root / subfolder
    if not root.is_dir():
        raise ValueError(
            f"tensor-parallel {cls.__name__} needs a local checkpoint directory, "
            f"got {root}"
        )

    config = cls.load_config(root)
    # Parameters go to meta so nothing allocates before it is sharded, but
    # buffers stay real: a model whose rotary tables are non-persistent buffers
    # computes them here and no checkpoint can supply them later.
    with init_empty_weights(include_buffers=False):
        model = cls.from_config(config)
    if torch_dtype is not None:
        model = model.to(torch_dtype)

    rewrite = apply_tensor_parallel_plan(model, plan, degree=degree)
    load_tensor_parallel_checkpoint(model, root, device=device)
    model.eval()
    return model, rewrite


__all__ = ["load_tensor_parallel_diffusers_model"]
