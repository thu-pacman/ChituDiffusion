from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from .linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)

TensorTransform = Callable[
    [str, Any, nn.Module, str], torch.Tensor | None
]


def _weight_map(checkpoint_dir: Path) -> dict[str, Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        return {
            name: checkpoint_dir / filename
            for name, filename in payload["weight_map"].items()
        }
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected one safetensors file or an index in {checkpoint_dir}"
        )
    with safe_open(files[0], framework="pt", device="cpu") as handle:
        return {name: files[0] for name in handle.keys()}


def iter_checkpoint_tensors(
    checkpoint_dir: str | Path,
) -> Iterator[tuple[str, Path]]:
    yield from _weight_map(Path(checkpoint_dir)).items()


def _read_local_tensor(
    tensor_slice: Any,
    module: nn.Module,
    parameter_name: str,
) -> torch.Tensor:
    shape = tuple(tensor_slice.get_shape())
    if isinstance(module, MergedColumnParallelLinear):
        local_parts = []
        source_offset = 0
        for logical_size, local_size in zip(
            module.output_sizes, module.output_partition_sizes, strict=True
        ):
            start = source_offset + module.tp_rank * local_size
            stop = start + local_size
            if len(shape) == 1:
                local_parts.append(tensor_slice[start:stop])
            else:
                local_parts.append(tensor_slice[start:stop, :])
            source_offset += logical_size
        return torch.cat(local_parts, dim=0)
    if isinstance(module, ColumnParallelLinear):
        local_size = module.output_size_per_partition
        start = module.tp_rank * local_size
        stop = start + local_size
        return tensor_slice[start:stop] if len(shape) == 1 else tensor_slice[start:stop, :]
    if isinstance(module, RowParallelLinear) and parameter_name == "weight":
        local_size = module.input_size_per_partition
        start = module.tp_rank * local_size
        return tensor_slice[:, start : start + local_size]
    return tensor_slice[:]


def _resolve_module(
    modules: dict[str, nn.Module], parameter_name: str
) -> tuple[nn.Module, str]:
    module_name, _, leaf_name = parameter_name.rpartition(".")
    try:
        return modules[module_name], leaf_name
    except KeyError as exc:
        raise KeyError(f"checkpoint parameter has no module: {parameter_name}") from exc


def load_tensor_parallel_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    tensor_transform: TensorTransform | None = None,
    strict: bool = True,
    skip_checkpoint_parameter: Callable[[str], bool] | None = None,
) -> tuple[list[str], list[str]]:
    """Load a safetensors checkpoint while materializing only rank-local shards.

    ``skip_checkpoint_parameter`` lets a caller declare that a checkpoint tensor
    is intentionally absent from this rank's module tree. Expert-parallel ranks
    use it to ignore the experts they do not own instead of reporting them as
    unexpected keys.
    """

    root = Path(checkpoint_dir)
    weight_map = _weight_map(root)
    modules = dict(model.named_modules())
    expected = dict(model.named_parameters())
    expected.update(dict(model.named_buffers()))
    loaded: set[str] = set()
    unexpected: list[str] = []

    by_file: dict[Path, list[str]] = {}
    for name, file_path in weight_map.items():
        by_file.setdefault(file_path, []).append(name)

    for file_path, names in by_file.items():
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for name in names:
                if skip_checkpoint_parameter is not None and skip_checkpoint_parameter(
                    name
                ):
                    continue
                if name not in expected:
                    unexpected.append(name)
                    continue
                module, leaf_name = _resolve_module(modules, name)
                tensor_slice = handle.get_slice(name)
                tensor = (
                    tensor_transform(name, tensor_slice, module, leaf_name)
                    if tensor_transform is not None
                    else None
                )
                if tensor is None:
                    tensor = _read_local_tensor(tensor_slice, module, leaf_name)
                target = expected[name]
                target_device = device if device is not None else target.device
                if torch.device(target_device).type == "meta":
                    target_device = "cpu"
                tensor = tensor.to(device=target_device, dtype=target.dtype)
                if tensor.shape != target.shape:
                    raise ValueError(
                        f"{name}: local checkpoint shape {tuple(tensor.shape)} "
                        f"does not match model shape {tuple(target.shape)}"
                    )
                if isinstance(target, nn.Parameter):
                    setattr(
                        module,
                        leaf_name,
                        nn.Parameter(tensor, requires_grad=target.requires_grad),
                    )
                else:
                    setattr(module, leaf_name, tensor)
                loaded.add(name)

    missing = sorted(set(expected) - loaded)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}"
        )
    post_load = getattr(model, "post_load_weights", None)
    if callable(post_load):
        post_load()
    return missing, sorted(unexpected)

