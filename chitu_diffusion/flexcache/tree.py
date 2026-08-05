from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import torch

TensorTree = Any


def tree_map(function: Callable[[torch.Tensor], torch.Tensor], tree: TensorTree) -> TensorTree:
    if isinstance(tree, torch.Tensor):
        return function(tree)
    if isinstance(tree, tuple):
        return type(tree)(*(tree_map(function, value) for value in tree)) if hasattr(tree, "_fields") else tuple(
            tree_map(function, value) for value in tree
        )
    if isinstance(tree, list):
        return [tree_map(function, value) for value in tree]
    if isinstance(tree, Mapping):
        values = {key: tree_map(function, value) for key, value in tree.items()}
        try:
            return type(tree)(**values)
        except TypeError:
            return type(tree)(values)
    return tree


def tree_zip_map(
    function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    left: TensorTree,
    right: TensorTree,
) -> TensorTree:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return function(left, right)
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        values = (tree_zip_map(function, a, b) for a, b in zip(left, right))
        return type(left)(*values) if hasattr(left, "_fields") else tuple(values)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [tree_zip_map(function, a, b) for a, b in zip(left, right)]
    if isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys():
        values = {
            key: tree_zip_map(function, left[key], right[key]) for key in left
        }
        try:
            return type(left)(**values)
        except TypeError:
            return type(left)(values)
    if _is_leaf(left) and _is_leaf(right):
        # Non-tensor metadata rides along unchanged, mirroring tree_map.
        return left
    raise _mismatch(left, right)


def _is_leaf(value: TensorTree) -> bool:
    return not isinstance(value, (torch.Tensor, tuple, list, Mapping))


def tree_clone(tree: TensorTree) -> TensorTree:
    return tree_map(lambda tensor: tensor.detach().clone(), tree)


def tree_add(left: TensorTree, right: TensorTree) -> TensorTree:
    return tree_zip_map(torch.add, left, right)


def tree_sub(left: TensorTree, right: TensorTree) -> TensorTree:
    return tree_zip_map(torch.sub, left, right)


def tree_mul(tree: TensorTree, scalar: float) -> TensorTree:
    return tree_map(lambda tensor: tensor * scalar, tree)


def tree_nbytes(tree: TensorTree) -> int:
    total = 0

    def count(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total
        total += tensor.numel() * tensor.element_size()
        return tensor

    tree_map(count, tree)
    return total


def _mismatch(left: TensorTree, right: TensorTree) -> TypeError:
    return TypeError(
        "cannot combine mismatched tensor trees: "
        f"{type(left).__name__} vs {type(right).__name__}"
    )


def first_tensor(tree: TensorTree) -> torch.Tensor | None:
    if isinstance(tree, torch.Tensor):
        return tree
    if isinstance(tree, (tuple, list)):
        for value in tree:
            found = first_tensor(value)
            if found is not None:
                return found
    elif isinstance(tree, Mapping):
        for value in tree.values():
            found = first_tensor(value)
            if found is not None:
                return found
    return None


