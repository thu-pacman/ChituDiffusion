"""Tensor-only adapters for tensor-parallel linear layers."""

from __future__ import annotations

import torch

from .linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


class PlainColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel linear with the same return convention as ``nn.Linear``."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(inputs)[0]


class PlainMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged column-parallel linear returning only its output tensor."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(inputs)[0]


class PlainRowParallelLinear(RowParallelLinear):
    """Row-parallel linear with the same return convention as ``nn.Linear``."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(inputs)[0]


__all__ = [
    "PlainColumnParallelLinear",
    "PlainMergedColumnParallelLinear",
    "PlainRowParallelLinear",
]
