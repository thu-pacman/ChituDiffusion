"""RMS normalization over a dimension that tensor parallelism has split.

Most norms in a transformer sit on the full hidden state and can simply be
replicated. The exception is a norm applied to a column-parallel layer's output
before the matching row-parallel layer -- Wan normalizes Q and K across all
heads at once, so after the heads are split no rank holds the whole vector the
scale is computed from.

The fix is one reduction of the shard sums of squares. It carries a single
scalar per token, which is the hidden size smaller than the row-parallel
all-reduce already happening a few operations later.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from .linear import _divide
from .topology import get_tp_group, get_tp_rank, get_tp_world_size


class TensorParallelRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_size: int,
        eps: float,
        *,
        elementwise_affine: bool = True,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        tp_group: object | None = None,
    ) -> None:
        super().__init__()
        self.normalized_size = int(normalized_size)
        self.eps = float(eps)
        self.tp_group = tp_group if tp_group is not None else get_tp_group()
        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()
        self.size_per_partition = _divide(
            self.normalized_size, self.tp_size, "normalized_size"
        )
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(
                    self.size_per_partition, dtype=params_dtype, device=device
                )
            )
            if not self.weight.is_meta:
                nn.init.ones_(self.weight)
        else:
            self.register_parameter("weight", None)

    @classmethod
    def from_dense(
        cls, source: nn.Module, *, tp_group: object | None = None
    ) -> "TensorParallelRMSNorm":
        weight = getattr(source, "weight", None)
        shape = getattr(source, "normalized_shape", None)
        if shape is None:
            if weight is None:
                raise TypeError(f"{type(source).__name__} has no normalized shape")
            shape = tuple(weight.shape)
        if len(tuple(shape)) != 1:
            raise ValueError(
                f"{type(source).__name__} normalizes over {tuple(shape)}; tensor "
                "parallelism can only split a single dimension"
            )
        eps = getattr(source, "eps", None)
        if eps is None:
            raise ValueError(f"{type(source).__name__} does not expose eps")
        return cls(
            int(tuple(shape)[0]),
            float(eps),
            elementwise_affine=weight is not None,
            params_dtype=None if weight is None else weight.dtype,
            device=None if weight is None else weight.device,
            tp_group=tp_group,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        upcast = inputs.float()
        squares = upcast.pow(2).sum(-1, keepdim=True)
        if self.tp_size > 1:
            dist.all_reduce(squares, group=self.tp_group)
        scaled = upcast * torch.rsqrt(squares / self.normalized_size + self.eps)
        if self.weight is not None:
            scaled = scaled * self.weight.float()
        return scaled.to(inputs.dtype)

    def extra_repr(self) -> str:
        return (
            f"{self.size_per_partition} of {self.normalized_size}, "
            f"eps={self.eps}, tp={self.tp_size}"
        )


__all__ = ["TensorParallelRMSNorm"]
