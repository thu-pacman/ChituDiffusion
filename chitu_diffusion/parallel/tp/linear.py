from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .topology import get_tp_group, get_tp_rank, get_tp_world_size


def _divide(value: int, degree: int, name: str) -> int:
    if value % degree:
        raise ValueError(f"{name}={value} must be divisible by TP degree {degree}")
    return value // degree


def _initialize_weight(weight: nn.Parameter) -> None:
    if not weight.is_meta:
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class ReplicatedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            dtype=params_dtype,
            device=device,
        )

    @torch.no_grad()
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int | None = None,
    ) -> None:
        del shard_id
        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"replicated parameter shape {tuple(param.shape)} does not match "
                f"checkpoint shape {tuple(loaded_weight.shape)}"
            )
        param.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        tp_group: object | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tp_group = tp_group if tp_group is not None else get_tp_group()
        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()
        self.output_size_per_partition = _divide(
            self.out_features, self.tp_size, "out_features"
        )
        self.gather_output = bool(gather_output)
        self.skip_bias_add = bool(skip_bias_add)
        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.in_features,
                dtype=params_dtype,
                device=device,
            )
        )
        _initialize_weight(self.weight)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    dtype=params_dtype,
                    device=device,
                )
            )
            if not self.bias.is_meta:
                bound = 1 / math.sqrt(self.in_features)
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int | None = None,
    ) -> None:
        del shard_id
        shard_size = param.shape[0]
        if loaded_weight.shape == param.shape:
            shard = loaded_weight
        else:
            shard = loaded_weight.narrow(0, self.tp_rank * shard_size, shard_size)
        if shard.shape != param.shape:
            raise ValueError(
                f"column shard shape {tuple(shard.shape)} != {tuple(param.shape)}"
            )
        param.copy_(shard.to(device=param.device, dtype=param.dtype))

    def _gather(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return output
        pieces = [torch.empty_like(output) for _ in range(self.tp_size)]
        dist.all_gather(pieces, output.contiguous(), group=self.tp_group)
        return torch.cat(pieces, dim=-1).contiguous()

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, nn.Parameter | None]:
        bias = None if self.skip_bias_add else self.bias
        output = F.linear(inputs, self.weight, bias)
        if self.gather_output:
            output = self._gather(output)
        return output, self.bias if self.skip_bias_add else None


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        in_features: int,
        output_sizes: Sequence[int],
        bias: bool = True,
        *,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        tp_group: object | None = None,
    ) -> None:
        self.output_sizes = tuple(int(size) for size in output_sizes)
        super().__init__(
            in_features,
            sum(self.output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            device=device,
            tp_group=tp_group,
        )
        self.output_partition_sizes = tuple(
            _divide(size, self.tp_size, "merged output size")
            for size in self.output_sizes
        )

    @torch.no_grad()
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int | None = None,
    ) -> None:
        if shard_id is not None:
            if not 0 <= shard_id < len(self.output_sizes):
                raise ValueError(f"invalid merged shard id {shard_id}")
            local_offset = sum(self.output_partition_sizes[:shard_id])
            local_size = self.output_partition_sizes[shard_id]
            target = param.narrow(0, local_offset, local_size)
            if loaded_weight.shape == target.shape:
                shard = loaded_weight
            else:
                shard = loaded_weight.narrow(
                    0, self.tp_rank * local_size, local_size
                )
            target.copy_(shard.to(device=target.device, dtype=target.dtype))
            return

        if loaded_weight.shape == param.shape:
            param.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))
            return
        checkpoint_offset = 0
        for logical_id, logical_size in enumerate(self.output_sizes):
            logical = loaded_weight.narrow(0, checkpoint_offset, logical_size)
            self.weight_loader(param, logical, logical_id)
            checkpoint_offset += logical_size

    def _gather(self, output: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return output
        pieces = [torch.empty_like(output) for _ in range(self.tp_size)]
        dist.all_gather(pieces, output.contiguous(), group=self.tp_group)
        logical_outputs = []
        for logical_id, local_size in enumerate(self.output_partition_sizes):
            local_offset = sum(self.output_partition_sizes[:logical_id])
            logical_outputs.append(
                torch.cat(
                    [
                        piece.narrow(-1, local_offset, local_size)
                        for piece in pieces
                    ],
                    dim=-1,
                )
            )
        return torch.cat(logical_outputs, dim=-1).contiguous()


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        tp_group: object | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tp_group = tp_group if tp_group is not None else get_tp_group()
        self.tp_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()
        self.input_size_per_partition = _divide(
            self.in_features, self.tp_size, "in_features"
        )
        self.input_is_parallel = bool(input_is_parallel)
        self.reduce_results = bool(reduce_results)
        self.skip_bias_add = bool(skip_bias_add)
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.input_size_per_partition,
                dtype=params_dtype,
                device=device,
            )
        )
        _initialize_weight(self.weight)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, dtype=params_dtype, device=device)
            )
            if not self.bias.is_meta:
                bound = 1 / math.sqrt(self.in_features)
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: int | None = None,
    ) -> None:
        del shard_id
        if param is self.bias:
            if loaded_weight.shape != param.shape:
                raise ValueError("row-parallel bias is replicated")
            param.copy_(loaded_weight.to(device=param.device, dtype=param.dtype))
            return
        shard_size = param.shape[1]
        if loaded_weight.shape == param.shape:
            shard = loaded_weight
        else:
            shard = loaded_weight.narrow(1, self.tp_rank * shard_size, shard_size)
        if shard.shape != param.shape:
            raise ValueError(
                f"row shard shape {tuple(shard.shape)} != {tuple(param.shape)}"
            )
        param.copy_(shard.to(device=param.device, dtype=param.dtype))

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, nn.Parameter | None]:
        if not self.input_is_parallel and self.tp_size > 1:
            inputs = inputs.chunk(self.tp_size, dim=-1)[self.tp_rank].contiguous()
        output = F.linear(inputs, self.weight, None)
        if self.reduce_results and self.tp_size > 1:
            dist.all_reduce(output, group=self.tp_group)
        if not self.skip_bias_add and self.bias is not None:
            output = output + self.bias
        return output, self.bias if self.skip_bias_add else None

