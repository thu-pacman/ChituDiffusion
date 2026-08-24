from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from chitu_diffusion.parallel import (
    ColumnParallelLinear,
    EpeParallelContext,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


def main() -> None:
    dist.init_process_group("gloo")
    world_size = dist.get_world_size()
    context = EpeParallelContext.from_torchrun(
        allowed_widths=(1,),
        tensor_parallel_degree=world_size,
        owns_process_group=False,
    )
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(3, 8, generator=generator)

    column_weight = torch.randn(12, 8, generator=generator)
    column_bias = torch.randn(12, generator=generator)
    column = ColumnParallelLinear(8, 12, gather_output=True)
    column.weight_loader(column.weight, column_weight)
    column.weight_loader(column.bias, column_bias)
    column_output, _ = column(inputs)
    torch.testing.assert_close(
        column_output, F.linear(inputs, column_weight, column_bias)
    )

    row_weight = torch.randn(6, 8, generator=generator)
    row_bias = torch.randn(6, generator=generator)
    row = RowParallelLinear(8, 6, input_is_parallel=False)
    row.weight_loader(row.weight, row_weight)
    row.weight_loader(row.bias, row_bias)
    row_output, _ = row(inputs)
    torch.testing.assert_close(row_output, F.linear(inputs, row_weight, row_bias))

    merged_weight = torch.randn(16, 8, generator=generator)
    merged = MergedColumnParallelLinear(
        8, (6, 10), bias=False, gather_output=True
    )
    merged.weight_loader(merged.weight, merged_weight)
    merged_output, _ = merged(inputs)
    torch.testing.assert_close(merged_output, F.linear(inputs, merged_weight))
    if dist.get_rank() == 0:
        print("TP linear parity passed")
    context.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

