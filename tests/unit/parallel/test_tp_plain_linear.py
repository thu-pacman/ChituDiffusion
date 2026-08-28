import torch
import torch.nn.functional as F

from chitu_diffusion.parallel.tp import (
    PlainColumnParallelLinear,
    PlainMergedColumnParallelLinear,
    PlainRowParallelLinear,
)


def test_plain_tp_linear_adapters_return_only_the_output_tensor() -> None:
    inputs = torch.randn(3, 4)
    layers = (
        PlainColumnParallelLinear(4, 6),
        PlainMergedColumnParallelLinear(4, (2, 4)),
        PlainRowParallelLinear(4, 6),
    )

    for layer in layers:
        output = layer(inputs)

        assert isinstance(output, torch.Tensor)
        torch.testing.assert_close(output, F.linear(inputs, layer.weight, layer.bias))
