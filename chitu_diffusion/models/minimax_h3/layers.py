from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from chitu_diffusion.parallel.tp.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)

from .config import MiniMaxH3DiTConfig


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = config.timestep_input_dim
        self.proj_in = ColumnParallelLinear(
            config.timestep_input_dim,
            config.time_embed_hidden_size,
            params_dtype=torch.float32,
            device=device,
        )
        self.proj_out = RowParallelLinear(
            config.time_embed_hidden_size,
            config.time_embed_dim,
            input_is_parallel=True,
            params_dtype=torch.float32,
            device=device,
        )
        self.register_buffer("_frequency_cache", None, persistent=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = self._frequency_cache
        if frequencies is None or frequencies.device != timesteps.device:
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(
                    half, dtype=torch.float32, device=timesteps.device
                )
                / half
            )
            self._frequency_cache = frequencies
        angles = timesteps.to(torch.float32).reshape(-1, 1) * frequencies
        hidden, _ = self.proj_in(
            torch.cat((angles.cos(), angles.sin()), dim=-1)
        )
        output, _ = self.proj_out(F.silu(hidden))
        return output


class MiniMaxH3MLP(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = MergedColumnParallelLinear(
            config.hidden_size,
            (config.ffn_hidden_size, config.ffn_hidden_size),
            bias=False,
            params_dtype=torch.bfloat16,
            device=device,
        )
        self.fc2 = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            device=device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.fc1(inputs)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.fc2(F.silu(gate) * up)
        return output


class MiniMaxH3AdalnProj(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        out_features: int,
        *,
        expand_ratio: int,
        modality_count: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        expected = expand_ratio * modality_count * config.hidden_size
        if out_features != expected:
            raise ValueError(f"AdaLN output {out_features} != expected {expected}")
        self.expand_ratio = expand_ratio
        self.modality_count = modality_count
        self.hidden_size = config.hidden_size
        self.linear = ColumnParallelLinear(
            config.time_embed_dim,
            out_features,
            gather_output=True,
            params_dtype=torch.bfloat16,
            device=device,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output, _ = self.linear(inputs)
        output = output.view(
            inputs.shape[0] * self.modality_count,
            self.expand_ratio * self.hidden_size,
        )
        return tuple(output.chunk(self.expand_ratio, dim=-1))


def modulate_scale_shift(
    inputs: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    selected_shift = shift.index_select(0, indices).to(inputs.dtype)
    selected_scale = scale.index_select(0, indices).to(inputs.dtype)
    return inputs * (1 + selected_scale) + selected_shift


def modulate_gate(
    residual: torch.Tensor,
    gate: torch.Tensor,
    hidden: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return residual + gate.index_select(0, indices).to(hidden.dtype) * hidden

