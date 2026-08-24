from __future__ import annotations

import torch
from torch import nn


class MiniMaxH3Rope(nn.Module):
    def __init__(
        self,
        inv_freq_len: int = 16,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(
                    0,
                    inv_freq_len * 2,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / (inv_freq_len * 2)
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        if position_ids.ndim == 3:
            if position_ids.shape[0] != 1:
                raise ValueError("position_ids batch dimension must be one")
            position_ids = position_ids[0]
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError("position_ids must be [tokens, 3] or [1, tokens, 3]")
        per_axis = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq
        half = torch.cat(tuple(per_axis.unbind(dim=1)), dim=-1)
        return torch.cat((half, half), dim=-1)


def apply_rope(
    tensor: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = frequencies.shape[-1]
    rotary, passthrough = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
    half = rotary_dim // 2
    cos = frequencies[..., :half].cos()
    sin = frequencies[..., :half].sin()
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(1).to(rotary.dtype)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(1).to(rotary.dtype)
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1)

