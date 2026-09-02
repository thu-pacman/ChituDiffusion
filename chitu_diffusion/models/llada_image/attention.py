from __future__ import annotations

from typing import Any

import torch

from ...parallel.cp import EpeParallelContext, ImageContextParallelAttention
from .diffusers_components import LLaDAImageAttention


class LLaDAImageCpAttnProcessor:
    """Route an image-prefix sequence through Chitu context parallelism.

    The processor remains a transparent wrapper until :meth:`enable` is called.
    This keeps the width-one path on the exact Diffusers attention processor,
    including any backend selected by Diffusers.
    """

    def __init__(
        self,
        parallel: EpeParallelContext,
        *,
        mode: str = "agkv",
        native_processor: Any,
    ) -> None:
        self.parallel = parallel
        self.context_parallel = ImageContextParallelAttention(mode=mode)
        self.native_processor = native_processor
        self._image_tokens = 0
        self._enabled = False

    @property
    def mode(self) -> str:
        return self.context_parallel.mode

    def enable(self, image_tokens: int) -> None:
        image_tokens = int(image_tokens)
        if image_tokens <= 0:
            raise ValueError("image_tokens must be positive")
        self._image_tokens = image_tokens
        self._enabled = self.parallel.active.width > 1

    def disable(self) -> None:
        self._enabled = False
        self._image_tokens = 0

    @staticmethod
    def _apply_rotary(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x_in.device.type, enabled=False):
            value = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            value = torch.view_as_real(value * freqs_cis.unsqueeze(2)).flatten(3)
        return value.to(dtype=x_in.dtype)

    def __call__(
        self,
        attn: LLaDAImageAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._enabled:
            return self.native_processor(
                attn,
                hidden_states,
                attention_mask,
                freqs_cis,
            )

        if attention_mask is not None and not bool(attention_mask.to(torch.bool).all()):
            raise NotImplementedError(
                "LLaDA-Image context parallelism does not support outer ragged masks"
            )
        if self._image_tokens > hidden_states.shape[1]:
            raise ValueError(
                f"image prefix has {self._image_tokens} tokens, but the sequence "
                f"contains only {hidden_states.shape[1]}"
            )

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, attn.head_dim))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
            key = attn.norm_k(key)
        if freqs_cis is not None:
            query = self._apply_rotary(query, freqs_cis)
            key = self._apply_rotary(key, freqs_cis)

        output_dtype = query.dtype
        query = query.to(value.dtype).contiguous()
        key = key.to(value.dtype).contiguous()
        value = value.contiguous()
        image_tokens = self._image_tokens
        joint_output, image_output = self.context_parallel(
            query[:, :image_tokens],
            key[:, :image_tokens],
            value[:, :image_tokens],
            query[:, image_tokens:],
            key[:, image_tokens:],
            value[:, image_tokens:],
            lane_process_group=self.parallel.active.process_group,
            ulysses_topology=(
                self.parallel.active_ulysses if self.mode == "ulysses" else None
            ),
            agkv_transport=(
                self.parallel.active_agkv_transport if self.mode == "agkv" else None
            ),
        )
        output = torch.cat([image_output, joint_output], dim=1)
        output = output.flatten(2, 3).to(output_dtype)
        return attn.to_out[0](output)
