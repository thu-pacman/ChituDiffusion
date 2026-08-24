from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from chitu_diffusion.parallel.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)

from .qwen3vl_config import MiniMaxH3Qwen3VLConfig


def _rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Qwen3VLRMSNorm(nn.Module):
    def __init__(
        self,
        size: int,
        eps: float,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=dtype, device=device))
        self.eps = float(eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(inputs.dtype) * self.weight


class Qwen3VLRotaryEmbedding(nn.Module):
    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = config.mrope_section

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = (
                position_ids.unsqueeze(1)
                if position_ids.shape[0] == 3
                else position_ids.unsqueeze(0).expand(3, -1, -1)
            )
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("position_ids must have shape [batch, length] or [3, batch, length]")
        frequencies = (
            position_ids[:, :, :, None].float()
            * self.inv_freq.to(position_ids.device)[None, None, None, :]
        )
        interleaved = frequencies[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            stop = self.mrope_section[axis] * 3
            interleaved[..., offset:stop:3] = frequencies[
                axis, ..., offset:stop:3
            ]
        angles = torch.cat((interleaved, interleaved), dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)


class Qwen3VLAttention(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3Qwen3VLConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.qkv_proj = MergedColumnParallelLinear(
            config.hidden_size,
            (q_width, kv_width, kv_width),
            bias=config.attention_bias,
            params_dtype=torch.bfloat16,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            q_width,
            config.hidden_size,
            bias=config.attention_bias,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            device=device,
        )
        self.q_norm = Qwen3VLRMSNorm(
            config.head_dim, config.rms_norm_eps, device=device
        )
        self.k_norm = Qwen3VLRMSNorm(
            config.head_dim, config.rms_norm_eps, device=device
        )
        self.head_dim = config.head_dim
        self.local_q_heads = config.num_attention_heads // self.qkv_proj.tp_size
        self.local_kv_heads = config.num_key_value_heads // self.qkv_proj.tp_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.local_q_heads * self.head_dim
        kv_size = self.local_kv_heads * self.head_dim
        query, key, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        batch, length, _ = query.shape
        query = self.q_norm(query.view(batch, length, self.local_q_heads, -1))
        key = self.k_norm(key.view(batch, length, self.local_kv_heads, -1))
        value = value.view(batch, length, self.local_kv_heads, -1)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        query, key, value = (
            tensor.transpose(1, 2) for tensor in (query, key, value)
        )

        additive_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, length]")
            additive_mask = torch.zeros(
                batch, 1, length, length, dtype=query.dtype, device=query.device
            )
            additive_mask.masked_fill_(
                ~attention_mask[:, None, None, :].bool(),
                torch.finfo(query.dtype).min,
            )
            causal = torch.ones(
                length, length, dtype=torch.bool, device=query.device
            ).triu(1)
            additive_mask.masked_fill_(causal, torch.finfo(query.dtype).min)

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=additive_mask,
            is_causal=attention_mask is None,
            scale=1.0 / math.sqrt(self.head_dim),
            enable_gqa=self.local_q_heads != self.local_kv_heads,
        )
        output = output.transpose(1, 2).reshape(batch, length, q_size)
        output, _ = self.o_proj(output)
        return output


class Qwen3VLMLP(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3Qwen3VLConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            (config.intermediate_size, config.intermediate_size),
            bias=False,
            params_dtype=torch.bfloat16,
            device=device,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            device=device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(inputs)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3Qwen3VLConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3VLAttention(config, device=device)
        self.mlp = Qwen3VLMLP(config, device=device)
        self.input_layernorm = Qwen3VLRMSNorm(
            config.hidden_size, config.rms_norm_eps, device=device
        )
        self.post_attention_layernorm = Qwen3VLRMSNorm(
            config.hidden_size, config.rms_norm_eps, device=device
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), cos, sin, attention_mask
        )
        return hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )


class MiniMaxH3Qwen3VLEncoder(nn.Module):
    """Native H3 encoder: embedding plus decoder layers 0..49, without final norm."""

    vision_supported = True

    def __init__(
        self,
        config: MiniMaxH3Qwen3VLConfig,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        from chitu_diffusion.parallel.tensor_parallel import get_tp_world_size

        config.validate_tensor_parallel(get_tp_world_size())
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        self.layers = nn.ModuleList(
            Qwen3VLDecoderLayer(config, device=device)
            for _ in range(config.num_hidden_layers)
        )
        self.rotary_emb = Qwen3VLRotaryEmbedding(config)
        self.visual: nn.Module | None = None
        if config.vision_config is not None:
            # Lazy import keeps prompt-only users independent from the rapidly
            # evolving Qwen3-VL Transformers implementation.
            from transformers.models.qwen3_vl.configuration_qwen3_vl import (
                Qwen3VLVisionConfig,
            )
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLVisionModel,
            )

            vision_payload = dict(config.vision_config)
            vision_payload.pop("model_type", None)
            vision_config = Qwen3VLVisionConfig(**vision_payload)
            construction_device = device if device is not None else "cpu"
            with torch.device(construction_device):
                self.visual = Qwen3VLVisionModel(vision_config)
            self.visual.to(dtype=torch.bfloat16)
            rotary = self.visual.rotary_pos_emb
            if rotary.inv_freq.is_meta:
                rotary.inv_freq = 1.0 / (
                    rotary.theta
                    ** (
                        torch.arange(0, rotary.dim, 2, dtype=torch.float32)
                        / rotary.dim
                    )
                )

    def _vision_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.visual is None:
            raise RuntimeError("this encoder was constructed without a vision tower")
        parameter = next(self.visual.parameters())
        visual_output = self.visual(
            pixel_values.to(device=parameter.device, dtype=parameter.dtype),
            grid_thw=image_grid_thw.to(parameter.device),
        )
        pooled = visual_output.pooler_output
        deepstack = list(visual_output.deepstack_features)
        return pooled, deepstack

    @staticmethod
    def _insert_visual_features(
        hidden_states: torch.Tensor,
        visual_mask: torch.Tensor,
        visual_features: torch.Tensor,
        *,
        operation: str,
    ) -> torch.Tensor:
        expected = int(visual_mask.sum())
        if visual_features.ndim != 2:
            raise ValueError(f"{operation} features must have shape [tokens, hidden]")
        if visual_features.shape != (expected, hidden_states.shape[-1]):
            raise ValueError(
                f"{operation} feature shape {tuple(visual_features.shape)} does "
                f"not match {expected} image tokens and hidden size "
                f"{hidden_states.shape[-1]}"
            )
        output = hidden_states.clone()
        features = visual_features.to(output.device, output.dtype)
        if operation == "replace":
            output[visual_mask] = features
        elif operation == "add":
            output[visual_mask] = output[visual_mask] + features
        else:
            raise ValueError(f"unsupported visual operation {operation!r}")
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together")
        squeeze_batch = input_ids.ndim == 1
        if squeeze_batch:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [length] or [batch, length]")
        batch, length = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(
                length, dtype=torch.long, device=input_ids.device
            ).expand(batch, -1)
        elif (
            position_ids.ndim == 2
            and position_ids.shape == (3, length)
            and batch == 1
        ):
            position_ids = position_ids.unsqueeze(1)
        position_ids = position_ids.to(input_ids.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)
        hidden_states = self.embed_tokens(input_ids)
        visual_mask = input_ids == self.config.image_token_id
        deepstack_features: list[torch.Tensor] = []
        if pixel_values is not None:
            if batch != 1:
                raise ValueError("image-conditioned H3 encoding currently requires batch=1")
            pooled, deepstack_features = self._vision_features(
                pixel_values, image_grid_thw
            )
            hidden_states = self._insert_visual_features(
                hidden_states,
                visual_mask,
                pooled,
                operation="replace",
            )
        elif visual_mask.any():
            raise ValueError("image placeholders require pixel_values")
        cos, sin = self.rotary_emb(position_ids, hidden_states.dtype)
        if len(deepstack_features) > len(self.layers):
            raise ValueError("vision tower returned more DeepStack features than layers")
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, cos, sin, attention_mask
            )
            if layer_index < len(deepstack_features):
                hidden_states = self._insert_visual_features(
                    hidden_states,
                    visual_mask,
                    deepstack_features[layer_index],
                    operation="add",
                )
        hidden_states = hidden_states.to(torch.bfloat16)
        return hidden_states[0] if squeeze_batch else hidden_states


__all__ = [
    "MiniMaxH3Qwen3VLEncoder",
    "Qwen3VLRMSNorm",
    "Qwen3VLRotaryEmbedding",
]
