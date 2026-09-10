"""Layer-wise spatial VAE decoding with global normalization and attention.

Each rank owns complete channels and a disjoint band of rows. Convolutions
exchange only their boundary rows, norms exchange moments, and attention
computes local queries against global keys/values. No overlap is blended.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .topology import VaeParallelTopology


@dataclass(frozen=True)
class _DecodeContext:
    group: Any
    rank: int
    ranks: tuple[int, ...]
    rows: tuple[int, ...]

    @property
    def width(self) -> int:
        return len(self.rows)

    def lengths(self, local_extent: int) -> tuple[int, ...]:
        factor, remainder = divmod(local_extent, self.rows[self.rank])
        if remainder:
            raise RuntimeError("VAE layer changed the spatial shard grid")
        return tuple(length * factor for length in self.rows)

    def halo(self, x: torch.Tensor, radius: int) -> torch.Tensor:
        if radius == 0:
            return x
        if min(self.lengths(x.shape[-2])) < radius:
            raise ValueError("VAE shards are smaller than a convolution halo")
        shape = list(x.shape)
        shape[-2] = radius
        above, below = x.new_zeros(shape), x.new_zeros(shape)
        ops = []
        sends = []
        if self.rank > 0:
            sends.append(x[..., :radius, :].contiguous())
            peer = self.ranks[self.rank - 1]
            ops.extend(
                (
                    dist.P2POp(dist.isend, sends[-1], peer, self.group),
                    dist.P2POp(dist.irecv, above, peer, self.group),
                )
            )
        if self.rank + 1 < self.width:
            sends.append(x[..., -radius:, :].contiguous())
            peer = self.ranks[self.rank + 1]
            ops.extend(
                (
                    dist.P2POp(dist.isend, sends[-1], peer, self.group),
                    dist.P2POp(dist.irecv, below, peer, self.group),
                )
            )
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        return torch.cat((above, x, below), dim=-2)

    def gather_kv(self, key: torch.Tensor, value: torch.Tensor):
        lengths = self.lengths(key.shape[-2])
        local = torch.stack((key, value), dim=0)
        padding = max(lengths) - key.shape[-2]
        if padding:
            local = F.pad(local, (0, 0, 0, padding))
        buffers = [torch.empty_like(local) for _ in self.rows]
        dist.all_gather(buffers, local.contiguous(), group=self.group)
        whole = torch.cat(
            [part[..., :length, :] for part, length in zip(buffers, lengths)], dim=-2
        )
        return whole.unbind(0)

    def gather_output(self, output: torch.Tensor) -> torch.Tensor | None:
        lengths = self.lengths(output.shape[-2])
        output = output.contiguous()
        if self.rank != 0:
            for work in dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, output, self.ranks[0], self.group)]
            ):
                work.wait()
            return None
        pieces = [output]
        ops = []
        for peer, length in zip(self.ranks[1:], lengths[1:]):
            shape = list(output.shape)
            shape[-2] = length
            piece = output.new_empty(shape)
            pieces.append(piece)
            ops.append(dist.P2POp(dist.irecv, piece, peer, self.group))
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        return torch.cat(pieces, dim=-2)


def _group_norm(module: nn.GroupNorm, x: torch.Tensor, context: _DecodeContext):
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    grouped = x.reshape(x.shape[0], module.num_groups, -1).to(dtype)
    variance, mean = torch.var_mean(grouped, dim=-1, correction=0)
    moments = torch.stack((mean, variance))
    gathered = [torch.empty_like(moments) for _ in context.rows]
    dist.all_gather(gathered, moments, group=context.group)
    # Combining local central moments avoids E[x^2] - E[x]^2 cancellation.
    weights = moments.new_tensor(context.rows) / sum(context.rows)
    means, variances = torch.stack(gathered, dim=1).unbind(0)
    weights = weights[:, None, None]
    mean = (means * weights).sum(0)
    variance = ((variances + (means - mean).square()) * weights).sum(0)
    del grouped, gathered
    if x.is_cuda:
        # Inference BatchNorm fuses normalization and the channel affine. Fold
        # batch into channels and supply each channel's global group moments;
        # it performs no batch-statistics computation or running-state update.
        batch, channels = x.shape[:2]
        repeats = channels // module.num_groups
        normalized = F.batch_norm(
            x.reshape(1, batch * channels, -1),
            mean.repeat_interleave(repeats, dim=1).flatten(),
            variance.repeat_interleave(repeats, dim=1).flatten(),
            module.weight.to(dtype).repeat(batch)
            if module.weight is not None
            else None,
            module.bias.to(dtype).repeat(batch) if module.bias is not None else None,
            training=False,
            eps=module.eps,
        )
        return normalized.reshape(x.shape)
    inv_std = torch.rsqrt(variance + module.eps)
    shape = (x.shape[0], module.num_groups, -1)
    normalized = (x.reshape(shape).to(dtype) - mean[..., None]) * inv_std[..., None]
    normalized = normalized.reshape(x.shape)
    affine_shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
    if module.weight is not None:
        normalized = normalized * module.weight.to(dtype).view(affine_shape)
    if module.bias is not None:
        normalized = normalized + module.bias.to(dtype).view(affine_shape)
    return normalized.to(x.dtype)


def _attention(module: nn.Module, x: torch.Tensor, context: _DecodeContext):
    residual = x
    batch, channels, height, width = x.shape
    hidden = x.reshape(batch, channels, height * width).transpose(1, 2)
    if module.group_norm is not None:
        hidden = module.group_norm(hidden.transpose(1, 2)).transpose(1, 2)
    heads = module.heads
    query, key, value = (
        projection(hidden)
        .reshape(batch, -1, heads, module.inner_dim // heads)
        .transpose(1, 2)
        for projection in (module.to_q, module.to_k, module.to_v)
    )
    if module.norm_q is not None:
        query = module.norm_q(query)
    if module.norm_k is not None:
        key = module.norm_k(key)
    key, value = context.gather_kv(key, value)
    hidden = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    hidden = hidden.transpose(1, 2).reshape(batch, -1, module.inner_dim).to(query.dtype)
    hidden = module.to_out[1](module.to_out[0](hidden))
    hidden = hidden.transpose(1, 2).reshape(batch, channels, height, width)
    if module.residual_connection:
        hidden = hidden + residual
    return hidden / module.rescale_output_factor


def _video_attention(module: nn.Module, x: torch.Tensor, context: _DecodeContext):
    batch, channels, frames, height, width = x.shape
    hidden = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    hidden = module.norm(hidden)
    qkv = module.to_qkv(hidden).reshape(batch * frames, 1, channels * 3, -1)
    query, key, value = qkv.permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
    key, value = context.gather_kv(key, value)
    hidden = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    hidden = (
        hidden.squeeze(1)
        .permute(0, 2, 1)
        .reshape(batch * frames, channels, height, width)
    )
    hidden = module.proj(hidden).reshape(batch, frames, channels, height, width)
    return hidden.permute(0, 2, 1, 3, 4) + x


class _ParallelDecoder:
    """Preserve module identities/cache accounting during spatial decoding."""

    def __init__(self, decoder: nn.Module):
        self.decoder = decoder
        self.active: ContextVar[_DecodeContext | None] = ContextVar(
            "vae_decode", default=None
        )
        self.radius = 0
        self._validate(decoder)
        self._adapt(decoder)

    def _validate(self, decoder: nn.Module) -> None:
        from diffusers.models.attention_processor import Attention, AttnProcessor2_0
        from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
            QwenImageDecoder3d,
        )
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanDecoder3d
        from diffusers.models.autoencoders.vae import Decoder

        if type(decoder) not in (Decoder, WanDecoder3d, QwenImageDecoder3d):
            raise NotImplementedError(
                f"no exact spatial adapter for {type(decoder).__name__}"
            )
        if decoder.training:
            raise ValueError("parallel VAE decoding requires eval mode")
        for module in decoder.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                radius = module.dilation[-2] * (module.kernel_size[-2] - 1) // 2
                padding = getattr(module, "_padding", None)
                height_padding = (
                    padding[2] if padding is not None else module.padding[-2]
                )
                if (
                    module.stride[-2] != 1
                    or module.kernel_size[-2] % 2 != 1
                    or height_padding != radius
                    or module.padding_mode != "zeros"
                ):
                    raise NotImplementedError(
                        "exact VAE rows require centered unit-stride convolutions"
                    )
                self.radius = max(self.radius, radius)
            if isinstance(module, Attention) and (
                type(module.processor) is not AttnProcessor2_0
                or module.spatial_norm is not None
                or module.is_cross_attention
                or module.added_kv_proj_dim is not None
            ):
                raise NotImplementedError(
                    "exact VAE rows require standard SDPA self-attention"
                )
            if isinstance(module, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
                raise NotImplementedError(
                    "transposed VAE convolutions need a spatial adapter"
                )

    def _adapt(self, decoder: nn.Module) -> None:
        from diffusers.models.attention_processor import Attention
        from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
            QwenImageAttentionBlock,
        )
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock

        for module in decoder.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                self._adapt_conv(module)
            elif isinstance(module, nn.GroupNorm):
                self._adapt_forward(module, _group_norm)
            elif isinstance(module, Attention):
                self._adapt_forward(module, _attention)
            elif isinstance(module, (WanAttentionBlock, QwenImageAttentionBlock)):
                self._adapt_forward(module, _video_attention)

    def _adapt_forward(self, module: nn.Module, operation: Callable) -> None:
        original = module.forward

        def forward(x, *args, **kwargs):
            context = self.active.get()
            if context is None:
                return original(x, *args, **kwargs)
            if any(arg is not None for arg in args) or any(
                value is not None for value in kwargs.values()
            ):
                raise NotImplementedError(
                    "parallel VAE attention does not accept masks or conditioning"
                )
            return operation(module, x, context)

        module.forward = forward

    def _adapt_conv(self, module: nn.Conv2d | nn.Conv3d) -> None:
        original = module._conv_forward
        radius = module.dilation[-2] * (module.kernel_size[-2] - 1) // 2
        if not radius:
            return

        def conv_forward(x, weight, bias):
            context = self.active.get()
            if context is None:
                return original(x, weight, bias)
            padding = list(module.padding)
            if hasattr(module, "_padding"):
                # Causal video convolutions already pad before _conv_forward.
                # Remove local row padding, then replace it with real neighbours.
                x = x[..., radius:-radius, :]
            else:
                padding[-2] = 0
            x = context.halo(x, radius)
            convolution = F.conv2d if isinstance(module, nn.Conv2d) else F.conv3d
            return convolution(
                x,
                weight,
                bias,
                module.stride,
                tuple(padding),
                module.dilation,
                module.groups,
            )

        module._conv_forward = conv_forward


def _adapter(vae: nn.Module) -> _ParallelDecoder:
    adapter = getattr(vae, "_chitu_parallel_decoder", None)
    if adapter is None or adapter.decoder is not vae.decoder:
        adapter = _ParallelDecoder(vae.decoder)
        vae._chitu_parallel_decoder = adapter
    else:
        adapter._validate(vae.decoder)
    return adapter


@torch.inference_mode()
def parallel_vae_decode(
    vae: nn.Module,
    latents: torch.Tensor,
    *,
    topology: VaeParallelTopology,
    decode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    enabled: bool = True,
) -> torch.Tensor | None:
    """Decode disjoint row shards, returning the complete output on the leader.

    Input latents are replicated. Unsupported decoder families fall back to
    the original decode on the leader with a warning. The configured tiling
    flag must be off: independent Diffusers tiles do not preserve full-image
    semantics, even when their internal layers are correctly distributed.
    """
    decode = decode_fn or (lambda value: vae.decode(value, return_dict=False)[0])
    vae._chitu_vae_decode_stats = {
        "vae_decode_mode": "leader",
        "parallel_vae": False,
        "vae_parallel_degree": 1,
        "vae_parallel_halo": None,
    }
    if not enabled or topology.width == 1:
        return decode(latents) if topology.is_leader else None
    if topology.process_group is None:
        raise RuntimeError("parallel VAE decode requires an active process group")
    if getattr(vae, "use_tiling", False):
        raise ValueError("disable VAE tiling before exact parallel decode")
    try:
        adapter = _adapter(vae)
    except NotImplementedError as exc:
        warnings.warn(
            f"{exc}; using whole-image VAE decode on the leader", stacklevel=2
        )
        return decode(latents) if topology.is_leader else None
    base, remainder = divmod(latents.shape[-2], topology.width)
    if base < max(1, adapter.radius):
        return decode(latents) if topology.is_leader else None
    rows = tuple(base + int(rank < remainder) for rank in range(topology.width))
    group = topology.process_group
    context = _DecodeContext(
        group=group,
        rank=topology.rank_in_lane,
        ranks=tuple(
            dist.get_global_rank(group, rank) for rank in range(topology.width)
        ),
        rows=rows,
    )
    local = latents.narrow(
        -2, sum(rows[: context.rank]), rows[context.rank]
    ).contiguous()
    token = adapter.active.set(context)
    try:
        output = decode(local)
    finally:
        adapter.active.reset(token)
    vae._chitu_vae_decode_stats.update(
        vae_decode_mode="layerwise",
        parallel_vae=True,
        vae_parallel_degree=topology.width,
    )
    return context.gather_output(output)
