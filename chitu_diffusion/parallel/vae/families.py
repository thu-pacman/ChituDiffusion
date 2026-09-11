"""Decoder-specific layouts on top of shared spatial collectives.

Released model code supplies packing, rotary embedding, and residual blocks.
Adapters only change operations whose semantics cross a spatial partition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DecoderSpec:
    name: str
    forwards: tuple[tuple[type, Callable], ...] = ()


def decoder_spec(decoder: nn.Module) -> DecoderSpec:
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        QwenImageDecoder3d,
    )
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanDecoder3d
    from diffusers.models.autoencoders.vae import Decoder

    if type(decoder) in (Decoder, WanDecoder3d, QwenImageDecoder3d):
        return DecoderSpec(type(decoder).__name__)
    source = import_module(type(decoder).__module__)
    if (
        source.__name__.split(".")[-1] == "autoencoder_kl_3d"
        and type(decoder) is getattr(source, "Decoder", None)
        and isinstance(decoder.mid.attn_1, source.AttnBlock)
    ):
        return DecoderSpec(
            "hunyuan_conv3d",
            (
                (source.AttnBlock, _hunyuan_attention),
                (source.Conv3d, _hunyuan_convolution),
            ),
        )
    if source.__name__.split(".")[-1] == "vae_vit" and type(decoder) is getattr(
        source, "ViT3DDecoder", None
    ):
        attention = import_module(source.__package__ + ".attention")
        if decoder.spatial_parallel or any(
            type(block.attn) is not attention.Attention or block.attn.spatial_parallel
            for block in decoder.transformer_blocks
        ):
            raise NotImplementedError(
                "disable release VAE sequence parallelism before Chitu VAEP"
            )
        return DecoderSpec(
            "minimax_vit3d",
            (
                (type(decoder), _minimax_decoder),
                (attention.Attention, _minimax_attention),
            ),
        )
    raise NotImplementedError(f"no exact spatial adapter for {type(decoder).__name__}")


def _hunyuan_attention(module, x, context):
    hidden = module.norm(x)
    batch, channels, frames, height, width = x.shape
    query, key, value = (
        projection(hidden).flatten(2).transpose(1, 2).unsqueeze(1)
        for projection in (module.q, module.k, module.v)
    )
    key, value = context.gather_kv(key, value)
    hidden = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    hidden = (
        hidden.squeeze(1)
        .transpose(1, 2)
        .reshape(batch, channels, frames, height, width)
    )
    return x + module.proj_out(hidden)


def _hunyuan_convolution(module, x, context):
    # The release chunks time based on local tensor size. Uneven row shards
    # could choose different chunk counts and therefore different collectives.
    # _conv_forward still supplies the common spatial halo implementation.
    return nn.Conv3d.forward(module, x)


def _minimax_decoder(module, x, context):
    source = import_module(type(module).__module__)
    batch, _, frames, height, width = x.shape
    rows = context.lengths(height)
    start = sum(rows[: context.rank])
    module.loss_info = {}
    hidden = source._pack_tensors_3d(x, 1, 1)
    with torch.autocast("cuda", enabled=False):
        hidden = source._linear_with_module_dtype(
            module.x_embedder, hidden, hidden.dtype
        )
    patches = hidden.shape[1]
    suffix = module.num_register_tokens + 1
    pieces = [hidden]
    if module.register_tokens is not None:
        pieces.append(module.register_tokens.expand(batch, -1, -1))
    pieces.append(torch.zeros_like(hidden[:, :1]))
    hidden = torch.cat(pieces, dim=1)

    # Use the released coordinate calculation at the global height before slicing.
    # This also preserves its BF16 coordinate rounding, which a local linspace changes.
    coords = torch.arange(0.5, sum(rows), device=x.device, dtype=x.dtype)
    coords = 2.0 * (coords / sum(rows)) - 1.0
    ids = source.create_token_ids(
        (frames, coords[start : start + height], width), x.device, x.dtype
    )
    ids = torch.cat((ids.expand(batch, -1, -1), ids.new_zeros(batch, suffix, 3)), dim=1)
    rotary = module.pos_embed(ids)

    lengths = tuple(
        frames * length * width + (suffix if i == 0 else 0)
        for i, length in enumerate(rows)
    )
    mask = None
    if module.t_causal:
        times = torch.arange(frames, device=x.device)
        query_times = torch.cat(
            (times.repeat_interleave(height * width), times.new_full((suffix,), -1))
        )
        key_times = []
        for i, length in enumerate(rows):
            key_times.append(times.repeat_interleave(length * width))
            if i == 0:
                key_times.append(times.new_full((suffix,), -1))
        key_times = torch.cat(key_times)
        mask = (query_times[:, None] >= key_times[None, :]) | (
            query_times[:, None] == -1
        )
    pack_info = {"vae_lengths": lengths, "vae_patches": patches, "vae_mask": mask}
    for block in module.transformer_blocks:
        hidden = block(hidden, rotary, pack_info)
    hidden = module.norm_out(hidden)
    with torch.autocast("cuda", enabled=False):
        output = source._linear_with_module_dtype(
            module.proj_out, hidden[:, :patches], hidden.dtype
        )
    spatial = module.config.patch_size
    temporal = module.config.patch_size_t
    return source._unpack_tensors_3d(
        output, spatial, temporal, frames * temporal, height * spatial, width * spatial
    )


def _minimax_attention(module, hidden, context, rotary_pos_emb=None, pack_info=None):
    source = import_module(type(module).__module__)
    batch, length, _ = hidden.shape
    query, key, value = (
        module.to_qkv(hidden)
        .reshape(batch, length, module.heads, 3 * module.dim_head)
        .chunk(3, dim=-1)
    )
    if module.norm_q is not None:
        query = module.norm_q(source._vit_norm_input(module.norm_q, query)).to(
            query.dtype
        )
    if module.norm_k is not None:
        key = module.norm_k(source._vit_norm_input(module.norm_k, key)).to(key.dtype)
    if rotary_pos_emb is not None:
        query = source.apply_rotary_pos_emb(query, rotary_pos_emb)
        key = source.apply_rotary_pos_emb(key, rotary_pos_emb)
    # Suffix queries are replicated, but their K/V contribution has one owner.
    if context.rank != 0:
        key, value = (
            key[:, : pack_info["vae_patches"]],
            value[:, : pack_info["vae_patches"]],
        )
    key, value = context.gather_kv(
        key.transpose(1, 2), value.transpose(1, 2), lengths=pack_info["vae_lengths"]
    )
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key,
        value,
        attn_mask=pack_info["vae_mask"],
        dropout_p=0.0,
    )
    return module.to_out(
        output.transpose(1, 2).nan_to_num(0.0).reshape(batch, length, -1)
    )
