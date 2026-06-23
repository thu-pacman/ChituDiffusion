"""Generic tile-parallel VAE decode.

VAE decode is normally run on rank 0 only while every other GPU sits idle. As the
denoise stage gets cheaper (caching, parallelism, distilled few-step models), this
single-rank decode becomes a growing share of end-to-end latency. This module
splits the decode spatially across *all* world ranks and reassembles the image.

Design (model-agnostic):
  * The full (normalized, unpacked) latent is identical on every rank.
  * The latent is split along one spatial dim into `world_size` contiguous
    `core` regions. Each rank decodes its core padded with a `halo` of neighbour
    latent rows on the *interior* sides (the outer image border keeps the VAE's
    natural boundary, so border pixels stay bit-faithful to a full decode).
  * A VAE decoder is convolutional, so a halo at least as large as the decoder's
    receptive field makes every core output pixel identical to the full decode --
    no convolution seams. Each rank then keeps only its core's pixels, and we
    `all_gather` the equal-sized cores and concatenate them back into the image.

The only source of difference vs a full decode is global mid-block self-attention
(which sees a tile instead of the whole image); with a reasonable halo this is
imperceptible (measured at ~48 dB PSNR for Qwen-Image). `vae.decode` is treated
as a black box via a `decode_fn` callable, so this works for Qwen-Image, Flux,
Flux2, and 4D/5D video latents alike.
"""
from __future__ import annotations

import os
from logging import getLogger
from typing import Callable, Optional

import torch
import torch.distributed as dist

logger = getLogger(__name__)


def parallel_decode_enabled() -> bool:
    return os.getenv("CHITU_VAE_PARALLEL_DECODE", "1") == "1"


def _halo() -> int:
    try:
        return max(0, int(os.getenv("CHITU_VAE_DECODE_HALO", "8")))
    except ValueError:
        return 8


def parallel_tiled_vae_decode(
    latents: torch.Tensor,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    latent_split_dim: int,
    pixel_split_dim: int,
    scale: int,
    group=None,
    halo: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Decode `latents` by splitting the work across all ranks in `group`.

    Args:
        latents: full latent ready for `decode_fn`, identical on every rank.
        decode_fn: maps a latent tile -> pixel tensor (e.g. `vae.decode`).
        latent_split_dim: spatial dim of `latents` to split (the H axis).
        pixel_split_dim: matching spatial dim of the decoded pixel tensor.
        scale: pixel/latent ratio along the split axis (e.g. 8 for an 8x VAE).
        group: a CommGroup spanning the ranks to use (defaults to world group).
        halo: latent rows of interior overlap per side (defaults to env / 8).

    Returns:
        Full decoded pixel tensor on every rank when parallel decode runs; when
        disabled or not applicable, the full image on rank 0 and None elsewhere
        (preserving the legacy rank-0-only contract).
    """
    if group is None:
        from chitu_diffusion.core.distributed.parallel_state import get_world_group

        group = get_world_group()
    world = group.group_size
    rank_in_group = group.rank_in_group
    halo = _halo() if halo is None else halo

    global_rank = dist.get_rank() if dist.is_initialized() else 0

    h_latent = latents.shape[latent_split_dim]
    # Fall back to legacy rank-0-only decode when parallelism can't help or is off.
    if not parallel_decode_enabled() or world <= 1 or h_latent < world:
        if global_rank != 0:
            return None
        return decode_fn(latents)

    base = (h_latent + world - 1) // world

    core_start = rank_in_group * base
    core_end = min(core_start + base, h_latent)
    # Decode this core plus an interior halo; the outer image border is left to the
    # VAE's own boundary handling, so border pixels match a full decode.
    lo = max(0, core_start - halo)
    hi = min(h_latent, core_end + halo)

    tile = latents.narrow(latent_split_dim, lo, hi - lo).contiguous()
    pixel_tile = decode_fn(tile)

    # Slice out just this core's pixels from the decoded tile.
    core_off_px = (core_start - lo) * scale
    core_len_px = (core_end - core_start) * scale
    core_px = pixel_tile.narrow(pixel_split_dim, core_off_px, core_len_px)

    # all_gather needs equal shapes: pad every core up to `base * scale` rows.
    full_core_px = base * scale
    if core_len_px < full_core_px:
        pad_shape = list(core_px.shape)
        pad_shape[pixel_split_dim] = full_core_px - core_len_px
        pad = core_px.new_zeros(pad_shape)
        core_px = torch.cat([core_px, pad], dim=pixel_split_dim)
    core_px = core_px.contiguous()

    gathered = [torch.empty_like(core_px) for _ in range(world)]
    dist.all_gather(gathered, core_px, group=group.gpu_group)

    # Crop each gathered core back to its true (unpadded) length and concatenate.
    pieces = []
    for rank in range(world):
        rank_len = (min((rank + 1) * base, h_latent) - rank * base) * scale
        if rank_len <= 0:
            continue
        pieces.append(gathered[rank].narrow(pixel_split_dim, 0, rank_len))
    return torch.cat(pieces, dim=pixel_split_dim).contiguous()
