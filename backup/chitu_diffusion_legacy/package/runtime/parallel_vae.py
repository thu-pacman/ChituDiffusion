"""Backward-compatible import path for parallel VAE decode."""

from chitu_diffusion.parallel.vae import parallel_decode_enabled, parallel_tiled_vae_decode

__all__ = ["parallel_decode_enabled", "parallel_tiled_vae_decode"]
