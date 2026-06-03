from __future__ import annotations

import math
import inspect
from typing import Optional

import numpy as np
import torch
from torch import Tensor


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    timesteps: Optional[list[int]] = None,
    sigmas: Optional[list[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of timesteps or sigmas can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"{scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(f"{scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def flowmatch_sigmas(num_inference_steps: int) -> np.ndarray:
    return np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)


def calculate_flux1_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def compute_flux2_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    if image_seq_len >= 4096:
        return 0.75
    a1, b1 = 7.429875840156292e-05, 0.5252699038513061
    a2, b2 = 0.0001735774813837763, 0.04077856568717515
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def prepare_flux1_latent_image_ids(batch_size: int, height: int, width: int, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    latent_image_ids = latent_image_ids.reshape((height // 2) * (width // 2), 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def pack_flux1_latents(latents, batch_size: int, num_channels_latents: int, height: int, width: int):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


def unpack_flux1_latents(latents, height: int, width: int, vae_scale_factor: int):
    batch_size, _, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, height * 2, width * 2)


def prepare_flux1_latents(
    *,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    vae_scale_factor: int,
    dtype: torch.dtype,
    device,
    generator: torch.Generator,
):
    latent_height = 2 * (int(height) // vae_scale_factor)
    latent_width = 2 * (int(width) // vae_scale_factor)
    shape = (batch_size, num_channels_latents, latent_height, latent_width)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    latents = pack_flux1_latents(latents, batch_size, num_channels_latents, latent_height, latent_width)
    latent_image_ids = prepare_flux1_latent_image_ids(batch_size, latent_height, latent_width, device, dtype)
    return latents, latent_image_ids


def prepare_flux2_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = latents.shape
    coords = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1))
    return coords.unsqueeze(0).expand(batch_size, -1, -1)


def pack_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)


def unpack_flux2_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    x_list = []
    for data, pos in zip(x, x_ids):
        _, channels = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        height = torch.max(h_ids) + 1
        width = torch.max(w_ids) + 1
        flat_ids = h_ids * width + w_ids
        out = torch.zeros((height * width, channels), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, channels), data)
        x_list.append(out.view(height, width, channels).permute(2, 0, 1))
    return torch.stack(x_list, dim=0)


def unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)


def prepare_flux2_latents(
    *,
    batch_size: int,
    num_latents_channels: int,
    height: int,
    width: int,
    vae_scale_factor: int,
    dtype: torch.dtype,
    device,
    generator: torch.Generator,
):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    latent_ids = prepare_flux2_latent_ids(latents).to(device)
    return pack_flux2_latents(latents), latent_ids


def generalized_time_snr_shift(t: Tensor, mu: float, sigma: float) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_flux2_shifted_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_flux2_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()
