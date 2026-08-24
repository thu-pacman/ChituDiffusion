from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file


_REQUIRED_TENSORS = ("prompt_embeds", "video_latents", "audio_latents")


def _shape_metadata(metadata: Mapping[str, str], name: str) -> tuple[int, ...] | None:
    value = metadata.get(f"{name}_shape")
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise ValueError(f"{name}_shape metadata must be a sequence")
    shape = tuple(int(dim) for dim in parsed)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"{name}_shape metadata must contain positive dimensions")
    return shape


@dataclass(frozen=True, slots=True)
class MiniMaxH3Conditioning:
    """Offline inputs for the DiT-only MiniMax-H3 latent backend."""

    prompt_embeds: torch.Tensor
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    metadata: Mapping[str, str]

    def to(
        self,
        device: torch.device | str,
        *,
        prompt_dtype: torch.dtype = torch.bfloat16,
        latent_dtype: torch.dtype = torch.float32,
    ) -> "MiniMaxH3Conditioning":
        return MiniMaxH3Conditioning(
            prompt_embeds=self.prompt_embeds.to(device=device, dtype=prompt_dtype),
            video_latents=self.video_latents.to(device=device, dtype=latent_dtype),
            audio_latents=self.audio_latents.to(device=device, dtype=latent_dtype),
            metadata=dict(self.metadata),
        )


def load_conditioning_package(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> MiniMaxH3Conditioning:
    """Load a local safetensors package without invoking any online encoder."""

    package = Path(path)
    if not package.is_file():
        raise FileNotFoundError(f"conditioning package does not exist: {package}")
    with safe_open(package, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = dict(handle.metadata() or {})
    missing = sorted(set(_REQUIRED_TENSORS) - keys)
    if missing:
        raise ValueError(
            "conditioning package is missing tensors: " + ", ".join(missing)
        )
    tensors = load_file(package, device=str(device))
    for name in _REQUIRED_TENSORS:
        expected = _shape_metadata(metadata, name)
        if expected is not None and tuple(tensors[name].shape) != expected:
            raise ValueError(
                f"{name} shape {tuple(tensors[name].shape)} does not match "
                f"metadata {expected}"
            )
    return MiniMaxH3Conditioning(
        prompt_embeds=tensors["prompt_embeds"],
        video_latents=tensors["video_latents"],
        audio_latents=tensors["audio_latents"],
        metadata=metadata,
    )


def deterministic_sin_tensor(
    shape: Sequence[int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    offset: int = 0,
) -> torch.Tensor:
    """Build the same ``sin(arange * .001)`` signal used by the H3 smoke test."""

    dimensions = tuple(int(dim) for dim in shape)
    if not dimensions or any(dim <= 0 for dim in dimensions):
        raise ValueError("synthetic tensor shape must contain positive dimensions")
    count = 1
    for dim in dimensions:
        count *= dim
    values = torch.arange(
        offset,
        offset + count,
        device=device,
        dtype=torch.float32,
    )
    return torch.sin(values.reshape(dimensions) * 0.001).to(dtype)


def make_synthetic_conditioning(
    *,
    text_length: int,
    text_dim: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    latent_channels: int,
    audio_t: int,
    audio_channels: int,
    audio_latent_dim: int,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> MiniMaxH3Conditioning:
    """Create deterministic, encoder-free conditioning for smoke and warmup."""

    prompt_shape = (text_length, text_dim)
    video_shape = (latent_channels, latent_t, latent_h, latent_w)
    audio_shape = (audio_channels, audio_t, audio_latent_dim)
    prompt_count = math.prod(prompt_shape)
    video_count = math.prod(video_shape)
    base = int(seed) * 1_000_003
    prompt = deterministic_sin_tensor(
        prompt_shape, device=device, dtype=torch.bfloat16, offset=base
    )
    video = deterministic_sin_tensor(
        video_shape, device=device, offset=base + prompt_count
    )
    audio = deterministic_sin_tensor(
        audio_shape,
        device=device,
        offset=base + prompt_count + video_count,
    )
    metadata = {
        "text_length": str(text_length),
        "latent_t": str(latent_t),
        "latent_h": str(latent_h),
        "latent_w": str(latent_w),
        "audio_t": str(audio_t),
        "audio_channels": str(audio_channels),
        "prompt_embeds_shape": json.dumps(list(prompt_shape)),
        "video_latents_shape": json.dumps(list(video_shape)),
        "audio_latents_shape": json.dumps(list(audio_shape)),
        "generator": "sin(arange * 0.001)",
        "seed": str(seed),
    }
    return MiniMaxH3Conditioning(prompt, video, audio, metadata)
