from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from chitu_diffusion.parallel.vae import (
    VaeParallelTopology,
    parallel_spatial_vae_decode,
)

VIDEO_LATENT_CHANNELS = 24


def _read_release_config(component_dir: Path) -> dict[str, Any]:
    config_path = component_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MiniMax H3 video VAE config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("MiniMax H3 video VAE config.json must contain an object")
    return config


def _validate_stats(
    config: dict[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if config.get("latent_channels") != VIDEO_LATENT_CHANNELS:
        raise ValueError(
            "MiniMax H3 video VAE latent_channels must be "
            f"{VIDEO_LATENT_CHANNELS}, got {config.get('latent_channels')!r}"
        )
    values: list[tuple[float, ...]] = []
    for name in ("latents_mean", "latents_std"):
        raw = config.get(name)
        if (
            not isinstance(raw, list)
            or len(raw) != VIDEO_LATENT_CHANNELS
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw)
        ):
            raise ValueError(
                f"MiniMax H3 video VAE {name} must contain exactly "
                f"{VIDEO_LATENT_CHANNELS} numbers"
            )
        parsed = tuple(float(value) for value in raw)
        if not all(math.isfinite(value) for value in parsed):
            raise ValueError(f"MiniMax H3 video VAE {name} must be finite")
        if name == "latents_std" and not all(value > 0 for value in parsed):
            raise ValueError("MiniMax H3 video VAE latents_std must be positive")
        values.append(parsed)
    return values[0], values[1]


def _load_release_class(component_dir: Path, config: dict[str, Any]) -> type:
    """Load the locally published entry point without vendoring its implementation."""
    auto_map = config.get("auto_map")
    entry = auto_map.get("AutoModel") if isinstance(auto_map, dict) else None
    if entry != "minimax_h3_video_vae.MiniMaxH3VideoVAE":
        raise ValueError(
            "unsupported MiniMax H3 video VAE auto_map entry; expected the "
            "published minimax_h3_video_vae.MiniMaxH3VideoVAE"
        )
    module_name, class_name = entry.rsplit(".", 1)
    module_path = component_dir / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"MiniMax H3 video VAE entry module not found: {module_path}")

    digest = hashlib.sha256(str(component_dir.resolve()).encode()).hexdigest()[:16]
    package_name = f"_chitu_minimax_h3_video_vae_{digest}"
    qualified_name = f"{package_name}.{module_name}"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(component_dir)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package
    module = sys.modules.get(qualified_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(qualified_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load MiniMax H3 video VAE module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(qualified_name, None)
            raise
    release_class = getattr(module, class_name, None)
    if not isinstance(release_class, type) or not callable(
        getattr(release_class, "from_pretrained", None)
    ):
        raise TypeError("published MiniMax H3 video VAE entry class is invalid")
    return release_class


class MiniMaxH3VideoVAE(nn.Module):
    """Strict decode-only wrapper around the locally released H3 video VAE."""

    def __init__(
        self,
        model: nn.Module,
        latents_mean: Sequence[float],
        latents_std: Sequence[float],
        *,
        require_tiled_decoder: bool = False,
    ) -> None:
        super().__init__()
        if not callable(getattr(model, "decode_base", None)):
            raise TypeError("published video VAE is missing decode_base")
        processor = getattr(model, "processor", None)
        if processor is None or not callable(getattr(processor, "revert_tensor", None)):
            raise TypeError("published video VAE is missing processor.revert_tensor")
        if require_tiled_decoder and not callable(getattr(model, "tiled_decode", None)):
            raise TypeError("published video VAE is missing tiled_decode")
        if require_tiled_decoder and not hasattr(model, "decoder_tiling"):
            raise TypeError("published video VAE is missing decoder_tiling")

        stats_config = {
            "latent_channels": VIDEO_LATENT_CHANNELS,
            "latents_mean": list(latents_mean),
            "latents_std": list(latents_std),
        }
        mean, std = _validate_stats(stats_config)
        self.model = model.to(dtype=torch.float32).eval()
        self.require_tiled_decoder = require_tiled_decoder
        self.register_buffer("latents_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("latents_std", torch.tensor(std, dtype=torch.float32))

    @classmethod
    def from_pretrained(
        cls,
        component_dir: str | Path,
        *,
        device: torch.device | str = "cuda",
    ) -> "MiniMaxH3VideoVAE":
        """Lazily import and instantiate the self-contained local release."""
        root = Path(component_dir)
        config = _read_release_config(root)
        mean, std = _validate_stats(config)
        if config.get("vae_decoder_tiling") != 1:
            raise ValueError("published MiniMax H3 video VAE must enable decoder tiling")
        release_class = _load_release_class(root, config)
        model = release_class.from_pretrained(str(root))
        if not isinstance(model, nn.Module):
            raise TypeError("published MiniMax H3 video VAE did not return nn.Module")
        model.to(device=device, dtype=torch.float32).eval()
        return cls(model, mean, std, require_tiled_decoder=True)

    def _device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.model.buffers(), None)
        return buffer.device if buffer is not None else self.latents_mean.device

    def _prepare_cuda_autocast(self) -> None:
        prepare = getattr(self.model, "prepare_decoder_autocast_weights", None)
        if callable(prepare):
            prepare(torch.float16)
            return
        decoder = getattr(self.model, "decoder", None)
        prepare = getattr(decoder, "prepare_autocast_linear_weights", None)
        if callable(prepare):
            prepare(torch.float16)

    @torch.no_grad()
    def encode(self, image: Any, *, seed: int = 42) -> torch.Tensor:
        """Encode one keyframe using H3's deterministic sampled contract."""

        if not callable(getattr(self.model, "encode_images", None)):
            raise TypeError("published video VAE is missing encode_images")
        device = self._device()
        cuda_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed(int(seed))
            encoded = self.model.encode_images(image, use_fp16_latent=True)
        if not isinstance(encoded, list) or len(encoded) != 1:
            raise TypeError("video VAE encode_images must return one latent tensor")
        latent = encoded[0].to(device=device, dtype=torch.float32)
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        elif latent.ndim == 3:
            latent = latent.unsqueeze(0).unsqueeze(2)
        if latent.ndim != 5 or latent.shape[1] != VIDEO_LATENT_CHANNELS:
            raise ValueError(
                "encoded keyframe must be [1,24,1,H,W], got "
                f"{tuple(latent.shape)}"
            )
        mean = self.latents_mean.to(device=device).view(1, -1, 1, 1, 1)
        std = self.latents_std.to(device=device).view(1, -1, 1, 1, 1)
        return ((latent - mean) / std).contiguous()

    def _prepare_decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
            raise ValueError("video latents must be a tensor shaped [1, 24, T, H, W]")
        if tuple(latents.shape[:2]) != (1, VIDEO_LATENT_CHANNELS):
            raise ValueError(
                "video latents must be shaped [1, 24, T, H, W], got "
                f"{tuple(latents.shape)}"
            )

        device = self._device()
        normalized = latents.to(device=device, dtype=torch.float32)
        mean = self.latents_mean.to(device=device).view(1, -1, 1, 1, 1)
        std = self.latents_std.to(device=device).view(1, -1, 1, 1, 1)
        return (normalized * std + mean).contiguous()

    def _decode_raw(
        self,
        decode_latents: torch.Tensor,
        *,
        spatial_tiling: bool,
    ) -> torch.Tensor:
        device = self._device()
        backend = getattr(self.model, "model", self.model)
        previous_tiling = getattr(backend, "decoder_tiling", None)
        short_clip = decode_latents.shape[2] < 7
        if self.require_tiled_decoder:
            backend.decoder_tiling = spatial_tiling and not short_clip
        use_autocast = device.type == "cuda"
        if use_autocast:
            self._prepare_cuda_autocast()
        try:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_autocast,
            ):
                # decode_temporal requires at least one complete 17-frame
                # release clip. The only shorter FL2VA geometry is five
                # frames; decode its two latent frames directly and trim.
                direct_decode = getattr(backend, "decode", None)
                if short_clip and callable(direct_decode):
                    decoded = direct_decode(decode_latents)
                    trim_output = getattr(backend, "trim_output", None)
                    if callable(trim_output):
                        decoded = trim_output(decoded, 5)
                else:
                    decoded = self.model.decode_base(decode_latents)
        finally:
            if previous_tiling is not None:
                backend.decoder_tiling = previous_tiling
        if not isinstance(decoded, torch.Tensor):
            raise TypeError("video VAE decode pipeline must return a tensor")
        return decoded

    def _finish_decode(self, decoded: torch.Tensor) -> torch.Tensor:
        decoded = self.model.processor.revert_tensor(decoded)
        if not isinstance(decoded, torch.Tensor):
            raise TypeError("video VAE decode pipeline must return a tensor")
        if decoded.ndim == 4:
            if decoded.shape[0] < 1 or decoded.shape[1] != 3:
                raise ValueError(f"unsupported decoded video shape {tuple(decoded.shape)}")
            decoded = decoded.unsqueeze(0).transpose(1, 2)
        if decoded.ndim != 5 or decoded.shape[0] != 1 or decoded.shape[1] != 3:
            raise ValueError(
                "decoded video must be [1, 3, F, H, W], got "
                f"{tuple(decoded.shape)}"
            )
        return decoded.float().clamp_(0, 1).contiguous()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        decode_latents = self._prepare_decode_latents(latents)
        return self._finish_decode(
            self._decode_raw(decode_latents, spatial_tiling=True)
        )

    @torch.no_grad()
    def decode_parallel(
        self,
        latents: torch.Tensor,
        *,
        topology: VaeParallelTopology,
        enabled: bool = True,
    ) -> torch.Tensor | None:
        """Decode H3 release tiles across a lane and return only on its leader."""

        if not enabled or latents.shape[2] < 7:
            return self.decode(latents) if topology.is_leader else None
        decode_latents = self._prepare_decode_latents(latents)
        backend = getattr(self.model, "model", self.model)
        decoded = parallel_spatial_vae_decode(
            decode_latents,
            lambda tile: self._decode_raw(tile, spatial_tiling=False),
            topology=topology,
            scale=int(getattr(backend, "vae_ratio", 16)),
            tile_size=int(getattr(backend, "decoder_tile_size", 256)),
            overlap_min=int(getattr(backend, "decoder_tile_overlap_min", 64)),
        )
        return None if decoded is None else self._finish_decode(decoded)


def load_video_vae(
    component_dir: str | Path,
    *,
    device: torch.device | str = "cuda",
) -> MiniMaxH3VideoVAE:
    return MiniMaxH3VideoVAE.from_pretrained(component_dir, device=device)


__all__ = ["MiniMaxH3VideoVAE", "VIDEO_LATENT_CHANNELS", "load_video_vae"]
