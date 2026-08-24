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


AUDIO_LATENT_CHANNELS = 32
AUDIO_OUTPUT_CHANNELS = 2
AUDIO_SAMPLE_RATE = 32000


def _read_release_config(component_dir: Path) -> dict[str, Any]:
    config_path = component_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MiniMax H3 audio VAE config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("MiniMax H3 audio VAE config.json must contain an object")
    return config


def _validate_config(
    config: dict[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    expected = {
        "latent_channels": AUDIO_LATENT_CHANNELS,
        "output_channel": AUDIO_OUTPUT_CHANNELS,
        "sample_rate": AUDIO_SAMPLE_RATE,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(
                f"MiniMax H3 audio VAE {name} must be {value}, "
                f"got {config.get(name)!r}"
            )

    values: list[tuple[float, ...]] = []
    for name in ("latents_mean", "latents_std"):
        raw = config.get(name)
        if (
            not isinstance(raw, list)
            or len(raw) != AUDIO_LATENT_CHANNELS
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw)
        ):
            raise ValueError(
                f"MiniMax H3 audio VAE {name} must contain exactly "
                f"{AUDIO_LATENT_CHANNELS} numbers"
            )
        parsed = tuple(float(value) for value in raw)
        if not all(math.isfinite(value) for value in parsed):
            raise ValueError(f"MiniMax H3 audio VAE {name} must be finite")
        if name == "latents_std" and not all(value > 0 for value in parsed):
            raise ValueError("MiniMax H3 audio VAE latents_std must be positive")
        values.append(parsed)
    return values[0], values[1]


def _load_release_class(component_dir: Path, config: dict[str, Any]) -> type:
    """Load the local release's licensed entry point only when requested."""
    auto_map = config.get("auto_map")
    entry = auto_map.get("AutoModel") if isinstance(auto_map, dict) else None
    if entry != "minimax_h3_audio_vae.MiniMaxH3AudioVAE":
        raise ValueError(
            "unsupported MiniMax H3 audio VAE auto_map entry; expected the "
            "published minimax_h3_audio_vae.MiniMaxH3AudioVAE"
        )
    module_name, class_name = entry.rsplit(".", 1)
    module_path = component_dir / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"MiniMax H3 audio VAE entry module not found: {module_path}")

    digest = hashlib.sha256(str(component_dir.resolve()).encode()).hexdigest()[:16]
    package_name = f"_chitu_minimax_h3_audio_vae_{digest}"
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
            raise ImportError(f"cannot load MiniMax H3 audio VAE module: {module_path}")
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
        raise TypeError("published MiniMax H3 audio VAE entry class is invalid")
    return release_class


class MiniMaxH3AudioVAE(nn.Module):
    """Strict decode-only wrapper around the local H3 stereo audio VAE."""

    sample_rate = AUDIO_SAMPLE_RATE

    def __init__(
        self,
        model: nn.Module,
        latents_mean: Sequence[float],
        latents_std: Sequence[float],
    ) -> None:
        super().__init__()
        if not callable(getattr(model, "decode", None)):
            raise TypeError("published audio VAE is missing decode")
        model_rate = getattr(model, "sample_rate", AUDIO_SAMPLE_RATE)
        if int(model_rate) != AUDIO_SAMPLE_RATE:
            raise ValueError(
                f"published audio VAE sample_rate must be {AUDIO_SAMPLE_RATE}, "
                f"got {model_rate!r}"
            )
        stats_config = {
            "latent_channels": AUDIO_LATENT_CHANNELS,
            "output_channel": AUDIO_OUTPUT_CHANNELS,
            "sample_rate": AUDIO_SAMPLE_RATE,
            "latents_mean": list(latents_mean),
            "latents_std": list(latents_std),
        }
        mean, std = _validate_config(stats_config)
        self.model = model.to(dtype=torch.float32).eval()
        self.register_buffer("latents_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("latents_std", torch.tensor(std, dtype=torch.float32))

    @classmethod
    def from_pretrained(
        cls,
        component_dir: str | Path,
        *,
        device: torch.device | str = "cuda",
    ) -> "MiniMaxH3AudioVAE":
        root = Path(component_dir)
        config = _read_release_config(root)
        mean, std = _validate_config(config)
        release_class = _load_release_class(root, config)
        model = release_class.from_pretrained(str(root))
        if not isinstance(model, nn.Module):
            raise TypeError("published MiniMax H3 audio VAE did not return nn.Module")
        model.to(device=device, dtype=torch.float32).eval()
        return cls(model, mean, std)

    def _device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.model.buffers(), None)
        return buffer.device if buffer is not None else self.latents_mean.device

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if not isinstance(latents, torch.Tensor) or latents.ndim != 3:
            raise ValueError("audio latents must be a tensor shaped [2, 32, T]")
        if tuple(latents.shape[:2]) != (
            AUDIO_OUTPUT_CHANNELS,
            AUDIO_LATENT_CHANNELS,
        ):
            raise ValueError(
                "audio latents must be shaped [2, 32, T], got "
                f"{tuple(latents.shape)}"
            )

        device = self._device()
        normalized = latents.to(device=device, dtype=torch.float32)
        mean = self.latents_mean.to(device=device).view(1, -1, 1)
        std = self.latents_std.to(device=device).view(1, -1, 1)
        decode_latents = normalized * std + mean
        use_autocast = device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_autocast,
        ):
            waveform = self.model.decode(decode_latents)
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("audio VAE decode must return a tensor")
        if (
            waveform.ndim != 3
            or waveform.shape[0] != AUDIO_OUTPUT_CHANNELS
            or waveform.shape[1] != 1
        ):
            raise ValueError(
                "native audio VAE output must be [2, 1, L], got "
                f"{tuple(waveform.shape)}"
            )
        return waveform.permute(1, 0, 2).float().contiguous()


def load_audio_vae(
    component_dir: str | Path,
    *,
    device: torch.device | str = "cuda",
) -> MiniMaxH3AudioVAE:
    return MiniMaxH3AudioVAE.from_pretrained(component_dir, device=device)


__all__ = [
    "AUDIO_LATENT_CHANNELS",
    "AUDIO_OUTPUT_CHANNELS",
    "AUDIO_SAMPLE_RATE",
    "MiniMaxH3AudioVAE",
    "load_audio_vae",
]
