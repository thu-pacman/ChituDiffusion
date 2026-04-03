import itertools
import os
from pathlib import Path
from typing import Optional, Set

import torch


def _parse_layers(raw: Optional[str]) -> Optional[Set[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    layers = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        layers.add(int(part))
    return layers or None


def _resolve_dtype(raw: Optional[str]) -> Optional[torch.dtype]:
    if raw is None:
        return torch.float32
    value = raw.strip().lower()
    if value in {"", "none", "keep"}:
        return None
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported CHITU_WAN_KV_CAPTURE_DTYPE: {raw}")
    return mapping[value]


class WanKVCaptureWriter:
    def __init__(
        self,
        *,
        layers: Optional[Set[int]] = None,
        layer_interval: Optional[int] = None,
        tensor_dtype: Optional[torch.dtype] = torch.float32,
        output_dir: Optional[str] = "./outputs/kv_cache",
        output_dir_latents: Optional[str] = "./outputs/latents_cache",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir_latents = Path(output_dir_latents)
        self.output_dir_latents.mkdir(parents=True, exist_ok=True)

        self.layers = layers
        self.layer_interval = layer_interval
        self.tensor_dtype = tensor_dtype
        self._counter = itertools.count()

    @classmethod
    def from_env(cls) -> Optional["WanKVCaptureWriter"]:
        output_dir = os.getenv("CHITU_WAN_KV_CAPTURE_DIR")
        if not output_dir:
            return None
        return cls(
            output_dir=output_dir,
            layers=_parse_layers(os.getenv("CHITU_WAN_KV_CAPTURE_LAYERS")),
            layer_interval=int(os.getenv("CHITU_WAN_KV_CAPTURE_LAYER_INTERVAL", "0") or 0) or None,
            tensor_dtype=_resolve_dtype(os.getenv("CHITU_WAN_KV_CAPTURE_DTYPE")),
        )

    def should_capture(self, layer_idx: Optional[int]) -> bool:
        if layer_idx is None:
            return False
        if self.layers is None:
            if self.layer_interval is None:
                return True
            return layer_idx % self.layer_interval == 0
        return layer_idx in self.layers

    def _to_host(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach()
        if self.tensor_dtype is not None:
            tensor = tensor.to(dtype=self.tensor_dtype)
        return tensor.to(device="cpu", copy=True).contiguous()

    def capture(
        self,
        *,
        k: torch.Tensor,
        v: torch.Tensor,
        noise_step: Optional[int],
        layer_idx: int,
        is_pos: Optional[bool],
        patch_idx: Optional[int],
        pp_rank: Optional[int],
    ) -> Path:
        capture_idx = next(self._counter)
        branch = "unknown" if is_pos is None else ("pos" if is_pos else "neg")
        rank = "na" if pp_rank is None else str(pp_rank)
        step = "na" if noise_step is None else f"{int(noise_step):04d}"
        patch = "na" if patch_idx is None else f"{int(patch_idx):03d}"
        path = self.output_dir / (
            f"step{step}_{branch}_rank{rank}_layer{layer_idx:03d}_patch{patch}_{capture_idx:06d}.pt"
        )
        torch.save(
            {
                "meta": {
                    "layer_idx": int(layer_idx),
                    "branch": branch,
                    "patch_idx": patch_idx,
                    "pp_rank": pp_rank,
                },
                "k": self._to_host(k),
                "v": self._to_host(v),
            },
            path,
        )
        return path

    def capture_latents(
        self,
        *,
        latents: torch.Tensor,
        noise_step: Optional[int],
        is_pos: Optional[bool],
        patch_idx: Optional[int],
        pp_rank: Optional[int],
    ) -> Path:
        capture_idx = next(self._counter)
        branch = "unknown" if is_pos is None else ("pos" if is_pos else "neg")
        rank = "na" if pp_rank is None else str(pp_rank)
        step = "na" if noise_step is None else f"{int(noise_step):04d}"
        patch = "na" if patch_idx is None else f"{int(patch_idx):03d}"
        path = self.output_dir_latents / (
            f"step{step}_{branch}_rank{rank}_latents_patch{patch}_{capture_idx:06d}.pt"
        )
        torch.save(
            {
                "meta": {
                    "layer_idx": None,
                    "branch": branch,
                    "patch_idx": patch_idx,
                    "pp_rank": pp_rank,
                },
                "latents": self._to_host(latents),
            },
            path,
        )
        return path
