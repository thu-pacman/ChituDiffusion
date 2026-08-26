from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MiniMaxH3DiTConfig:
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 96768
    final_adaln_out_features: int = 10752
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def video_patch_dim(self) -> int:
        result = self.latents_dim
        for size in self.patch_size:
            result *= size
        return result

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "MiniMaxH3DiTConfig":
        root = Path(path)
        config_path = root / "config.json" if root.is_dir() else root
        payload = json.loads(config_path.read_text())
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in payload.items() if key in fields}
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        return cls(**values)

    def validate_parallel(self, tp_degree: int, ulysses_degree: int = 1) -> None:
        if tp_degree < 1 or ulysses_degree < 1:
            raise ValueError("TP and Ulysses degrees must be positive")
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("num_attention_heads", self.num_attention_heads),
            ("ffn_hidden_size", self.ffn_hidden_size),
            ("time_embed_hidden_size", self.time_embed_hidden_size),
            ("adaln_out_features", self.adaln_out_features),
            ("final_adaln_out_features", self.final_adaln_out_features),
            ("video_patch_dim", self.video_patch_dim),
            ("audio_latents_dim", self.audio_latents_dim),
        ):
            if value % tp_degree:
                raise ValueError(f"{name}={value} is not divisible by TP={tp_degree}")
        if self.num_attention_heads % (tp_degree * ulysses_degree):
            raise ValueError(
                f"heads={self.num_attention_heads} is not divisible by "
                f"TP x Ulysses={tp_degree * ulysses_degree}"
            )
        if 6 * self.rope_inv_freq_len > self.attention_head_dim:
            raise ValueError("3D RoPE dimensions exceed attention head dimension")
        if 64 % ulysses_degree:
            raise ValueError("Ulysses degree must divide packed alignment 64")

