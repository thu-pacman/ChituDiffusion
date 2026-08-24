from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MINIMAX_H3_QWEN3VL_SELECTED_LAYERS = 50


@dataclass(frozen=True, slots=True)
class MiniMaxH3Qwen3VLConfig:
    vocab_size: int = 151936
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_hidden_layers: int = MINIMAX_H3_QWEN3VL_SELECTED_LAYERS
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    attention_bias: bool = False
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.num_hidden_layers <= MINIMAX_H3_QWEN3VL_SELECTED_LAYERS:
            raise ValueError("encoder may execute only a non-empty prefix of layers 0..49")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by key/value heads")
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError("mrope_section must cover half of head_dim")

    def validate_tensor_parallel(self, tp_size: int) -> None:
        if tp_size < 1:
            raise ValueError("TP size must be positive")
        for name, value in (
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("intermediate_size", self.intermediate_size),
        ):
            if value % tp_size:
                raise ValueError(f"{name}={value} must be divisible by TP={tp_size}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MiniMaxH3Qwen3VLConfig:
        text = payload.get("text_config", payload)
        rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            # Deliberately trim the 64-layer checkpoint at construction time.
            num_hidden_layers=MINIMAX_H3_QWEN3VL_SELECTED_LAYERS,
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(
                text.get(
                    "head_dim",
                    int(text["hidden_size"]) // int(text["num_attention_heads"]),
                )
            ),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(text.get("rope_theta", 5_000_000.0)),
            max_position_embeddings=int(
                text.get("max_position_embeddings", 262144)
            ),
            mrope_section=tuple(int(x) for x in rope.get("mrope_section", (24, 20, 20))),
            attention_bias=bool(text.get("attention_bias", False)),
            image_token_id=int(payload.get("image_token_id", 151655)),
            video_token_id=int(payload.get("video_token_id", 151656)),
            vision_start_token_id=int(payload.get("vision_start_token_id", 151652)),
            vision_end_token_id=int(payload.get("vision_end_token_id", 151653)),
            vision_config=(
                dict(payload["vision_config"])
                if payload.get("vision_config") is not None
                else None
            ),
        )

    @property
    def vision_spatial_merge_size(self) -> int:
        return int((self.vision_config or {}).get("spatial_merge_size", 2))

    @classmethod
    def from_pretrained(
        cls, checkpoint_dir: str | Path
    ) -> MiniMaxH3Qwen3VLConfig:
        path = Path(checkpoint_dir) / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"Qwen3-VL config not found: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "MINIMAX_H3_QWEN3VL_SELECTED_LAYERS",
    "MiniMaxH3Qwen3VLConfig",
]
