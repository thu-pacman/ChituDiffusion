from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .fl2va_geometry import FL2VAResolvedPlan

FL2VA_KEYFRAME_SIGNATURES: tuple[tuple[int, ...], ...] = (
    (),
    (0,),
    (-1,),
    (0, -1),
)

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
TEXT_TOKEN_TAG = 1
VIDEO_TOKEN_TAG = 0


@dataclass(frozen=True, slots=True)
class FL2VAFrameMetadata:
    """Metadata for one request-ordered first/last keyframe."""

    frame_index: int
    uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or self.frame_index not in (0, -1):
            raise ValueError("frame_index must be 0 (first) or -1 (last)")


@dataclass(frozen=True, slots=True)
class FL2VAConditioning:
    """Prompt plus optional endpoint-frame metadata.

    An empty ``frames`` tuple is the prompt-only/T2VA path on FL2VA weights.
    """

    prompt: str
    frames: tuple[FL2VAFrameMetadata, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        signature = self.keyframe_indices
        if signature not in FL2VA_KEYFRAME_SIGNATURES:
            raise ValueError(
                "frames must be ordered as first, last, or first+last; "
                f"got {signature!r}"
            )

    @property
    def keyframe_indices(self) -> tuple[int, ...]:
        return tuple(frame.frame_index for frame in self.frames)

    @property
    def prompt_only(self) -> bool:
        return not self.frames

    def resolved_frame_indices(self, frame_count: int) -> tuple[int, ...]:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        return tuple(
            frame_count - 1 if index == -1 else index
            for index in self.keyframe_indices
        )


@dataclass(frozen=True, slots=True)
class FL2VAPresentation:
    input_ids: torch.Tensor
    token_tags: torch.Tensor


def _text_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(value) for value in encoded["input_ids"]]


def _vision_ids(tokenizer: Any, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("image token counts must be positive")
    return [
        int(tokenizer.convert_tokens_to_ids(VISION_START)),
        *([int(tokenizer.convert_tokens_to_ids(IMAGE_PAD))] * count),
        int(tokenizer.convert_tokens_to_ids(VISION_END)),
    ]


def build_fl2va_presentation(
    tokenizer: Any,
    conditioning: FL2VAConditioning,
    *,
    image_token_counts: Sequence[int] = (),
) -> FL2VAPresentation:
    """Build SGLang-compatible Qwen ids and aligned AdaLN token tags."""

    counts = tuple(int(value) for value in image_token_counts)
    if len(counts) != len(conditioning.frames):
        raise ValueError("one image_token_count is required per frame")
    if conditioning.prompt_only:
        ids = _text_ids(tokenizer, conditioning.prompt)
        return FL2VAPresentation(
            input_ids=torch.tensor(ids, dtype=torch.long),
            token_tags=torch.full((len(ids),), TEXT_TOKEN_TAG, dtype=torch.long),
        )

    ids: list[int] = []
    tags: list[int] = []
    for count in counts:
        label = _text_ids(tokenizer, ": ")
        vision = _vision_ids(tokenizer, count)
        prompt = _text_ids(tokenizer, conditioning.prompt)
        ids.extend(label)
        tags.extend([TEXT_TOKEN_TAG] * len(label))
        ids.extend(vision)
        tags.extend([VIDEO_TOKEN_TAG] * len(vision))
        ids.extend(prompt)
        tags.extend([TEXT_TOKEN_TAG] * len(prompt))
    return FL2VAPresentation(
        input_ids=torch.tensor(ids, dtype=torch.long),
        token_tags=torch.tensor(tags, dtype=torch.long),
    )


@dataclass(frozen=True, slots=True)
class FL2VAInitialNoise:
    """CPU-fp32 latent noise in both native and DiT-row layouts."""

    video: torch.Tensor
    video_rows: torch.Tensor
    audio_rows: torch.Tensor


def patchify_video_latent(
    latent: torch.Tensor,
    *,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    if latent.ndim != 5:
        raise ValueError("video latent must have shape [B,C,T,H,W]")
    pt, ph, pw = patch_size
    batch, channels, full_t, full_h, full_w = latent.shape
    if full_t % pt or full_h % ph or full_w % pw:
        raise ValueError("video latent dimensions must be divisible by patch_size")
    t, h, w = full_t // pt, full_h // ph, full_w // pw
    packed = latent.reshape(batch, channels, t, pt, h, ph, w, pw)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(batch * t * h * w, -1).contiguous()


def prepare_fl2va_noise(
    plan: FL2VAResolvedPlan,
    *,
    seed: int | None = None,
    video_channels: int = 24,
    audio_channels: int = 2,
    audio_latent_dim: int = 32,
) -> FL2VAInitialNoise:
    """Generate SGLang-identical CPU fp32 video/audio noise streams.

    Video and audio each receive a newly seeded CPU generator. Consequently,
    changing video shape never advances or changes the audio random stream.
    """

    resolved_seed = plan.seed if seed is None else seed
    if isinstance(resolved_seed, bool) or int(resolved_seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if min(video_channels, audio_channels, audio_latent_dim) <= 0:
        raise ValueError("latent channel dimensions must be positive")

    video_generator = torch.Generator(device="cpu").manual_seed(int(resolved_seed))
    video = torch.randn(
        1,
        video_channels,
        plan.video_latent_t,
        plan.latent_height,
        plan.latent_width,
        generator=video_generator,
        dtype=torch.float32,
        device="cpu",
    )
    audio_generator = torch.Generator(device="cpu").manual_seed(int(resolved_seed))
    audio_rows = torch.randn(
        plan.audio_latent_t * audio_channels,
        audio_latent_dim,
        generator=audio_generator,
        dtype=torch.float32,
        device="cpu",
    )
    return FL2VAInitialNoise(
        video=video,
        video_rows=patchify_video_latent(video),
        audio_rows=audio_rows,
    )


prepare_fl2va_latents = prepare_fl2va_noise
minimax_h3_patchify_video_latent = patchify_video_latent


__all__ = [
    "FL2VA_KEYFRAME_SIGNATURES",
    "FL2VAConditioning",
    "FL2VAFrameMetadata",
    "FL2VAInitialNoise",
    "FL2VAPresentation",
    "build_fl2va_presentation",
    "minimax_h3_patchify_video_latent",
    "patchify_video_latent",
    "prepare_fl2va_latents",
    "prepare_fl2va_noise",
]
