from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .fl2va_conditioning import FL2VA_KEYFRAME_SIGNATURES, FL2VAConditioning

PACKED_SEQUENCE_ALIGNMENT = 64

MINIMAX_H3_TEXT_ID = -5
MINIMAX_H3_IMGVID_COND_ID = -11
MINIMAX_H3_AUDIO_FIRST_ID = -15
MINIMAX_H3_AUDIO_ID = -14
MINIMAX_H3_VIDEO_FIRST_ID = -3
MINIMAX_H3_VIDEO_ID = -2
MINIMAX_H3_VIDEO_LAST_ID = -4
MINIMAX_H3_PAD_ID = -1

_INTERP = 32.0
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_PATCH_H = 2
_PATCH_W = 2


@dataclass(frozen=True, slots=True)
class FL2VAPackedSequence:
    """Structural tensors for ``text|condition|audio|video|pad64``."""

    sequence_length: int
    used_length: int
    text_slice: slice
    condition_slice: slice
    audio_slice: slice
    video_slice: slice
    pad_slice: slice
    input_ids: torch.Tensor
    image_mask: torch.Tensor
    audio_mask: torch.Tensor
    img_pos: torch.Tensor
    target_img_pos: torch.Tensor
    audio_pos: torch.Tensor
    text_pos: torch.Tensor
    update_mask: torch.Tensor
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    cu_seqlens: torch.Tensor
    document_id: torch.Tensor
    latent_grid: torch.Tensor

    @property
    def img_position_ids(self) -> torch.Tensor:
        return self.position_ids

    @property
    def seq_len(self) -> int:
        return self.sequence_length

    @property
    def max_seqlen(self) -> int:
        return max(self.used_length, self.sequence_length - self.used_length)

    def shard(self, rank: int, degree: int) -> slice:
        if degree <= 0 or not 0 <= rank < degree:
            raise ValueError("rank must be within a positive shard degree")
        if self.sequence_length % degree:
            raise ValueError("packed sequence length must be divisible by degree")
        local = self.sequence_length // degree
        return slice(rank * local, (rank + 1) * local)


def _axis_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    right = left + ratio
    grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
    return torch.from_numpy(grid).to(torch.float64)


def _time_grid(length: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            _FRAME_RESCALE * _FRAME_PER_TOKEN[index % len(_FRAME_PER_TOKEN)]
            for index in range(length)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        (torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0))
    )


def _temporal_position_span(length: int) -> float:
    # Keep numpy's summation order: SGLang uses this exact path for last-frame
    # anchors and it can differ from sequential fp64 summation by one ulp.
    spans = np.ones(int(length), dtype=np.float64) * _FRAME_RESCALE
    for index, multiplier in enumerate(_FRAME_PER_TOKEN):
        spans[index :: len(_FRAME_PER_TOKEN)] *= multiplier
    return float(spans.sum())


def _validate_keyframes(
    keyframe_indices: Sequence[int],
    frame_count: int | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in keyframe_indices):
        raise ValueError("keyframe_indices must contain integers")
    semantic = tuple(keyframe_indices)
    if semantic not in FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            "keyframe_indices must be (), (0,), (-1,), or (0, -1)"
        )
    if not semantic:
        return semantic, ()
    if frame_count is None or frame_count <= 0:
        raise ValueError("frame_count is required for keyframe conditioning")
    resolved = tuple(frame_count - 1 if value == -1 else value for value in semantic)
    if len(set(resolved)) != len(resolved):
        raise ValueError("keyframe indices must resolve to distinct frames")
    return semantic, resolved


def build_fl2va_packed_sequence(
    *,
    text_length: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channels: int = 2,
    keyframe_indices: Sequence[int] = (),
    frame_count: int | None = None,
    conditioning: FL2VAConditioning | None = None,
) -> FL2VAPackedSequence:
    """Build the exact SGLang FL2VA/T2VA packed-sequence layout."""

    for name, value in (
        ("text_length", text_length),
        ("latent_t", latent_t),
        ("latent_h", latent_h),
        ("latent_w", latent_w),
        ("audio_t", audio_t),
        ("audio_channels", audio_channels),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if latent_h % _PATCH_H or latent_w % _PATCH_W:
        raise ValueError("latent height and width must be divisible by 2")
    if conditioning is not None:
        if tuple(keyframe_indices):
            raise ValueError("pass conditioning or keyframe_indices, not both")
        keyframe_indices = conditioning.keyframe_indices
    _, resolved_indices = _validate_keyframes(keyframe_indices, frame_count)

    patch_h, patch_w = latent_h // _PATCH_H, latent_w // _PATCH_W
    frame_rows = patch_h * patch_w
    condition_rows = len(resolved_indices) * frame_rows
    audio_rows = audio_t * audio_channels
    video_rows = latent_t * frame_rows
    used = text_length + condition_rows + audio_rows + video_rows
    sequence_length = (
        (used + PACKED_SEQUENCE_ALIGNMENT - 1)
        // PACKED_SEQUENCE_ALIGNMENT
        * PACKED_SEQUENCE_ALIGNMENT
    )

    text_slice = slice(0, text_length)
    condition_slice = slice(text_slice.stop, text_slice.stop + condition_rows)
    audio_slice = slice(condition_slice.stop, condition_slice.stop + audio_rows)
    video_slice = slice(audio_slice.stop, audio_slice.stop + video_rows)
    pad_slice = slice(video_slice.stop, sequence_length)

    input_ids = torch.full(
        (sequence_length,), MINIMAX_H3_PAD_ID, dtype=torch.long
    )
    input_ids[text_slice] = MINIMAX_H3_TEXT_ID
    input_ids[condition_slice] = MINIMAX_H3_IMGVID_COND_ID
    input_ids[audio_slice] = MINIMAX_H3_AUDIO_ID
    input_ids[audio_slice.start] = MINIMAX_H3_AUDIO_FIRST_ID
    input_ids[video_slice] = MINIMAX_H3_VIDEO_ID
    input_ids[video_slice.start] = MINIMAX_H3_VIDEO_FIRST_ID
    input_ids[video_slice.stop - 1] = MINIMAX_H3_VIDEO_LAST_ID

    image_mask = torch.zeros(sequence_length, dtype=torch.bool)
    image_mask[condition_slice] = True
    image_mask[video_slice] = True
    audio_mask = torch.zeros(sequence_length, dtype=torch.bool)
    audio_mask[audio_slice] = True

    condition_pos = torch.arange(condition_slice.start, condition_slice.stop)
    target_img_pos = torch.arange(video_slice.start, video_slice.stop)
    img_pos = (
        torch.cat((condition_pos, target_img_pos))
        if condition_rows
        else target_img_pos
    )
    audio_pos = torch.arange(audio_slice.start, audio_slice.stop)
    text_pos = torch.arange(text_length)
    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[condition_rows:] = True

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[text_slice, 0] = torch.arange(
        text_length, dtype=torch.float64
    )
    sqrt_area = float(np.sqrt(latent_h * latent_w))
    h_grid = _axis_grid(latent_h, _PATCH_H, sqrt_area)
    w_grid = _axis_grid(latent_w, _PATCH_W, sqrt_area)
    grid_h, grid_w = torch.meshgrid(h_grid, w_grid, indexing="ij")
    spatial = torch.stack((grid_h.flatten(), grid_w.flatten()), dim=-1)

    video_grid = position_ids[video_slice].view(latent_t, frame_rows, 3)
    video_grid[..., 0] = _time_grid(latent_t, float(text_length))[:, None]
    video_grid[..., 1:] = spatial
    for block_index, frame_index in enumerate(resolved_indices):
        start = condition_slice.start + block_index * frame_rows
        block_slice = slice(start, start + frame_rows)
        if frame_index == 0:
            time_position = float(text_length)
        elif frame_count is not None and frame_index == frame_count - 1:
            time_position = (
                float(text_length)
                + _temporal_position_span(latent_t)
                - _FRAME_RESCALE
            )
        else:
            raise ValueError("only first/last frame anchors are supported")
        position_ids[block_slice, 0] = time_position
        position_ids[block_slice, 1:] = spatial

    audio_time = float(text_length) + torch.arange(
        audio_t, dtype=torch.float64
    )
    position_ids[audio_slice, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_slice.start : audio_slice.start + audio_t, 2] = float(
        w_grid[0]
    )
    position_ids[audio_slice.start + audio_t : audio_slice.stop, 2] = float(
        w_grid[-1]
    )

    token_tags = torch.full((sequence_length,), -1, dtype=torch.long)
    token_tags[text_slice] = 1
    token_tags[audio_slice] = 2
    token_tags[img_pos] = 0
    boundaries = [0, used]
    if sequence_length > used:
        boundaries.append(sequence_length)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
    document_id = torch.zeros(sequence_length, dtype=torch.int32)
    document_id[pad_slice] = 1
    return FL2VAPackedSequence(
        sequence_length=sequence_length,
        used_length=used,
        text_slice=text_slice,
        condition_slice=condition_slice,
        audio_slice=audio_slice,
        video_slice=video_slice,
        pad_slice=pad_slice,
        input_ids=input_ids,
        image_mask=image_mask,
        audio_mask=audio_mask,
        img_pos=img_pos,
        target_img_pos=target_img_pos,
        audio_pos=audio_pos,
        text_pos=text_pos,
        update_mask=update_mask,
        position_ids=position_ids,
        token_tags=token_tags,
        cu_seqlens=cu_seqlens,
        document_id=document_id,
        latent_grid=torch.tensor(
            [latent_t, patch_h, patch_w], dtype=torch.int64
        ),
    )


def minimax_h3_packed_sequence(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
    include_keyframe_cond: bool,
    keyframe_frame_indices: Sequence[int] | None = None,
    frame_count: int | None = None,
) -> FL2VAPackedSequence:
    """SGLang-shaped wrapper around :func:`build_fl2va_packed_sequence`."""

    if include_keyframe_cond:
        if keyframe_frame_indices is None:
            raise ValueError("keyframe_frame_indices are required")
        indices = keyframe_frame_indices
    else:
        if keyframe_frame_indices is not None:
            raise ValueError(
                "keyframe_frame_indices must be omitted without conditioning"
            )
        indices = ()
    return build_fl2va_packed_sequence(
        text_length=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        audio_channels=audio_channel,
        keyframe_indices=indices,
        frame_count=frame_count,
    )


__all__ = [
    "MINIMAX_H3_AUDIO_FIRST_ID",
    "MINIMAX_H3_AUDIO_ID",
    "MINIMAX_H3_IMGVID_COND_ID",
    "MINIMAX_H3_PAD_ID",
    "MINIMAX_H3_TEXT_ID",
    "MINIMAX_H3_VIDEO_FIRST_ID",
    "MINIMAX_H3_VIDEO_ID",
    "MINIMAX_H3_VIDEO_LAST_ID",
    "PACKED_SEQUENCE_ALIGNMENT",
    "FL2VAPackedSequence",
    "build_fl2va_packed_sequence",
    "minimax_h3_packed_sequence",
]
