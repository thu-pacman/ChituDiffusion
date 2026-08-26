from __future__ import annotations

from dataclasses import dataclass

import torch

PACKED_SEQUENCE_ALIGNMENT = 64
_INTERP = 32.0
_FRAME_RESCALE = 5.0 / 3.0
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


@dataclass(frozen=True)
class MiniMaxH3PackedSequence:
    sequence_length: int
    used_length: int
    img_pos: torch.Tensor
    audio_pos: torch.Tensor
    text_pos: torch.Tensor
    target_img_pos: torch.Tensor
    update_mask: torch.Tensor
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    cu_seqlens: torch.Tensor

    @property
    def max_seqlen(self) -> int:
        return max(self.used_length, self.sequence_length - self.used_length)

    def shard(self, rank: int, degree: int) -> slice:
        if self.sequence_length % degree:
            raise ValueError("packed sequence length must be divisible by degree")
        local = self.sequence_length // degree
        return slice(rank * local, (rank + 1) * local)


def _axis_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    count = dim // patch
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return (
        left + ratio * torch.arange(count, dtype=torch.float64) / count
    ) * _INTERP


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


def build_packed_sequence(
    *,
    text_length: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channels: int = 2,
    keyframe_indices: tuple[int, ...] = (),
    frame_count: int | None = None,
) -> MiniMaxH3PackedSequence:
    for name, value in (
        ("text_length", text_length),
        ("latent_t", latent_t),
        ("latent_h", latent_h),
        ("latent_w", latent_w),
        ("audio_t", audio_t),
        ("audio_channels", audio_channels),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if latent_h % 2 or latent_w % 2:
        raise ValueError("latent height and width must be divisible by patch size 2")
    if keyframe_indices and frame_count is None:
        raise ValueError("frame_count is required for keyframe conditioning")

    patch_h, patch_w = latent_h // 2, latent_w // 2
    frame_rows = patch_h * patch_w
    condition_rows = len(keyframe_indices) * frame_rows
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
    condition_pos = torch.arange(condition_slice.start, condition_slice.stop)
    target_img_pos = torch.arange(video_slice.start, video_slice.stop)
    img_pos = torch.cat((condition_pos, target_img_pos))
    audio_pos = torch.arange(audio_slice.start, audio_slice.stop)
    text_pos = torch.arange(text_length)
    update_mask = torch.cat(
        (
            torch.zeros(condition_rows, dtype=torch.bool),
            torch.ones(video_rows, dtype=torch.bool),
        )
    )

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[text_slice, 0] = torch.arange(
        text_length, dtype=torch.float64
    )
    sqrt_area = float((latent_h * latent_w) ** 0.5)
    h_grid = _axis_grid(latent_h, 2, sqrt_area)
    w_grid = _axis_grid(latent_w, 2, sqrt_area)
    grid_h, grid_w = torch.meshgrid(h_grid, w_grid, indexing="ij")
    spatial = torch.stack((grid_h.flatten(), grid_w.flatten()), dim=-1)
    video_grid = position_ids[video_slice].view(latent_t, frame_rows, 3)
    video_grid[..., 0] = _time_grid(latent_t, float(text_length))[:, None]
    video_grid[..., 1:] = spatial

    for block_index, semantic_index in enumerate(keyframe_indices):
        resolved = frame_count - 1 if semantic_index == -1 else semantic_index
        if resolved not in {0, frame_count - 1}:
            raise ValueError("FL2VA supports only first/last keyframe anchors")
        start = condition_slice.start + block_index * frame_rows
        block_slice = slice(start, start + frame_rows)
        if resolved == 0:
            time_position = float(text_length)
        else:
            time_position = float(_time_grid(latent_t, text_length)[-1])
        position_ids[block_slice, 0] = time_position
        position_ids[block_slice, 1:] = spatial

    audio_time = (
        float(text_length) + torch.arange(audio_t, dtype=torch.float64)
    )
    position_ids[audio_slice, 0] = audio_time.repeat(audio_channels)
    for channel in range(audio_channels):
        channel_slice = slice(
            audio_slice.start + channel * audio_t,
            audio_slice.start + (channel + 1) * audio_t,
        )
        position_ids[channel_slice, 2] = (
            w_grid[0] if channel == 0 else w_grid[-1]
        )

    token_tags = torch.full((sequence_length,), -1, dtype=torch.long)
    token_tags[text_slice] = 1
    token_tags[audio_slice] = 2
    token_tags[img_pos] = 0
    boundaries = [0, used]
    if sequence_length > used:
        boundaries.append(sequence_length)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32)
    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        used_length=used,
        img_pos=img_pos,
        audio_pos=audio_pos,
        text_pos=text_pos,
        target_img_pos=target_img_pos,
        update_mask=update_mask,
        position_ids=position_ids,
        token_tags=token_tags,
        cu_seqlens=cu_seqlens,
    )

