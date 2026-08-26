from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import torch

from chitu_diffusion.models.minimax_h3.fl2va_conditioning import (
    FL2VAConditioning,
    FL2VAFrameMetadata,
    build_fl2va_presentation,
    prepare_fl2va_noise,
)
from chitu_diffusion.models.minimax_h3.fl2va_geometry import (
    FL2VACanvas,
    FL2VAResolvedPlan,
    FL2VATimeRequest,
    align_frame_count,
    audio_latent_t,
    frame_count_from_video_latent_t,
    resolve_canvas,
    resolve_fl2va_plan,
    video_latent_t,
)
from chitu_diffusion.models.minimax_h3.fl2va_packed_sequence import (
    MINIMAX_H3_AUDIO_FIRST_ID,
    MINIMAX_H3_IMGVID_COND_ID,
    MINIMAX_H3_PAD_ID,
    MINIMAX_H3_TEXT_ID,
    MINIMAX_H3_VIDEO_FIRST_ID,
    MINIMAX_H3_VIDEO_LAST_ID,
    build_fl2va_packed_sequence,
)
from chitu_diffusion.models.minimax_h3.packed_sequence import build_packed_sequence


class _Tokenizer:
    _special: ClassVar[dict[str, int]] = {
        "<|vision_start|>": 1001,
        "<|image_pad|>": 1002,
        "<|vision_end|>": 1003,
    }

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special[token]


def test_fl2va_exact_time_and_canvas_formulas() -> None:
    assert [align_frame_count(value) for value in (1, 5, 6, 22, 120)] == [
        5,
        5,
        22,
        22,
        124,
    ]
    assert align_frame_count(0) == 1
    assert [video_latent_t(value) for value in (5, 22, 39, 124)] == [
        2,
        7,
        12,
        37,
    ]
    assert frame_count_from_video_latent_t(37) == 124
    assert audio_latent_t(124 / 24) == 207

    canvas = resolve_canvas(16, 9)
    assert (canvas.width, canvas.height, canvas.size_mode) == (1344, 768, "area")
    assert (canvas.latent_width, canvas.latent_height) == (84, 48)

    plan = resolve_fl2va_plan(
        FL2VATimeRequest(duration_seconds=5.0, width=16, height=9, seed=7)
    )
    assert (
        plan.requested_frame_count,
        plan.frame_count,
        plan.video_latent_t,
        plan.audio_latent_t,
    ) == (120, 124, 37, 207)
    assert plan.duration_seconds == 124 / 24


def test_conditioning_supports_prompt_only_and_endpoint_metadata() -> None:
    prompt_only = FL2VAConditioning("hello")
    assert prompt_only.prompt_only
    assert prompt_only.resolved_frame_indices(22) == ()

    frames = FL2VAConditioning(
        "hello",
        (
            FL2VAFrameMetadata(0, uri="first.png"),
            FL2VAFrameMetadata(-1, uri="last.png"),
        ),
    )
    assert frames.keyframe_indices == (0, -1)
    assert frames.resolved_frame_indices(22) == (0, 21)
    with pytest.raises(ValueError, match="ordered"):
        FL2VAConditioning(
            "hello",
            (FL2VAFrameMetadata(-1), FL2VAFrameMetadata(0)),
        )


def test_presentation_matches_sglang_text_and_vision_tagging() -> None:
    tokenizer = _Tokenizer()
    prompt_only = build_fl2va_presentation(
        tokenizer,
        FL2VAConditioning("ab"),
    )
    assert prompt_only.input_ids.tolist() == [ord("a"), ord("b")]
    assert prompt_only.token_tags.tolist() == [1, 1]

    conditioned = build_fl2va_presentation(
        tokenizer,
        FL2VAConditioning("x", (FL2VAFrameMetadata(0),)),
        image_token_counts=(2,),
    )
    assert conditioned.input_ids.tolist() == [
        ord(":"),
        ord(" "),
        1001,
        1002,
        1002,
        1003,
        ord("x"),
    ]
    assert conditioned.token_tags.tolist() == [1, 1, 0, 0, 0, 0, 1]


def test_cpu_fp32_noise_resets_generator_for_each_modality() -> None:
    plan = FL2VAResolvedPlan(
        requested_frame_count=22,
        frame_count=22,
        duration_seconds=22 / 24,
        fps=24,
        video_latent_t=7,
        audio_latent_t=37,
        canvas=FL2VACanvas(32, 32, 1, 1, "short_edge"),
        seed=123,
    )
    noise = prepare_fl2va_noise(plan)
    assert noise.video.device.type == noise.audio_rows.device.type == "cpu"
    assert noise.video.dtype == noise.audio_rows.dtype == torch.float32
    assert noise.video.shape == (1, 24, 7, 2, 2)
    assert noise.video_rows.shape == (7, 96)
    assert noise.audio_rows.shape == (74, 32)
    torch.testing.assert_close(
        noise.video.flatten()[:32],
        noise.audio_rows[0],
        rtol=0,
        atol=0,
    )


def test_packed_layout_positions_labels_and_masks_match_sglang() -> None:
    packed = build_fl2va_packed_sequence(
        text_length=3,
        latent_t=7,
        latent_h=4,
        latent_w=6,
        audio_t=4,
        keyframe_indices=(0, -1),
        frame_count=22,
    )
    assert packed.used_length == 65
    assert packed.sequence_length == 128
    assert packed.text_slice == slice(0, 3)
    assert packed.condition_slice == slice(3, 15)
    assert packed.audio_slice == slice(15, 23)
    assert packed.video_slice == slice(23, 65)
    assert packed.pad_slice == slice(65, 128)
    assert packed.input_ids[0].item() == MINIMAX_H3_TEXT_ID
    assert packed.input_ids[3].item() == MINIMAX_H3_IMGVID_COND_ID
    assert packed.input_ids[15].item() == MINIMAX_H3_AUDIO_FIRST_ID
    assert packed.input_ids[23].item() == MINIMAX_H3_VIDEO_FIRST_ID
    assert packed.input_ids[64].item() == MINIMAX_H3_VIDEO_LAST_ID
    assert packed.input_ids[65].item() == MINIMAX_H3_PAD_ID
    assert packed.update_mask.tolist() == [False] * 12 + [True] * 42
    assert packed.token_tags[:3].tolist() == [1, 1, 1]
    assert packed.token_tags[15:23].tolist() == [2] * 8
    assert packed.token_tags[23:65].tolist() == [0] * 42
    assert packed.token_tags[65:].tolist() == [-1] * 63
    assert packed.cu_seqlens.tolist() == [0, 65, 128]
    assert packed.document_id[:65].tolist() == [0] * 65
    assert packed.document_id[65:].tolist() == [1] * 63

    spans = np.ones(7, dtype=np.float64) * (5.0 / 3.0)
    for index, multiplier in enumerate((1, 4, 4, 4, 4)):
        spans[index::5] *= multiplier
    expected_last_anchor = 3.0 + float(spans.sum()) - 5.0 / 3.0
    assert packed.position_ids[3, 0].item() == 3.0
    assert packed.position_ids[9, 0].item() == expected_last_anchor
    assert packed.position_ids[23, 0].item() == 3.0
    assert packed.position_ids[15:19, 0].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert packed.position_ids[19:23, 0].tolist() == [3.0, 4.0, 5.0, 6.0]


def test_prompt_only_layout_has_no_condition_rows() -> None:
    packed = build_fl2va_packed_sequence(
        text_length=2,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=2,
    )
    assert packed.condition_slice == slice(2, 2)
    assert packed.img_pos.equal(packed.target_img_pos)
    assert packed.update_mask.all()


def test_aligned_packed_layouts_do_not_append_empty_padding_documents() -> None:
    arguments = {
        "text_length": 4,
        "latent_t": 14,
        "latent_h": 4,
        "latent_w": 4,
        "audio_t": 2,
    }
    latent = build_packed_sequence(**arguments)
    fl2va = build_fl2va_packed_sequence(**arguments)

    for packed in (latent, fl2va):
        assert packed.used_length == packed.sequence_length == 64
        assert packed.cu_seqlens.tolist() == [0, 64]


def test_padded_packed_layouts_keep_live_and_padding_documents() -> None:
    arguments = {
        "text_length": 5,
        "latent_t": 14,
        "latent_h": 4,
        "latent_w": 4,
        "audio_t": 2,
    }
    latent = build_packed_sequence(**arguments)
    fl2va = build_fl2va_packed_sequence(**arguments)

    for packed in (latent, fl2va):
        assert packed.used_length == 65
        assert packed.sequence_length == 128
        assert packed.cu_seqlens.tolist() == [0, 65, 128]
