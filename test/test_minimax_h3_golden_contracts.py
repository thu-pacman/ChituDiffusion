from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "script" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = _load_script("verify_h3_e2e_parity")


def test_presentation_reference_preserves_prompt_and_tag_alignment() -> None:
    prompt = "  Golden prompt: 海浪\n"
    contract = {"prompt": prompt}
    built = verify.reference_presentation(contract)

    assert built["text_only_ids"].tolist() == [ord(char) for char in prompt]
    assert len(built["multi_image"]["ids"]) == len(
        built["multi_image"]["token_tags"]
    )
    assert len(built["ref_mixed"]["ids"]) == len(
        built["ref_mixed"]["token_tags"]
    )
    assert built["multi_image"]["token_tags"].unique().tolist() == [0, 1]
    video_pad = verify.SPECIAL_IDS["<|video_pad|>"]
    video_positions = built["ref_mixed"]["ids"] == video_pad
    assert int(video_positions.sum()) == 5
    assert not built["ref_mixed"]["token_tags"][video_positions].any()


def test_cover_crop_reference_uses_cover_scale_and_center_floor() -> None:
    plan = verify.reference_cover_crop((1001, 777), (768, 512), False)

    assert plan["scale"] == 768 / 1001
    assert plan["resized_size"] == (768, 596)
    assert plan["crop_box"] == (0, 42, 768, 554)


def test_condition_noise_restarts_rng_for_each_reference() -> None:
    clean = torch.zeros(8, 32)
    actual = verify.reference_audio_noise(
        clean,
        condition_audio_t=[2, 2],
        seed=17,
        noise_aug=0.0,
    )

    torch.testing.assert_close(actual[:4], actual[4:], rtol=0, atol=0)
    generator = torch.Generator(device="cpu").manual_seed(18)
    expected = torch.randn((4, 32), generator=generator)
    torch.testing.assert_close(actual[:4], expected, rtol=0, atol=0)


def test_image_noise_draws_target_plus_condition_temporal_length() -> None:
    shape = (1, 2, 2)
    clean = torch.zeros(1, 96)
    actual = verify.reference_image_noise(
        clean,
        condition_shapes=[shape],
        target_latent_t=3,
        imgvid_cond_num_frames=2,
        seed=9,
        noise_aug=0.0,
    )
    generator = torch.Generator(device="cpu").manual_seed(9)
    full_draw = torch.randn(1, 24, 5, 2, 2, generator=generator)[:, :, :1]

    torch.testing.assert_close(
        actual, verify.reference_patchify(full_draw), rtol=0, atol=0
    )


def test_standard_packed_layout_tracks_cond_and_target_rows() -> None:
    arguments = {
        "text_len": 11,
        "latent_t": 16,
        "latent_h": 4,
        "latent_w": 6,
        "audio_t": 7,
        "include_keyframe_cond": True,
        "keyframe_frame_indices": [0, -1],
        "frame_count": 49,
    }
    layout = verify.reference_standard_layout(arguments)
    frame_rows = 2 * 3

    assert layout["seq_len"] % 64 == 0
    assert int((~layout["update_mask"]).sum()) == 2 * frame_rows
    first_cond = layout["img_pos"][:frame_rows]
    last_cond = layout["img_pos"][frame_rows : 2 * frame_rows]
    assert layout["img_position_ids"][first_cond, 0].unique().item() == 11.0
    spans = torch.tensor(
        [
            verify.FRAME_RESCALE * verify.FRAME_PER_TOKEN[index % 5]
            for index in range(16)
        ],
        dtype=torch.float64,
    )
    expected_last = 11.0 + float(spans.sum()) - verify.FRAME_RESCALE
    assert layout["img_position_ids"][last_cond, 0].unique().item() == expected_last
    assert layout["token_tags"][layout["audio_pos"]].unique().tolist() == [2]


def test_ref_layout_preserves_shared_video_audio_origin() -> None:
    arguments = {
        "text_len": 5,
        "latent_t": 3,
        "latent_h": 4,
        "latent_w": 6,
        "audio_t": 4,
        "ref_blocks": [
            {"kind": "image", "latent_h": 4, "latent_w": 4},
            {
                "kind": "video_audio",
                "ref_audio_t": 3,
                "latent_t": 2,
                "latent_h": 4,
                "latent_w": 6,
            },
            {"kind": "audio", "ref_audio_t": 1},
        ],
    }
    layout = verify.reference_ref_layout(arguments)
    image_rows = 4
    video_audio_rows = 6

    ref_video_t0 = layout["img_position_ids"][
        layout["img_pos"][image_rows], 0
    ]
    ref_audio_t0 = layout["img_position_ids"][
        layout["audio_pos"][0], 0
    ]
    assert float(ref_video_t0) == float(ref_audio_t0)
    assert int((~layout["update_mask"]).sum()) == image_rows + 12
    assert int((~layout["audio_update_mask"]).sum()) == video_audio_rows + 2


def test_heavy_stages_are_not_default_dump_stages() -> None:
    dump = _load_script("dump_h3_e2e_golden")

    assert set(dump.LIGHT_STAGES).isdisjoint(dump.HEAVY_STAGES)
    assert dump.LIGHT_STAGES == ("presentation", "geometry", "noise", "packed")
