from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SPECIAL_IDS = {
    "<|vision_start|>": 1_000_001,
    "<|vision_end|>": 1_000_002,
    "<|image_pad|>": 1_000_003,
    "<|video_pad|>": 1_000_004,
}
LIGHT_STAGES = ("presentation", "geometry", "noise", "packed")
HEAVY_STAGES = ("qwen", "vae", "steps")
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _text(text: str) -> list[int]:
    return [ord(char) for char in text]


def _vision(pad: str, count: int) -> list[int]:
    return [SPECIAL_IDS["<|vision_start|>"]] + [SPECIAL_IDS[pad]] * count + [
        SPECIAL_IDS["<|vision_end|>"]
    ]


def reference_presentation(contract: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(contract["prompt"])
    multi_ids: list[int] = []
    multi_tags: list[int] = []
    for index, count in enumerate((2, 3), start=1):
        label = _text(f"<Picture {index}>: ")
        block = _vision("<|image_pad|>", count)
        multi_ids += label + block
        multi_tags += [1] * len(label) + [0] * len(block)
    multi_ids += _text(prompt)
    multi_tags += [1] * len(_text(prompt))

    ref_ids: list[int] = []
    ref_tags: list[int] = []
    segments = [
        ("text", _text("<Audio 1>: ")),
        ("text", _text("<Picture 1>: ")),
        ("vision", _vision("<|image_pad|>", 2)),
        ("text", _text("<Video 1>: ")),
        ("text", _text("<0.2 seconds>")),
        ("vision", _vision("<|video_pad|>", 3)),
        ("text", _text("<1.1 seconds>")),
        ("vision", _vision("<|video_pad|>", 2)),
        ("text", _text(prompt)),
    ]
    for kind, ids in segments:
        ref_ids += ids
        ref_tags += [0 if kind == "vision" else 1] * len(ids)
    return {
        "prompt": prompt,
        "special_ids": dict(SPECIAL_IDS),
        "text_only_ids": torch.tensor(_text(prompt), dtype=torch.long),
        "multi_image": {
            "ids": torch.tensor(multi_ids, dtype=torch.long),
            "token_tags": torch.tensor(multi_tags, dtype=torch.long),
        },
        "ref_mixed": {
            "ids": torch.tensor(ref_ids, dtype=torch.long),
            "token_tags": torch.tensor(ref_tags, dtype=torch.long),
        },
    }


def reference_cover_crop(
    source_size: Sequence[int],
    target_size: Sequence[int],
    allow_upscale: bool,
) -> dict[str, Any]:
    source_w, source_h = (int(value) for value in source_size)
    target_w, target_h = (int(value) for value in target_size)
    scale = max(target_w / float(source_w), target_h / float(source_h))
    if scale > 1.0 and not allow_upscale:
        raise ValueError("cover crop would upscale")
    resized_w = max(target_w, int(round(source_w * scale)))
    resized_h = max(target_h, int(round(source_h * scale)))
    left = max(0, (resized_w - target_w) // 2)
    top = max(0, (resized_h - target_h) // 2)
    return {
        "scale": scale,
        "resized_size": (resized_w, resized_h),
        "crop_box": (left, top, left + target_w, top + target_h),
    }


def reference_patchify(latent: torch.Tensor) -> torch.Tensor:
    batch, channels, temporal, height, width = latent.shape
    packed = latent.reshape(batch, channels, temporal, 1, height // 2, 2, width // 2, 2)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(batch * temporal * (height // 2) * (width // 2), channels * 4)


def reference_image_noise(
    clean_rows: torch.Tensor,
    *,
    condition_shapes: Sequence[Sequence[int]],
    target_latent_t: int,
    imgvid_cond_num_frames: int,
    seed: int,
    noise_aug: float,
) -> torch.Tensor:
    parts = []
    offset = 0
    for temporal, height, width in condition_shapes:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(
            1,
            24,
            int(target_latent_t) + int(imgvid_cond_num_frames),
            int(height),
            int(width),
            generator=generator,
            dtype=torch.float32,
        )[:, :, : int(temporal)]
        noise_rows = reference_patchify(noise)
        count = int(noise_rows.shape[0])
        clean = clean_rows[offset : offset + count].float()
        parts.append(float(noise_aug) * clean + (1.0 - float(noise_aug)) * noise_rows)
        offset += count
    return torch.cat(parts).contiguous()


def reference_audio_noise(
    clean_rows: torch.Tensor,
    *,
    condition_audio_t: Sequence[int],
    seed: int,
    noise_aug: float,
) -> torch.Tensor:
    parts = []
    offset = 0
    for audio_t in condition_audio_t:
        count = 2 * int(audio_t)
        clean = clean_rows[offset : offset + count].cpu().float()
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
        parts.append(float(noise_aug) * clean + (1.0 - float(noise_aug)) * noise)
        offset += count
    return torch.cat(parts).contiguous()


def _axis(dim: int, other: int) -> torch.Tensor:
    sqrt_area = np.sqrt(dim * other)
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return torch.from_numpy(
        np.linspace(left, left + ratio, dim // 2, endpoint=False) * 32
    ).to(torch.float64)


def _t_grid(length: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [FRAME_RESCALE * FRAME_PER_TOKEN[index % 5] for index in range(length)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _base_layout(
    *,
    text_len: int,
    cond_rows: int,
    audio_rows: int,
    video_rows: int,
) -> tuple[int, slice, slice, slice, slice]:
    used = text_len + cond_rows + audio_rows + video_rows
    seq_len = (used + 63) // 64 * 64
    text_sl = slice(0, text_len)
    cond_sl = slice(text_len, text_len + cond_rows)
    audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)
    return seq_len, text_sl, cond_sl, audio_sl, video_sl


def reference_standard_layout(arguments: Mapping[str, Any]) -> dict[str, Any]:
    text_len = int(arguments["text_len"])
    latent_t = int(arguments["latent_t"])
    latent_h = int(arguments["latent_h"])
    latent_w = int(arguments["latent_w"])
    audio_t = int(arguments["audio_t"])
    frame_rows = (latent_h // 2) * (latent_w // 2)
    keyframes = list(arguments.get("keyframe_frame_indices") or [])
    cond_rows = len(keyframes) * frame_rows
    seq_len, text_sl, cond_sl, audio_sl, video_sl = _base_layout(
        text_len=text_len,
        cond_rows=cond_rows,
        audio_rows=audio_t * 2,
        video_rows=latent_t * frame_rows,
    )
    target_pos = torch.arange(video_sl.start, video_sl.stop)
    img_pos = (
        torch.cat([torch.arange(cond_sl.start, cond_sl.stop), target_pos])
        if cond_rows
        else target_pos
    )
    audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
    positions = torch.zeros(seq_len, 3, dtype=torch.float64)
    positions[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
    h_grid, w_grid = _axis(latent_h, latent_w), _axis(latent_w, latent_h)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack((hh.flatten(), ww.flatten()), dim=-1)
    target_grid = positions[video_sl].view(latent_t, frame_rows, 3)
    target_grid[:, :, 0] = _t_grid(latent_t, float(text_len))[:, None]
    target_grid[:, :, 1:] = frame
    for index, semantic_index in enumerate(keyframes):
        cond_t = float(text_len)
        if semantic_index == -1:
            spans = np.ones(latent_t, dtype=np.float64) * FRAME_RESCALE
            for group in range(5):
                spans[group::5] *= FRAME_PER_TOKEN[group]
            cond_t += float(spans.sum()) - FRAME_RESCALE
        block = slice(cond_sl.start + index * frame_rows, cond_sl.start + (index + 1) * frame_rows)
        positions[block, 0] = cond_t
        positions[block, 1:] = frame
    positions[audio_sl, 0] = (
        float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    ).repeat(2)
    positions[audio_sl.start : audio_sl.start + audio_t, 2] = w_grid[0]
    positions[audio_sl.start + audio_t : audio_sl.stop, 2] = w_grid[-1]
    update_mask = torch.zeros(len(img_pos), dtype=torch.bool)
    update_mask[cond_rows:] = True
    tags = torch.full((seq_len,), -1, dtype=torch.long)
    tags[text_sl] = 1
    tags[audio_sl] = 2
    tags[img_pos] = 0
    used = video_sl.stop
    return {
        "seq_len": seq_len,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": torch.arange(text_len),
        "update_mask": update_mask,
        "img_position_ids": positions,
        "token_tags": tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
    }


def reference_ref_layout(arguments: Mapping[str, Any]) -> dict[str, Any]:
    text_len = int(arguments["text_len"])
    latent_t, latent_h, latent_w = (
        int(arguments[key]) for key in ("latent_t", "latent_h", "latent_w")
    )
    audio_t = int(arguments["audio_t"])
    blocks = list(arguments["ref_blocks"])
    cursor = text_len
    slices = []
    ref_visual_rows = ref_audio_rows = 0
    for block in blocks:
        item = dict(block)
        kind = str(item["kind"])
        if kind == "image":
            rows = (int(item["latent_h"]) // 2) * (int(item["latent_w"]) // 2)
            item["visual_sl"] = slice(cursor, cursor + rows)
            cursor += rows
            ref_visual_rows += rows
        elif kind == "audio":
            rows = int(item["ref_audio_t"]) * 2
            item["audio_sl"] = slice(cursor, cursor + rows)
            cursor += rows
            ref_audio_rows += rows
        else:
            a_rows = int(item["ref_audio_t"]) * 2
            v_rows = int(item["latent_t"]) * (int(item["latent_h"]) // 2) * (
                int(item["latent_w"]) // 2
            )
            item["audio_sl"] = slice(cursor, cursor + a_rows)
            item["visual_sl"] = slice(cursor + a_rows, cursor + a_rows + v_rows)
            cursor += a_rows + v_rows
            ref_audio_rows += a_rows
            ref_visual_rows += v_rows
        slices.append(item)
    target_audio_sl = slice(cursor, cursor + audio_t * 2)
    target_frame_rows = (latent_h // 2) * (latent_w // 2)
    target_video_sl = slice(
        target_audio_sl.stop, target_audio_sl.stop + latent_t * target_frame_rows
    )
    used = target_video_sl.stop
    seq_len = (used + 63) // 64 * 64
    positions = torch.zeros(seq_len, 3, dtype=torch.float64)
    positions[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
    target_h_grid, target_w_grid = _axis(latent_h, latent_w), _axis(latent_w, latent_h)
    ref_img_parts, ref_audio_parts = [], []
    t_cursor = float(text_len)
    for item in slices:
        kind = str(item["kind"])
        if kind == "image":
            sl = item["visual_sl"]
            rh, rw = int(item["latent_h"]), int(item["latent_w"])
            hh, ww = torch.meshgrid(_axis(rh, rw), _axis(rw, rh), indexing="ij")
            positions[sl, 0] = t_cursor
            positions[sl, 1] = hh.flatten()
            positions[sl, 2] = ww.flatten()
            ref_img_parts.append(torch.arange(sl.start, sl.stop))
            t_cursor += 1.0
        elif kind == "audio":
            sl = item["audio_sl"]
            ref_t = int(item["ref_audio_t"])
            positions[sl, 0] = (t_cursor + torch.arange(ref_t, dtype=torch.float64)).repeat(2)
            if ref_t:
                positions[sl.start : sl.start + ref_t, 2] = target_w_grid[0]
                positions[sl.start + ref_t : sl.stop, 2] = target_w_grid[-1]
            ref_audio_parts.append(torch.arange(sl.start, sl.stop))
            t_cursor += ref_t
        else:
            audio_sl, visual_sl = item["audio_sl"], item["visual_sl"]
            ref_t = int(item["ref_audio_t"])
            vt, vh, vw = (
                int(item[key]) for key in ("latent_t", "latent_h", "latent_w")
            )
            rh_grid, rw_grid = _axis(vh, vw), _axis(vw, vh)
            positions[audio_sl, 0] = (
                t_cursor + torch.arange(ref_t, dtype=torch.float64)
            ).repeat(2)
            if ref_t:
                positions[audio_sl.start : audio_sl.start + ref_t, 2] = rw_grid[0]
                positions[audio_sl.start + ref_t : audio_sl.stop, 2] = rw_grid[-1]
            hh, ww = torch.meshgrid(rh_grid, rw_grid, indexing="ij")
            frame = torch.stack((hh.flatten(), ww.flatten()), dim=-1)
            grid = positions[visual_sl].view(vt, -1, 3)
            grid[:, :, 0] = _t_grid(vt, t_cursor)[:, None]
            grid[:, :, 1:] = frame
            ref_audio_parts.append(torch.arange(audio_sl.start, audio_sl.stop))
            ref_img_parts.append(torch.arange(visual_sl.start, visual_sl.stop))
            video_span = sum(
                FRAME_RESCALE * FRAME_PER_TOKEN[index % 5] for index in range(vt)
            )
            t_cursor += max(float(ref_t), video_span)
    positions[target_audio_sl, 0] = (
        t_cursor + torch.arange(audio_t, dtype=torch.float64)
    ).repeat(2)
    positions[target_audio_sl.start : target_audio_sl.start + audio_t, 2] = target_w_grid[0]
    positions[target_audio_sl.start + audio_t : target_audio_sl.stop, 2] = target_w_grid[-1]
    hh, ww = torch.meshgrid(target_h_grid, target_w_grid, indexing="ij")
    target_frame = torch.stack((hh.flatten(), ww.flatten()), dim=-1)
    target_grid = positions[target_video_sl].view(latent_t, target_frame_rows, 3)
    target_grid[:, :, 0] = _t_grid(latent_t, t_cursor)[:, None]
    target_grid[:, :, 1:] = target_frame
    target_img = torch.arange(target_video_sl.start, target_video_sl.stop)
    target_audio = torch.arange(target_audio_sl.start, target_audio_sl.stop)
    img_pos = torch.cat(ref_img_parts + [target_img])
    audio_pos = torch.cat(ref_audio_parts + [target_audio])
    update = torch.zeros(len(img_pos), dtype=torch.bool)
    update[ref_visual_rows:] = True
    audio_update = torch.zeros(len(audio_pos), dtype=torch.bool)
    audio_update[ref_audio_rows:] = True
    tags = torch.full((seq_len,), -1, dtype=torch.long)
    tags[:text_len] = 1
    tags[audio_pos] = 2
    tags[img_pos] = 0
    return {
        "seq_len": seq_len,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": torch.arange(text_len),
        "update_mask": update,
        "audio_update_mask": audio_update,
        "img_position_ids": positions,
        "token_tags": tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
    }


def assert_nested_close(actual: Any, expected: Any, path: str = "root") -> None:
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual):
            raise AssertionError(f"{path}: expected tensor, got {type(actual).__name__}")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda msg: f"{path}: {msg}")
    elif isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: keys {set(actual)} != {set(expected)}")
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: sequence lengths differ")
        for index, (a_item, e_item) in enumerate(zip(actual, expected, strict=True)):
            assert_nested_close(a_item, e_item, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def _verify_light(stage: str, contract: Mapping[str, Any]) -> None:
    if stage == "presentation":
        assert_nested_close(contract, reference_presentation(contract), stage)
    elif stage == "geometry":
        for index, case in enumerate(contract["cover_crop"]):
            expected = reference_cover_crop(
                case["source_size"], case["target_size"], case["allow_upscale"]
            )
            assert_nested_close(
                {key: case[key] for key in expected}, expected, f"geometry[{index}]"
            )
    elif stage == "noise":
        image = contract["image"]
        audio = contract["audio"]
        torch.testing.assert_close(
            image["noised_rows"],
            reference_image_noise(image["clean_rows"], **image["arguments"]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            audio["noised_rows"],
            reference_audio_noise(audio["clean_rows"], **audio["arguments"]),
            rtol=0,
            atol=0,
        )
    elif stage == "packed":
        for name, item in contract.items():
            expected = (
                reference_ref_layout(item["arguments"])
                if name == "ref2va"
                else reference_standard_layout(item["arguments"])
            )
            assert_nested_close(item["layout"], expected, f"packed.{name}")


def _verify_heavy(stage: str, contract: Mapping[str, Any], candidate: str) -> None:
    path = Path(candidate)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != contract["sha256"]:
        raise AssertionError(f"{stage}: SHA256 mismatch")
    actual = torch.load(path, map_location="cpu", weights_only=True)
    assert_nested_close(actual, contract["payload"], stage)


def _verify_chitu_packed(contract: Mapping[str, Any]) -> None:
    from chitu_diffusion.models.minimax_h3 import build_packed_sequence

    for name in ("t2va", "fl2va"):
        arguments = contract[name]["arguments"]
        kwargs = {
            "text_length": arguments["text_len"],
            "latent_t": arguments["latent_t"],
            "latent_h": arguments["latent_h"],
            "latent_w": arguments["latent_w"],
            "audio_t": arguments["audio_t"],
        }
        if name == "fl2va":
            kwargs.update(
                include_keyframe_cond=True,
                keyframe_frame_indices=arguments["keyframe_frame_indices"],
                frame_count=arguments["frame_count"],
            )
        actual = build_packed_sequence(**kwargs)
        fields = {
            "seq_len": actual.sequence_length,
            "img_pos": actual.img_pos,
            "audio_pos": actual.audio_pos,
            "text_pos": actual.text_pos,
            "update_mask": actual.update_mask,
            "img_position_ids": actual.position_ids,
            "token_tags": actual.token_tags,
            "cu_seqlens": actual.cu_seqlens,
        }
        assert_nested_close(fields, contract[name]["layout"], f"chitu.{name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify MiniMax-H3 golden contracts with independent formulas."
    )
    parser.add_argument("--golden", required=True)
    parser.add_argument("--stages", nargs="+", choices=LIGHT_STAGES + HEAVY_STAGES)
    parser.add_argument("--qwen-artifact")
    parser.add_argument("--vae-artifact")
    parser.add_argument("--steps-artifact")
    parser.add_argument(
        "--check-chitu-packed",
        action="store_true",
        help="delay-import Chitu and compare its standard packed layouts",
    )
    args = parser.parse_args()
    payload = torch.load(args.golden, map_location="cpu", weights_only=True)
    if payload.get("schema") != "minimax_h3_e2e_golden" or payload.get("schema_version") != 1:
        raise ValueError("unsupported MiniMax-H3 golden schema")
    selected = args.stages or payload["stages"]
    candidates = {
        "qwen": args.qwen_artifact,
        "vae": args.vae_artifact,
        "steps": args.steps_artifact,
    }
    for stage in selected:
        if stage not in payload["contracts"]:
            raise ValueError(f"golden does not contain stage {stage!r}")
        if stage in LIGHT_STAGES:
            _verify_light(stage, payload["contracts"][stage])
        else:
            if not candidates[stage]:
                raise ValueError(f"--{stage}-artifact is required to verify {stage!r}")
            _verify_heavy(stage, payload["contracts"][stage], candidates[stage])
        print(f"passed {stage}")
    if args.check_chitu_packed:
        if "packed" not in payload["contracts"]:
            raise ValueError("--check-chitu-packed requires packed golden stage")
        _verify_chitu_packed(payload["contracts"]["packed"])
        print("passed chitu packed parity")


if __name__ == "__main__":
    main()
