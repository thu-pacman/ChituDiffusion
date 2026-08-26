from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch


LIGHT_STAGES = ("presentation", "geometry", "noise", "packed")
HEAVY_STAGES = ("qwen", "vae", "steps")
SCHEMA_VERSION = 1


class ContractTokenizer:
    """Small deterministic tokenizer used to exercise SGLang presentation code."""

    SPECIAL_IDS = {
        "<|vision_start|>": 1_000_001,
        "<|vision_end|>": 1_000_002,
        "<|image_pad|>": 1_000_003,
        "<|video_pad|>": 1_000_004,
    }

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        if add_special_tokens:
            raise AssertionError("H3 presentation must disable special tokens")
        return {"input_ids": [ord(char) for char in text]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.SPECIAL_IDS[token]


def _dump_presentation() -> dict[str, Any]:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
        minimax_h3_multi_image_presentation,
        minimax_h3_ref2va_video_presentation,
        minimax_h3_text_only_ids,
    )

    tokenizer = ContractTokenizer()
    prompt = "  Golden prompt: 海浪\n"
    multi_ids, multi_tags = minimax_h3_multi_image_presentation(
        tokenizer,
        prompt=prompt,
        image_token_counts=[2, 3],
    )
    ref_ids, ref_tags = minimax_h3_ref2va_video_presentation(
        tokenizer,
        prompt=prompt,
        condition_labels=[("audio", 1), ("image", 1), ("video", 1)],
        image_token_count=[2],
        video_block_token_counts=[[3, 2]],
        video_block_timestamps=[[0.25, 1.05]],
    )
    return {
        "prompt": prompt,
        "special_ids": dict(tokenizer.SPECIAL_IDS),
        "text_only_ids": minimax_h3_text_only_ids(tokenizer, prompt),
        "multi_image": {"ids": multi_ids, "token_tags": multi_tags},
        "ref_mixed": {"ids": ref_ids, "token_tags": ref_tags},
    }


def _dump_geometry() -> dict[str, Any]:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.canvas import (
        minimax_h3_cover_crop_plan,
    )

    cases = [
        (1920, 1080, 1280, 720, False),
        (1600, 1200, 1280, 720, False),
        (640, 480, 1280, 720, True),
        (1001, 777, 768, 512, False),
    ]
    plans = []
    for source_w, source_h, target_w, target_h, allow_upscale in cases:
        plan = minimax_h3_cover_crop_plan(
            source_width=source_w,
            source_height=source_h,
            target_width=target_w,
            target_height=target_h,
            allow_upscale=allow_upscale,
        )
        plans.append(
            {
                "source_size": (source_w, source_h),
                "target_size": (target_w, target_h),
                "allow_upscale": allow_upscale,
                **plan,
            }
        )
    return {"cover_crop": plans}


def _dump_noise(seed: int) -> dict[str, Any]:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.condition_noise import (
        minimax_h3_audio_cond_noise_aug_rows,
        minimax_h3_imgvid_cond_noise_aug_rows,
    )

    image_shapes = [(1, 4, 6), (2, 2, 4)]
    image_rows = sum(t * (h // 2) * (w // 2) for t, h, w in image_shapes)
    clean_image = torch.linspace(-1.0, 1.0, image_rows * 96).reshape(image_rows, 96)
    audio_lengths = [2, 3]
    clean_audio = torch.linspace(-0.75, 0.5, 2 * sum(audio_lengths) * 32).reshape(
        2 * sum(audio_lengths), 32
    )
    image_args = {
        "condition_shapes": image_shapes,
        "target_latent_t": 7,
        "imgvid_cond_num_frames": 2,
        "seed": seed,
        "noise_aug": 0.625,
    }
    audio_args = {
        "condition_audio_t": audio_lengths,
        "seed": seed,
        "noise_aug": 0.375,
    }
    return {
        "image": {
            "clean_rows": clean_image,
            "arguments": image_args,
            "noised_rows": minimax_h3_imgvid_cond_noise_aug_rows(
                clean_image, **image_args
            ),
        },
        "audio": {
            "clean_rows": clean_audio,
            "arguments": audio_args,
            "noised_rows": minimax_h3_audio_cond_noise_aug_rows(
                clean_audio, **audio_args
            ),
        },
    }


def _tensor_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if torch.is_tensor(item) or isinstance(item, int)
    }


def _dump_packed() -> dict[str, Any]:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
        minimax_h3_packed_sequence_ref2va_blocks,
    )

    t2va_args = dict(
        text_len=7,
        latent_t=6,
        latent_h=4,
        latent_w=6,
        audio_t=5,
        include_keyframe_cond=False,
    )
    fl2va_args = dict(
        text_len=11,
        latent_t=16,
        latent_h=4,
        latent_w=6,
        audio_t=7,
        include_keyframe_cond=True,
        keyframe_frame_indices=[0, -1],
        frame_count=49,
    )
    ref_args = dict(
        text_len=5,
        latent_t=3,
        latent_h=4,
        latent_w=6,
        audio_t=4,
        ref_blocks=[
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
    )
    return {
        "t2va": {
            "arguments": t2va_args,
            "layout": _tensor_fields(minimax_h3_packed_sequence(**t2va_args)),
        },
        "fl2va": {
            "arguments": fl2va_args,
            "layout": _tensor_fields(minimax_h3_packed_sequence(**fl2va_args)),
        },
        "ref2va": {
            "arguments": ref_args,
            "layout": _tensor_fields(
                minimax_h3_packed_sequence_ref2va_blocks(**ref_args)
            ),
        },
    }


def _load_heavy_artifact(path: str, stage: str) -> dict[str, Any]:
    artifact_path = Path(path)
    raw = artifact_path.read_bytes()
    value = torch.load(artifact_path, map_location="cpu", weights_only=True)
    return {
        "stage": stage,
        "source_name": artifact_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": value,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump staged MiniMax-H3 end-to-end contracts from installed SGLang. "
            "The default stages are CPU-only and do not load model weights."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=LIGHT_STAGES + HEAVY_STAGES,
        default=list(LIGHT_STAGES),
    )
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--qwen-artifact",
        help="torch-saved Qwen embedding artifact; required only for stage qwen",
    )
    parser.add_argument(
        "--vae-artifact",
        help="torch-saved VAE latent artifact; required only for stage vae",
    )
    parser.add_argument(
        "--steps-artifact",
        help="torch-saved per-step latent artifact; required only for stage steps",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stages = list(dict.fromkeys(args.stages))
    heavy_paths = {
        "qwen": args.qwen_artifact,
        "vae": args.vae_artifact,
        "steps": args.steps_artifact,
    }
    for stage in HEAVY_STAGES:
        if stage in stages and not heavy_paths[stage]:
            raise ValueError(f"--{stage}-artifact is required for stage {stage!r}")

    builders = {
        "presentation": _dump_presentation,
        "geometry": _dump_geometry,
        "noise": lambda: _dump_noise(args.seed),
        "packed": _dump_packed,
    }
    contracts: dict[str, Any] = {}
    for stage in stages:
        contracts[stage] = (
            builders[stage]()
            if stage in builders
            else _load_heavy_artifact(heavy_paths[stage], stage)
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "minimax_h3_e2e_golden",
            "schema_version": SCHEMA_VERSION,
            "seed": args.seed,
            "stages": stages,
            "contracts": contracts,
        },
        output,
    )
    print(f"wrote {output} ({', '.join(stages)})")


if __name__ == "__main__":
    main()
