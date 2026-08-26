#!/usr/bin/env python3
"""Generate and optionally run the MiniMax-H3 TP2 x CP{1,2,4} cost matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chitu_diffusion.models.minimax_h3.fl2va_geometry import (  # noqa: E402
    FL2VATimeRequest,
    resolve_fl2va_plan,
)
from chitu_diffusion.models.minimax_h3.qwen3vl_processor import (  # noqa: E402
    MiniMaxH3Qwen3VLProcessor,
    build_h3_presentation,
)
from script.simulate_h3_epe_trace import _record_startup_warmup  # noqa: E402


FULL_SHAPES = (
    (384, 768, 2.0),
    (512, 768, 3.0),
    (768, 768, 5.0),
    (768, 1024, 8.0),
    (768, 1280, 8.0),
)
PROMPT_TARGETS = (32, 64, 128, 256)
BOUNDARY_RESIDUES = (0, 1, 31, 63)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("examples/stage-minimax-h3.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/chitu-api/h3-cost-profile-full"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=21_600.0)
    parser.add_argument("--measure-steps", type=int, default=15)
    parser.add_argument("--burnin-steps", type=int, default=3)
    return parser


def _write_keyframe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (128, 128))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (x * 2 % 256, y * 2 % 256, (x + y) % 256)
    image.save(path)


def _presentation_length(
    processor: MiniMaxH3Qwen3VLProcessor,
    prompt: str,
    image_token_counts: tuple[int, ...],
) -> int:
    input_ids, _ = build_h3_presentation(
        processor.tokenizer,
        prompt,
        image_token_counts=image_token_counts,
    )
    return int(input_ids.numel())


def _prompt_candidates(
    processor: MiniMaxH3Qwen3VLProcessor,
    *,
    image_token_counts: tuple[int, ...],
    fixed_tokens: int,
    smoke: bool,
) -> list[tuple[str, int, str]]:
    candidates = []
    for repeats in range(1, 513):
        prompt = ("video " * repeats).strip()
        text_tokens = _presentation_length(processor, prompt, image_token_counts)
        candidates.append((prompt, text_tokens))

    selected: list[tuple[str, int, str]] = []
    targets = (32, 128) if smoke else PROMPT_TARGETS
    for target in targets:
        prompt, tokens = min(candidates, key=lambda item: abs(item[1] - target))
        selected.append((prompt, tokens, f"text-{target}"))
    if not smoke:
        for residue in BOUNDARY_RESIDUES:
            prompt, tokens = min(
                candidates,
                key=lambda item: (
                    min(
                        ((fixed_tokens + item[1] - residue) % 64),
                        ((residue - fixed_tokens - item[1]) % 64),
                    ),
                    abs(item[1] - 128),
                ),
            )
            selected.append((prompt, tokens, f"residue-{residue}"))

    unique: dict[int, tuple[str, int, str]] = {}
    for prompt, tokens, label in selected:
        unique.setdefault(tokens, (prompt, tokens, label))
    return list(unique.values())


def _image_token_count(
    processor: MiniMaxH3Qwen3VLProcessor,
    image: Image.Image,
) -> int:
    if processor.image_processor is None:
        raise RuntimeError("H3 image processor is required for keyframe profiles")
    processed = processor.image_processor(images=[image], return_tensors="pt")
    grid = processed["image_grid_thw"][0]
    return int(grid.prod()) // processor.spatial_merge_size**2


def _profiles(
    raw: dict[str, Any],
    *,
    keyframe_path: Path,
    smoke: bool,
) -> list[dict[str, Any]]:
    factory = raw["factory_args"]
    processor_path = (
        factory.get("processor_path")
        or factory.get("tokenizer_path")
        or factory["text_encoder_path"]
    )
    processor = MiniMaxH3Qwen3VLProcessor.from_pretrained(processor_path)
    base_image = Image.open(keyframe_path).convert("RGB")
    shapes = (FULL_SHAPES[0],) if smoke else FULL_SHAPES
    keyframe_counts = (0, 2) if smoke else (0, 1, 2)
    output = []
    for height, width, duration_s in shapes:
        plan = resolve_fl2va_plan(
            FL2VATimeRequest(
                duration_seconds=duration_s,
                width=width,
                height=height,
                fps=24,
            )
        )
        resized = base_image.resize((plan.canvas.width, plan.canvas.height))
        image_tokens = _image_token_count(processor, resized)
        frame_rows = (plan.latent_height // 2) * (plan.latent_width // 2)
        audio_tokens = plan.audio_latent_t * 2
        video_tokens = plan.video_latent_t * frame_rows
        for keyframes in keyframe_counts:
            presentation_image_counts = (image_tokens,) * keyframes
            fixed_tokens = keyframes * frame_rows + audio_tokens + video_tokens
            for prompt, text_tokens, label in _prompt_candidates(
                processor,
                image_token_counts=presentation_image_counts,
                fixed_tokens=fixed_tokens,
                smoke=smoke,
            ):
                used_tokens = fixed_tokens + text_tokens
                profile_id = (
                    f"{height}x{width}-{duration_s:g}s-kf{keyframes}-"
                    f"{label}-t{text_tokens}-r{used_tokens % 64}"
                )
                output.append(
                    {
                        "profile_id": profile_id,
                        "duration_s": duration_s,
                        "fps": 24,
                        "height": height,
                        "width": width,
                        "output_type": "latent",
                        "prompt": prompt,
                        "first_frame_path": (
                            str(keyframe_path.resolve()) if keyframes else None
                        ),
                        "last_frame_path": (
                            str(keyframe_path.resolve()) if keyframes == 2 else None
                        ),
                    }
                )
    return output


def main() -> None:
    args = _parser().parse_args()
    if min(args.measure_steps, args.burnin_steps + 1) <= 0:
        raise ValueError("measure and burn-in step counts must be valid")
    base_config = args.base_config.resolve()
    raw = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    keyframe_path = output_root / "profile-keyframe.png"
    _write_keyframe(keyframe_path)
    profiles = _profiles(raw, keyframe_path=keyframe_path, smoke=args.smoke)

    raw["name"] = "minimax_h3_cost_profile_smoke" if args.smoke else "minimax_h3_cost_profile"
    raw["output_root"] = str(output_root)
    raw["parallelism"]["sp"] = 4
    raw["parallelism"]["tp"] = 2
    pool = raw["parallelism"]["chitu_pool"]
    pool["policy"] = "elastic"
    pool["allowed_lane_widths"] = [1, 2, 4]
    pool["warmup_steps"] = args.measure_steps
    pool["warmup_resolutions"] = [128]
    factory = raw["factory_args"]
    factory["ulysses_degree"] = 4
    factory["warmup_burnin_steps"] = args.burnin_steps
    factory["warmup_media_profiles"] = profiles
    config_path = output_root / "stage_config.generated.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    (output_root / "profile_manifest.json").write_text(
        json.dumps(
            {
                "mode": "smoke" if args.smoke else "full",
                "profile_count": len(profiles),
                "measure_steps": args.measure_steps,
                "burnin_steps": args.burnin_steps,
                "profiles": profiles,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"generated {len(profiles)} profiles at {config_path}", flush=True)
    if not args.execute:
        return
    report = _record_startup_warmup(
        config_path,
        service_python=Path(sys.executable),
        timeout_s=args.timeout_s,
        log_path=output_root / "profile.log",
    )
    print(f"profile report: {report}", flush=True)


if __name__ == "__main__":
    main()
