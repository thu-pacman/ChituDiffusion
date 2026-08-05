#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from chitu_diffusion.evaluation.utils.reference_metrics import load_video_frames


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {str(row.get("case") or ""): row for row in csv.DictReader(f) if row.get("case")}


def fmt_metric(value: str | None, precision: int = 3) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except ValueError:
        return value


def short_prompt(prompt: str, limit: int = 44) -> str:
    prompt = " ".join(prompt.split())
    return prompt if len(prompt) <= limit else prompt[: limit - 3] + "..."


def short_case(case: str, limit: int = 22) -> str:
    case = " ".join(case.replace("_", " ").split())
    return case if len(case) <= limit else case[: limit - 3] + "..."


FAMILY_STYLES = {
    "origin": {"label": "Origin", "color": "#222222"},
    "torch_sdpa": {"label": "Torch SDPA", "color": "#64748b"},
    "torch": {"label": "Torch", "color": "#94a3b8"},
    "pab": {"label": "PAB", "color": "#dc2626"},
    "blockdance": {"label": "BlockDance", "color": "#2563eb"},
    "cubic": {"label": "Cubic", "color": "#059669"},
    "meancache": {"label": "MeanCache", "color": "#6d28d9"},
    "teacache": {"label": "TeaCache", "color": "#a855f7"},
    "taylorseer": {"label": "TaylorSeer", "color": "#d97706"},
    "sage": {"label": "SageAttention", "color": "#0f766e"},
    "flashinfer": {"label": "FlashInfer", "color": "#0891b2"},
    "sparge": {"label": "SpargeAttn", "color": "#9333ea"},
    "other": {"label": "Other", "color": "#6b7280"},
}


def family_for_case(case: str) -> str:
    for prefix in ("qwen_", "flux1_", "flux2_"):
        if case.startswith(prefix):
            case = case[len(prefix) :]
            break
    if case == "origin_flash":
        return "origin"
    if case.startswith("torch_sdpa_math"):
        return "torch"
    for family in ("torch_sdpa", "pab", "blockdance", "cubic", "meancache", "teacache", "taylorseer", "sage", "flashinfer", "sparge"):
        if case.startswith(family):
            return family
    return "other"


def display_case(case: str, is_qwen: bool = False) -> str:
    if case == "torch_sdpa":
        return "Flash Attention" if is_qwen else "Torch SDPA"
    labels = {
        "torch_sdpa_math": "Torch",
        "origin_flash": "origin_flash",
        "flashinfer": "flashinfer",
        "qwen_pab50_cfp2": "pab50",
        "qwen_blockdance50_cfp2": "bd50",
        "qwen_cubic15_50_cfp2": "cubic1.5",
        "qwen_cubic30_w9c1_tau10_50_cfp2": "cubic3.0",
        "qwen_meancache25_50_cfp2": "mc25",
        "qwen_meancache17_50_cfp2": "mc17",
        "qwen_meancache10_50_cfp2": "mc10",
    }
    if case in labels:
        return labels[case]
    match = re.match(r"(?:qwen|flux1|flux2)_pab_s(\d+)c(\d+)_", case)
    if match:
        return f"pab{match.group(1)}/{match.group(2)}"
    match = re.match(r"(?:qwen|flux1|flux2)_blockdance_g(\d+)_", case)
    if match:
        return f"bd-g{match.group(1)}"
    match = re.match(r"(?:qwen|flux1|flux2)_cubic(\d+)_", case)
    if match:
        value = int(match.group(1)) / 10.0
        return f"cubic{value:g}"
    match = re.match(r"(?:qwen|flux1|flux2)_meancache(\d+)_", case)
    if match:
        return f"mc{match.group(1)}"
    label = case
    for prefix in ("qwen_", "teacache_", "meancache_", "blockdance_", "taylorseer_", "cubic_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    return label


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = " ".join(text.split()).split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        box = draw.textbbox((0, 0), candidate, font=font)
        if box[2] - box[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = image.resize((round(src_w * scale), round(src_h * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - target_w) // 2)
    top = max(0, (resized.height - target_h) // 2)
    return resized.crop((left, top, left + target_w, top + target_h))


def load_visual(path: str, frame_index: int = -1) -> Image.Image:
    source = Path(path)
    if source.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return Image.open(source)
    frames = load_video_frames(str(source), max_frames=-1)
    if len(frames) == 0:
        raise ValueError(f"failed to read frames from {source}")
    idx = len(frames) // 2 if frame_index < 0 else min(max(frame_index, 0), len(frames) - 1)
    return Image.fromarray(frames[idx].clip(0, 255).astype("uint8"))


def draw_rounded_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    text_fill: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(xy, radius=10, fill=fill)
    box = draw.textbbox((0, 0), text, font=font)
    text_w = box[2] - box[0]
    text_h = box[3] - box[1]
    x0, y0, x1, y1 = xy
    draw.text((x0 + (x1 - x0 - text_w) // 2, y0 + (y1 - y0 - text_h) // 2 - 1), text, fill=text_fill, font=font)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="visuals/contact_sheet.png")
    parser.add_argument("--title", default="Flux Attention Backend Visual Check")
    parser.add_argument("--experiment-id", default="", help="Experiment id; controls model-specific case labels.")
    parser.add_argument("--cases", nargs="*", help="Optional explicit case order.")
    parser.add_argument("--prompt-id", help="Optional prompt id filter; keeps only rows whose task id contains this id.")
    parser.add_argument("--columns", type=int, default=0, help="Number of case columns before wrapping.")
    parser.add_argument("--group-by-family", action="store_true", help="Draw family-colored group boxes around cases.")
    parser.add_argument("--frame-index", type=int, default=-1, help="Video frame index to show; -1 uses the middle frame.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    is_qwen = str(args.experiment_id or "").startswith("qwen")
    if is_qwen:
        FAMILY_STYLES["torch_sdpa"]["label"] = "Flash Attention"
    rows = read_json(experiment_dir / "quality" / "quality_rows.json")
    if args.prompt_id:
        prompt_id = str(args.prompt_id)
        rows = [row for row in rows if prompt_id in str(row.get("task_id") or "")]
    summary_by_case = read_summary(experiment_dir / "summary.csv")
    if args.cases:
        cases = args.cases
    else:
        seen_cases = []
        for row in rows:
            case = str(row.get("case") or "")
            if case and case not in seen_cases:
                seen_cases.append(case)
        preferred = ["origin_flash", "torch_sdpa", "torch_sdpa_math", "flashinfer", "sage", "sparge", "baseline_1gpu"]
        cases = [case for case in preferred if case in seen_cases] + [case for case in seen_cases if case not in preferred]
    prompts = []
    for row in rows:
        prompt = str(row.get("prompt") or "")
        if prompt and prompt not in prompts:
            prompts.append(prompt)

    by_key = {(str(row.get("prompt") or ""), str(row.get("case")), int(row.get("seed"))): row for row in rows}
    columns = args.columns if args.columns > 0 else len(cases)
    columns = max(1, min(columns, len(cases)))
    case_rows = [cases[idx : idx + columns] for idx in range(0, len(cases), columns)]
    tile_w = 260 if columns > 4 else 300 if columns == 4 else 320
    tile_h = tile_w
    gap = 14 if columns > 4 else 18
    margin = 34
    title_h = 56
    header_h = 68 if args.group_by_family else 44
    row_label_h = 40
    group_gap = 24
    single_prompt_header = len(prompts) == 1
    prompt_font = load_font(15)
    prompt_lines = wrap_text(prompts[0], prompt_font, columns * tile_w + (columns - 1) * gap, ImageDraw.Draw(Image.new("RGB", (1, 1)))) if single_prompt_header else []
    prompt_h = len(prompt_lines) * 21 + 16 if single_prompt_header else 0
    prompt_rows_h = tile_h if single_prompt_header else len(prompts) * (row_label_h + tile_h + gap) - gap
    group_h = header_h + prompt_rows_h
    width = margin * 2 + columns * tile_w + (columns - 1) * gap
    height = margin * 2 + title_h + prompt_h + len(case_rows) * group_h + (len(case_rows) - 1) * group_gap
    sheet = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(26, bold=True)
    family_font = load_font(16, bold=True)
    case_font = load_font(17, bold=True)
    metric_font = load_font(13)
    small_font = load_font(12)

    draw.text((margin, margin - 4), args.title, fill="#111827", font=title_font)
    if single_prompt_header:
        prompt_y = margin + title_h - 4
        for line in prompt_lines:
            draw.text((margin, prompt_y), line, fill="#334155", font=prompt_font)
            prompt_y += 21

    for case_row_idx, row_cases in enumerate(case_rows):
        group_y = margin + title_h + prompt_h + case_row_idx * (group_h + group_gap)
        if args.group_by_family:
            spans: list[tuple[str, int, int]] = []
            start = 0
            current_family = family_for_case(row_cases[0])
            for idx, case in enumerate(row_cases[1:], start=1):
                family = family_for_case(case)
                if family != current_family:
                    spans.append((current_family, start, idx - 1))
                    current_family = family
                    start = idx
            spans.append((current_family, start, len(row_cases) - 1))
            for family, start_col, end_col in spans:
                color = hex_to_rgb(FAMILY_STYLES[family]["color"])
                x0 = margin + start_col * (tile_w + gap) - 8
                x1 = margin + end_col * (tile_w + gap) + tile_w + 8
                y0 = group_y - 6
                y1 = group_y + group_h + 6
                draw.rounded_rectangle((x0, y0, x1, y1), radius=18, outline=color, width=5)
                label = FAMILY_STYLES[family]["label"]
                label_box = draw.textbbox((0, 0), label, font=family_font)
                label_w = min(x1 - x0 - 20, label_box[2] - label_box[0] + 28)
                draw.rounded_rectangle((x0 + 10, y0 - 2, x0 + 10 + label_w, y0 + 27), radius=9, fill=color)
                draw.text((x0 + 24, y0 + 3), label, fill="#ffffff", font=family_font)

        for col, case in enumerate(row_cases):
            x = margin + col * (tile_w + gap)
            y = group_y + 30 if args.group_by_family else group_y
            family = family_for_case(case)
            color = hex_to_rgb(FAMILY_STYLES[family]["color"])
            draw_rounded_label(draw, (x, y, x + tile_w, y + 30), short_case(display_case(case, is_qwen)), case_font, color, (255, 255, 255))

        prompt_base_y = group_y + header_h
        for row_idx, prompt in enumerate(prompts):
            y = prompt_base_y if single_prompt_header else prompt_base_y + row_idx * (row_label_h + tile_h + gap)
            if not single_prompt_header:
                draw.text((margin, y + 6), short_prompt(prompt, 96), fill="#334155", font=prompt_font)
            for col, case in enumerate(row_cases):
                item = by_key.get((prompt, case, args.seed))
                x = margin + col * (tile_w + gap)
                image_y = y if single_prompt_header else y + row_label_h
                family = family_for_case(case)
                color = hex_to_rgb(FAMILY_STYLES[family]["color"])
                if item is None:
                    draw.rounded_rectangle((x, image_y, x + tile_w, image_y + tile_h), radius=8, outline="#cbd5e1", width=2)
                    draw.text((x + 22, image_y + 22), "missing", fill="#7f1d1d", font=case_font)
                    continue
                image = fit_cover(load_visual(item["generated"], args.frame_index), (tile_w, tile_h))
                sheet.paste(image, (x, image_y))
                draw.rounded_rectangle((x, image_y, x + tile_w, image_y + tile_h), radius=8, outline=color, width=3)
                summary = summary_by_case.get(case, {})
                speedup = fmt_metric(summary.get("speedup_vs_origin"), 2)
                lpips = fmt_metric(summary.get("one_minus_lpips_mean"), 3)
                label = f"{speedup}x  |  1-LPIPS {lpips}"
                label_box = draw.textbbox((0, 0), label, font=metric_font)
                label_w = min(tile_w - 18, label_box[2] - label_box[0] + 18)
                draw.rounded_rectangle(
                    (x + 9, image_y + tile_h - 35, x + 9 + label_w, image_y + tile_h - 10),
                    radius=8,
                    fill=(15, 23, 42),
                )
                draw.text((x + 18, image_y + tile_h - 31), label, fill="#f8fafc", font=metric_font)

    draw.text(
        (margin, height - margin + 10),
        "Cases are grouped by method. Images use the same seed for visual comparison.",
        fill="#64748b",
        font=small_font,
    )

    out_path = experiment_dir / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
