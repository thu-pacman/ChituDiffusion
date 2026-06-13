#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def short_prompt(prompt: str, limit: int = 44) -> str:
    prompt = " ".join(prompt.split())
    return prompt if len(prompt) <= limit else prompt[: limit - 3] + "..."


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
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    rows = read_json(experiment_dir / "quality" / "quality_rows.json")
    cases = ["origin_flash", "torch_sdpa_math", "sage", "sparge"]
    prompts = []
    for row in rows:
        prompt = str(row.get("prompt") or "")
        if prompt and prompt not in prompts:
            prompts.append(prompt)

    by_key = {(str(row.get("prompt") or ""), str(row.get("case")), int(row.get("seed"))): row for row in rows}
    tile_w, tile_h = 320, 320
    gap = 18
    margin = 34
    title_h = 56
    header_h = 44
    row_label_h = 40
    width = margin * 2 + len(cases) * tile_w + (len(cases) - 1) * gap
    height = margin * 2 + title_h + header_h + len(prompts) * (row_label_h + tile_h + gap) - gap
    sheet = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(26, bold=True)
    case_font = load_font(17, bold=True)
    prompt_font = load_font(15)
    metric_font = load_font(13)
    small_font = load_font(12)

    draw.text((margin, margin - 4), args.title, fill="#111827", font=title_font)

    for col, case in enumerate(cases):
        x = margin + col * (tile_w + gap)
        y = margin + title_h
        draw_rounded_label(draw, (x, y, x + tile_w, y + 30), case, case_font, (229, 231, 235), (17, 24, 39))

    for row_idx, prompt in enumerate(prompts):
        y = margin + title_h + header_h + row_idx * (row_label_h + tile_h + gap)
        draw.text((margin, y + 6), short_prompt(prompt, 96), fill="#334155", font=prompt_font)
        for col, case in enumerate(cases):
            item = by_key.get((prompt, case, args.seed))
            x = margin + col * (tile_w + gap)
            image_y = y + row_label_h
            if item is None:
                draw.rounded_rectangle((x, image_y, x + tile_w, image_y + tile_h), radius=8, outline="#cbd5e1", width=2)
                draw.text((x + 22, image_y + 22), "missing", fill="#7f1d1d", font=case_font)
                continue
            image = fit_cover(Image.open(item["generated"]), (tile_w, tile_h))
            sheet.paste(image, (x, image_y))
            draw.rounded_rectangle((x, image_y, x + tile_w, image_y + tile_h), radius=8, outline="#e2e8f0", width=2)
            hps = item.get("hpsv3_score")
            hps_label = "HPSv3 -" if hps is None else f"HPSv3 {float(hps):.2f}"
            label = f"seed {args.seed}  |  {hps_label}"
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
        "Rows: prompts. Columns: backend cases. Images use the same seed for visual comparison.",
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
