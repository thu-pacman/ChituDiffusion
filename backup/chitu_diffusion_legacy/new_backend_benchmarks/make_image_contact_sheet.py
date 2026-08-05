from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use LABEL=IMAGE_DIR")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("run must use LABEL=IMAGE_DIR")
    return label, Path(raw_path).expanduser().resolve()


def make_contact_sheet(
    runs: list[tuple[str, Path]],
    output: Path,
    *,
    thumbnail_px: int = 224,
) -> Path:
    if not runs:
        raise ValueError("at least one run is required")
    if thumbnail_px < 32:
        raise ValueError("thumbnail_px must be at least 32")
    request_ids = sorted(
        {path.stem for _, directory in runs for path in directory.glob("req*.png")}
    )
    if not request_ids:
        raise ValueError("run image directories contain no req*.png files")

    margin = 12
    label_width = 116
    header_height = 34
    cell_width = thumbnail_px + margin * 2
    cell_height = thumbnail_px + margin * 2
    canvas = Image.new(
        "RGB",
        (
            label_width + len(runs) * cell_width,
            header_height + len(request_ids) * cell_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, (label, _) in enumerate(runs):
        x = label_width + column * cell_width + margin
        draw.text((x, 10), label, fill="#101828", font=font)

    for row, request_id in enumerate(request_ids):
        y = header_height + row * cell_height
        draw.text((margin, y + margin), request_id, fill="#344054", font=font)
        for column, (_, directory) in enumerate(runs):
            image_path = directory / f"{request_id}.png"
            x = label_width + column * cell_width + margin
            if not image_path.is_file():
                draw.rectangle(
                    (x, y + margin, x + thumbnail_px, y + margin + thumbnail_px),
                    outline="#d92d20",
                    width=2,
                )
                draw.text((x + 8, y + margin + 8), "missing", fill="#d92d20")
                continue
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                original_size = image.size
                image.thumbnail(
                    (thumbnail_px, thumbnail_px),
                    Image.Resampling.LANCZOS,
                )
            left = x + (thumbnail_px - image.width) // 2
            top = y + margin + (thumbnail_px - image.height) // 2
            canvas.paste(image, (left, top))
            draw.rectangle(
                (x, y + margin, x + thumbnail_px, y + margin + thumbnail_px),
                outline="#98a2b3",
                width=1,
            )
            draw.text(
                (x + 4, y + margin + thumbnail_px - 14),
                f"{original_size[0]}x{original_size[1]}",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
                font=font,
            )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a request-by-strategy image contact sheet."
    )
    parser.add_argument("--run", action="append", type=_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumbnail-px", type=int, default=224)
    args = parser.parse_args()
    print(make_contact_sheet(args.run, args.output, thumbnail_px=args.thumbnail_px))


if __name__ == "__main__":
    main()
