from pathlib import Path

from PIL import Image

from chitu_diffusers.benchmarks.make_image_contact_sheet import make_contact_sheet


def test_make_contact_sheet_places_each_request_and_strategy(tmp_path: Path) -> None:
    runs = []
    for label, color in (("static_dp", "red"), ("elastic", "blue")):
        directory = tmp_path / label
        directory.mkdir()
        for request_id in ("req0000", "req0001"):
            Image.new("RGB", (64, 64), color=color).save(
                directory / f"{request_id}.png"
            )
        runs.append((label, directory))

    output = make_contact_sheet(runs, tmp_path / "sheet.png", thumbnail_px=64)

    with Image.open(output) as sheet:
        assert sheet.format == "PNG"
        assert sheet.width > 2 * 64
        assert sheet.height > 2 * 64
