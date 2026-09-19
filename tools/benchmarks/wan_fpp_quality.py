"""Paired PSNR/optional LPIPS on lossless --save-frames outputs in [0, 1]."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_frames(path):
    with np.load(path, allow_pickle=False) as data:
        frames = data["frames"].astype(np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3 or min(frames.shape) < 1:
        raise ValueError(f"{path}: expected [frames, height, width, 3]")
    if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
        raise ValueError(f"{path}: expected finite pixels in [0, 1]")
    return frames


def paired_psnr(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(
            "paired videos must have identical shapes; no silent resizing or truncation"
        )
    mse = float(np.mean((reference.astype(np.float64) - candidate) ** 2))
    return {
        "mse": mse,
        "psnr_db": None if mse == 0 else float(-10 * np.log10(mse)),
        "identical": mse == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--lpips-checkpoint",
        type=Path,
        help="Optional complete LPIPS AlexNet state_dict, including backbone; requires lpips installed. No downloads.",
    )
    args = parser.parse_args()
    reference = load_frames(args.reference)
    metric = None
    if args.lpips_checkpoint:
        import lpips
        import torch

        metric = lpips.LPIPS(
            net="alex", pnet_rand=True, pretrained=False, verbose=False
        ).eval()
        metric.load_state_dict(
            torch.load(args.lpips_checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )
    rows = []
    for path in args.candidate:
        candidate = load_frames(path)
        result = paired_psnr(reference, candidate)
        if metric is not None:
            values = []
            with torch.inference_mode():
                for left, right in zip(reference, candidate, strict=True):
                    a = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                    b = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                    values.append(metric(a, b).item())
            result["lpips"] = float(np.mean(values))
        rows.append({"candidate": str(path), **result})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "reference": str(args.reference),
                "pixel_range": [0, 1],
                "psnr_infinite_when_identical": True,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
