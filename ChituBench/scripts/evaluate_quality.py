#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from chitu_diffusion.evaluation.utils.reference_metrics import align_video_pair, load_video_frames


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_json(path: Path) -> Any | None:
    return read_json(path) if path.exists() else None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]


def is_warmup_request(request: dict[str, Any]) -> bool:
    task_id = str(request.get("request_id") or "")
    role = str(((request.get("params") or {}).get("role")) or "").lower()
    return role == "warmup" or task_id.startswith("warmup_")


def sidecar_for_image(image_path: Path) -> dict[str, Any]:
    payload = maybe_json(image_path.with_suffix(".json"))
    return payload if isinstance(payload, dict) else {}


def output_for_task(run_dir: Path, task_id: str) -> Path | None:
    task_dir = run_dir / "results" / task_id
    if not task_dir.exists():
        return None
    for sidecar in sorted(task_dir.glob("*.json")):
        payload = maybe_json(sidecar)
        if isinstance(payload, dict) and payload.get("filename"):
            output_path = task_dir / str(payload["filename"])
            if output_path.exists():
                return output_path
    matches = sorted(task_dir.glob("*.png"))
    if matches:
        return matches[0]
    matches = sorted(task_dir.glob("*.mp4"))
    return matches[0] if matches else None


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def representative_rgb(path: Path, frame_index: int = -1) -> np.ndarray:
    if not is_video_path(path):
        return load_rgb(path)
    frames = load_video_frames(str(path), max_frames=-1)
    if len(frames) == 0:
        raise ValueError(f"failed to read frames from {path}")
    idx = len(frames) // 2 if frame_index < 0 else min(max(frame_index, 0), len(frames) - 1)
    return frames[idx].astype(np.float32)


def representative_image_for_hpsv3(path: Path, output_dir: Path, frame_index: int = -1) -> Path:
    if not is_video_path(path):
        return path
    cache_dir = output_dir / "hpsv3_frames"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{path.stem}_frame{'mid' if frame_index < 0 else frame_index}.png"
    if not cache_path.exists():
        frame = representative_rgb(path, frame_index=frame_index).clip(0, 255).astype("uint8")
        Image.fromarray(frame).save(cache_path)
    return cache_path


def aligned_arrays(gen_path: Path, ref_path: Path, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    if is_video_path(gen_path) or is_video_path(ref_path):
        gen_frames = load_video_frames(str(gen_path), max_frames=-1)
        ref_frames = load_video_frames(str(ref_path), max_frames=-1)
        gen_aligned, ref_aligned = align_video_pair(gen_frames, ref_frames, max_frames=max_frames)
        return gen_aligned, ref_aligned
    return load_rgb(gen_path), load_rgb(ref_path)


def compute_lpips_rows(pairs: list[tuple[Path, Path]]) -> list[float | None]:
    try:
        import lpips
    except ImportError:
        return [None for _ in pairs]
    if not pairs:
        return []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_model = lpips.LPIPS(net="alex").to(device=device, dtype=torch.float32)
    metric_model.eval()
    scores = []
    with torch.no_grad():
        for gen_path, ref_path in pairs:
            gen = torch.from_numpy(representative_rgb(gen_path)).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            ref = torch.from_numpy(representative_rgb(ref_path)).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            gen = gen / 127.5 - 1.0
            ref = ref / 127.5 - 1.0
            if gen.shape[-2:] != (224, 224):
                gen = F.interpolate(gen, size=(224, 224), mode="bilinear", align_corners=False)
                ref = F.interpolate(ref, size=(224, 224), mode="bilinear", align_corners=False)
            scores.append(float(metric_model(gen, ref).item()))
    return scores


def build_reference_index(origin_dir: Path) -> dict[tuple[str, int], Path]:
    payload = maybe_json(origin_dir / "request_params.json") or {}
    index: dict[tuple[str, int], Path] = {}
    for request in payload.get("requests", []):
        if is_warmup_request(request):
            continue
        task_id = str(request.get("request_id") or "")
        params = request.get("params") or {}
        prompt = str(params.get("prompt") or "")
        seed = params.get("seed")
        output_path = output_for_task(origin_dir, task_id)
        if seed is not None and output_path is not None:
            index[(prompt, int(seed))] = output_path
    return index


def ensure_hpsv3_compat() -> None:
    import transformers.image_utils as image_utils
    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = Any


def compute_hpsv3(rows: list[dict[str, Any]], output_dir: Path, config_path: Path | None, checkpoint_path: Path | None, batch_size: int, frame_index: int) -> dict[str, float]:
    if not rows:
        return {}
    ensure_hpsv3_compat()
    from hpsv3 import HPSv3RewardInferencer

    if not torch.cuda.is_available():
        write_json(output_dir / "hpsv3_error.json", {"error": "cuda is required by HPSv3"})
        return {}
    if checkpoint_path and checkpoint_path.suffix != ".safetensors":
        link = output_dir / "HPSv3.safetensors"
        if link.exists() or link.is_symlink():
            if link.resolve() != checkpoint_path.resolve():
                link.unlink()
        if not link.exists():
            link.symlink_to(checkpoint_path)
        checkpoint_path = link

    print(f"Initializing HPSv3 for {len(rows)} images...", flush=True)
    inferencer = HPSv3RewardInferencer(
        config_path=str(config_path) if config_path else None,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        device="cuda",
    )
    language_model = getattr(inferencer.model.model, "language_model", None)
    if language_model is not None and not hasattr(inferencer.model.model, "embed_tokens"):
        inferencer.model.model.embed_tokens = language_model.embed_tokens
    print("HPSv3 initialized.", flush=True)

    out = {}
    raw = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            generated_paths = [Path(row["generated"]).resolve() for row in batch]
            image_paths = [
                str(representative_image_for_hpsv3(path, output_dir, frame_index=frame_index).resolve())
                for path in generated_paths
            ]
            prompts = [str(row.get("prompt") or sidecar_for_image(path).get("prompt", "")) for path, row in zip(generated_paths, batch)]
            rewards = inferencer.reward(image_paths, prompts)
            for generated_path, image_path, prompt, reward in zip(generated_paths, image_paths, prompts, rewards):
                score = float(reward[0].detach().cpu().item())
                out[str(generated_path)] = score
                raw.append({"generated_path": str(generated_path), "image_path": image_path, "prompt": prompt, "hpsv3_score": score})
            write_json(output_dir / "hpsv3_raw.json", raw)
            print(f"HPSv3 scored {min(start + batch_size, len(rows))}/{len(rows)} images.", flush=True)
    return out


def run_config(run_dir: Path) -> dict[str, Any]:
    system = maybe_json(run_dir / "system_params.json") or {}
    cfg = system.get("config") or {}
    infer = cfg.get("infer") or {}
    return {"attn_type": infer.get("attn_type")}


def case_for_request(run_dir: Path, cfg: dict[str, Any], request: dict[str, Any]) -> str:
    params = request.get("params") or {}
    role = str(params.get("role") or "").strip()
    if role and role.lower() != "warmup":
        return role
    case = str(cfg.get("attn_type") or run_dir.name)
    return "origin_flash" if case == "flash" else case


def collect_rows(experiment_dir: Path, origin_dir: Path, max_frames: int) -> list[dict[str, Any]]:
    ref_index = build_reference_index(origin_dir)
    runs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and (path / "request_params.json").exists() and not (path / ".skip_perf_collect").exists()
    ]
    rows = []
    lpips_pairs = []
    for run_dir in sorted(runs):
        cfg = run_config(run_dir)
        payload = maybe_json(run_dir / "request_params.json") or {}
        for request in payload.get("requests", []):
            if is_warmup_request(request):
                continue
            task_id = str(request.get("request_id") or "")
            params = request.get("params") or {}
            prompt = str(params.get("prompt") or "")
            seed = params.get("seed")
            gen_path = output_for_task(run_dir, task_id)
            ref_path = ref_index.get((prompt, int(seed))) if seed is not None else None
            if gen_path is None or ref_path is None:
                continue
            gen, ref = aligned_arrays(gen_path, ref_path, max_frames=max_frames)
            if gen.shape != ref.shape:
                continue
            gen_metric = gen.reshape((-1, *gen.shape[-3:])) if gen.ndim == 4 else gen
            ref_metric = ref.reshape((-1, *ref.shape[-3:])) if ref.ndim == 4 else ref
            if gen_metric.ndim == 4:
                psnr = mean(
                    float(peak_signal_noise_ratio(ref_frame, gen_frame, data_range=255.0))
                    for gen_frame, ref_frame in zip(gen_metric, ref_metric)
                )
                ssim = mean(
                    float(structural_similarity(ref_frame, gen_frame, channel_axis=-1, data_range=255.0))
                    for gen_frame, ref_frame in zip(gen_metric, ref_metric)
                )
            else:
                psnr = float(peak_signal_noise_ratio(ref_metric, gen_metric, data_range=255.0))
                ssim = float(structural_similarity(ref_metric, gen_metric, channel_axis=-1, data_range=255.0))
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "run_name": run_dir.name,
                    "case": case_for_request(run_dir, cfg, request),
                    "task_id": task_id,
                    "prompt": prompt,
                    "seed": seed,
                    "generated": str(gen_path.resolve()),
                    "reference": str(ref_path.resolve()),
                    "psnr": psnr,
                    "ssim": ssim,
                    "one_minus_lpips": None,
                    "hpsv3_score": None,
                }
            )
            lpips_pairs.append((gen_path, ref_path))
    for row, score in zip(rows, compute_lpips_rows(lpips_pairs)):
        row["one_minus_lpips"] = None if score is None else 1.0 - score
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)
    out = []
    for case, items in sorted(grouped.items()):
        out.append(
            {
                "case": case,
                "num_tasks": len(items),
                "psnr_mean": mean(numeric([item.get("psnr") for item in items])) if numeric([item.get("psnr") for item in items]) else None,
                "ssim_mean": mean(numeric([item.get("ssim") for item in items])) if numeric([item.get("ssim") for item in items]) else None,
                "one_minus_lpips_mean": mean(numeric([item.get("one_minus_lpips") for item in items])) if numeric([item.get("one_minus_lpips") for item in items]) else None,
                "hpsv3_score_mean": mean(numeric([item.get("hpsv3_score") for item in items])) if numeric([item.get("hpsv3_score") for item in items]) else None,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--origin-dir")
    parser.add_argument("--skip-hpsv3", action="store_true")
    parser.add_argument("--max-frames", type=int, default=-1, help="Maximum aligned video frames for PSNR/SSIM; <=0 uses all frames.")
    parser.add_argument("--hpsv3-batch-size", type=int, default=4)
    parser.add_argument("--hpsv3-config-path")
    parser.add_argument("--hpsv3-checkpoint-path")
    parser.add_argument("--hpsv3-frame-index", type=int, default=-1, help="Video frame index for HPSv3; -1 uses the middle frame.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    origin_dir = Path(args.origin_dir).resolve() if args.origin_dir else next(sorted(experiment_dir.glob("*flash*")).__iter__())
    rows = collect_rows(experiment_dir, origin_dir, args.max_frames)
    quality_dir = experiment_dir / "quality"
    if not args.skip_hpsv3:
        scores = compute_hpsv3(
            rows,
            quality_dir,
            Path(args.hpsv3_config_path).resolve() if args.hpsv3_config_path else None,
            Path(args.hpsv3_checkpoint_path).resolve() if args.hpsv3_checkpoint_path else None,
            max(1, args.hpsv3_batch_size),
            args.hpsv3_frame_index,
        )
        for row in rows:
            row["hpsv3_score"] = scores.get(str(Path(row["generated"]).resolve()))

    summary_rows = aggregate(rows)
    write_json(quality_dir / "quality_rows.json", rows)
    write_json(quality_dir / "quality_summary.json", summary_rows)
    write_csv(quality_dir / "quality_rows.csv", rows)
    write_csv(quality_dir / "quality_summary.csv", summary_rows)
    print(f"Wrote {quality_dir / 'quality_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
