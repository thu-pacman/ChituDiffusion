## 暂不可用

from logging import getLogger
import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Optional, Tuple

import torch

from chitu_diffusion.eval.strategy.reference_base import ReferenceMetricStrategy
from chitu_diffusion.eval.utils.distributed import get_rank
from chitu_diffusion.eval.utils.reference_metrics import align_video_pair, load_video_frames

logger = getLogger(__name__)


class FvdStrategy(ReferenceMetricStrategy):
    def __init__(self, output_dir: str = "./eval_out"):
        super().__init__(metric_name="fvd", output_dir=output_dir)

    def _build_metric(self, device: str):
        try:
            fvd_module = importlib.import_module("torchmetrics.video.fvd")
            FrechetVideoDistance = fvd_module.FrechetVideoDistance
        except ImportError as exc:
            try:
                torchmetrics_version = version("torchmetrics")
            except PackageNotFoundError:
                raise ImportError("torchmetrics package is not installed") from exc
            raise ImportError(
                f"torchmetrics=={torchmetrics_version} does not provide torchmetrics.video.fvd"
            ) from exc

        try:
            version("torch-fidelity")
        except PackageNotFoundError as exc:
            raise ImportError("torch-fidelity package is required by FrechetVideoDistance") from exc

        candidates = [
            {},
            {"feature": 400},
            {"normalize": True},
            {"feature": 400, "normalize": True},
        ]
        for kwargs in candidates:
            try:
                metric = FrechetVideoDistance(**kwargs).to(device)
                return metric
            except TypeError:
                continue
        return FrechetVideoDistance().to(device)

    def _build_frame_fid_metric(self, device: str):
        try:
            fid_module = importlib.import_module("torchmetrics.image.fid")
            FrechetInceptionDistance = fid_module.FrechetInceptionDistance
        except ImportError as exc:
            raise ImportError("torchmetrics.image.fid is required for frame-FID fallback") from exc

        # torchmetrics.image.fid relies on torch-fidelity under the hood.
        try:
            version("torch-fidelity")
        except PackageNotFoundError as exc:
            raise ImportError("torch-fidelity package is required by frame-FID fallback") from exc

        return FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    def _compute_video_fvd(self, metric, pairs, device: str, max_frames: int) -> Tuple[Optional[float], int]:
        valid_count = 0
        for pair in pairs:
            gen_frames = load_video_frames(pair["generated"], max_frames=-1)
            ref_frames = load_video_frames(pair["reference"], max_frames=-1)
            gen_frames, ref_frames = align_video_pair(gen_frames, ref_frames, max_frames=max_frames)
            if len(gen_frames) == 0 or len(ref_frames) == 0:
                continue

            gen_tensor = torch.from_numpy(gen_frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(device) / 255.0
            ref_tensor = torch.from_numpy(ref_frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(device) / 255.0

            try:
                metric.update(ref_tensor, real=True)
                metric.update(gen_tensor, real=False)
            except Exception:
                metric.update((ref_tensor * 255.0).to(torch.uint8), real=True)
                metric.update((gen_tensor * 255.0).to(torch.uint8), real=False)
            valid_count += 1

        if valid_count == 0:
            return None, 0
        score = metric.compute()
        if isinstance(score, torch.Tensor):
            score = score.item()
        return float(score), valid_count

    def _compute_frame_fid(self, metric, pairs, device: str, max_frames: int) -> Tuple[Optional[float], int]:
        valid_frames = 0
        valid_pairs = 0
        for pair in pairs:
            gen_frames = load_video_frames(pair["generated"], max_frames=-1)
            ref_frames = load_video_frames(pair["reference"], max_frames=-1)
            gen_frames, ref_frames = align_video_pair(gen_frames, ref_frames, max_frames=max_frames)
            if len(gen_frames) == 0 or len(ref_frames) == 0:
                continue

            gen_tensor = torch.from_numpy(gen_frames).permute(0, 3, 1, 2).to(device=device, dtype=torch.uint8)
            ref_tensor = torch.from_numpy(ref_frames).permute(0, 3, 1, 2).to(device=device, dtype=torch.uint8)
            metric.update(ref_tensor, real=True)
            metric.update(gen_tensor, real=False)

            valid_pairs += 1
            valid_frames += int(gen_tensor.shape[0])

        if valid_pairs == 0:
            return None, 0

        score = metric.compute()
        if isinstance(score, torch.Tensor):
            score = score.item()
        return float(score), valid_frames

    def evaluate(self, payload=None, args=None, max_frames: int = 16, **kwargs):
        if get_rank() != 0:
            return None

        pairs = (payload or {}).get("pairs", [])
        if not pairs:
            result = {
                "metric": "fvd",
                "status": "skipped",
                "message": (payload or {}).get("skip_reason", "no valid pairs"),
            }
            result_path = self.save_result(result)
            result["result_path"] = result_path
            logger.warning("FVD eval skipped: %s", result["message"])
            return result

        device = "cuda" if torch.cuda.is_available() else "cpu"
        backend = "torchmetrics.video.fvd"
        backend_note = ""

        try:
            metric = self._build_metric(device=device)
            score, valid_count = self._compute_video_fvd(metric, pairs, device=device, max_frames=max_frames)
        except ImportError as exc:
            try:
                metric = self._build_frame_fid_metric(device=device)
                score, valid_count = self._compute_frame_fid(metric, pairs, device=device, max_frames=max_frames)
                backend = "torchmetrics.image.fid"
                backend_note = f"fallback activated because: {exc}"
                logger.warning("FVD backend fallback to frame-FID: %s", exc)
            except ImportError as fid_exc:
                result = {
                    "metric": "fvd",
                    "status": "skipped",
                    "message": f"primary backend unavailable: {exc}; fallback unavailable: {fid_exc}",
                }
                result_path = self.save_result(result)
                result["result_path"] = result_path
                logger.warning("FVD eval skipped: %s", result["message"])
                return result

        if score is None or valid_count == 0:
            result = {
                "metric": "fvd",
                "status": "skipped",
                "message": "no valid frame pairs",
            }
            result_path = self.save_result(result)
            result["result_path"] = result_path
            logger.warning("FVD eval skipped: %s", result["message"])
            return result

        result = {
            "metric": "fvd",
            "status": "success",
            "score": float(score),
            "backend": backend,
            "backend_note": backend_note,
            "num_pairs": int(valid_count),
        }

        result_path = self.save_result(result)
        result["result_path"] = result_path
        logger.info("FVD eval done: %s", result_path)
        return result
