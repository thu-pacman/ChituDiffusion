from logging import getLogger
import importlib
from contextlib import nullcontext
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from chitu_diffusion.eval.strategy.reference_base import ReferenceMetricStrategy
from chitu_diffusion.eval.utils.distributed import get_rank
from chitu_diffusion.eval.utils.reference_metrics import align_video_pair, load_video_frames

logger = getLogger(__name__)


class LpipsStrategy(ReferenceMetricStrategy):
    def __init__(self, output_dir: str = "./eval_out", network: str = "alex"):
        super().__init__(metric_name="lpips", output_dir=output_dir)
        self.network = network

    def evaluate(self, payload=None, args=None, max_frames: int = 16, **kwargs):
        if get_rank() != 0:
            return None

        pairs = (payload or {}).get("pairs", [])
        if not pairs:
            result = {
                "metric": "lpips",
                "status": "skipped",
                "message": (payload or {}).get("skip_reason", "no valid pairs"),
            }
            result_path = self.save_result(result)
            result["result_path"] = result_path
            logger.warning("LPIPS eval skipped: %s", result["message"])
            return result

        try:
            lpips = importlib.import_module("lpips")
        except ImportError as exc:
            install_hint = (
                "lpips package not installed. "
                "Install with: uv sync --extra eval "
                "or lightweight install in current env: "
                f"{sys.executable} -m pip install --no-deps lpips==0.1.4 "
                "(if pip missing: python -m ensurepip --upgrade)"
            )
            result = {
                "metric": "lpips",
                "status": "skipped",
                "message": f"{install_hint}. python={sys.executable}. ImportError: {exc}",
            }
            result_path = self.save_result(result)
            result["result_path"] = result_path
            logger.warning("LPIPS eval skipped: %s", result["message"])
            return result

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*parameter 'pretrained' is deprecated.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
                    category=UserWarning,
                )
                metric_model = lpips.LPIPS(net=self.network).to(device=device, dtype=torch.float32)
            metric_model = metric_model.float()
        except Exception as exc:
            result = {
                "metric": "lpips",
                "status": "failed",
                "message": f"failed to initialize lpips model: {exc}",
            }
            result_path = self.save_result(result)
            result["result_path"] = result_path
            logger.exception("LPIPS model init failed")
            return result
        metric_model.eval()
        model_dtype = next(metric_model.parameters()).dtype

        per_video = []
        for pair in pairs:
            gen_frames = load_video_frames(pair["generated"], max_frames=-1)
            ref_frames = load_video_frames(pair["reference"], max_frames=-1)
            gen_frames, ref_frames = align_video_pair(gen_frames, ref_frames, max_frames=max_frames)
            if len(gen_frames) == 0 or len(ref_frames) == 0:
                continue

            frame_scores = []
            for gen_frame, ref_frame in zip(gen_frames, ref_frames):
                gen_tensor = torch.from_numpy(gen_frame).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=model_dtype)
                ref_tensor = torch.from_numpy(ref_frame).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=model_dtype)

                gen_tensor = gen_tensor / 127.5 - 1.0
                ref_tensor = ref_tensor / 127.5 - 1.0

                if gen_tensor.shape[-2:] != (224, 224):
                    gen_tensor = F.interpolate(gen_tensor, size=(224, 224), mode="bilinear", align_corners=False)
                    ref_tensor = F.interpolate(ref_tensor, size=(224, 224), mode="bilinear", align_corners=False)

                amp_ctx = torch.autocast(device_type="cuda", enabled=False) if device == "cuda" else nullcontext()
                with torch.no_grad(), amp_ctx:
                    score = metric_model(gen_tensor, ref_tensor)
                frame_scores.append(float(score.item()))

            if frame_scores:
                per_video.append(
                    {
                        "video_name": pair["video_name"],
                        "score": float(np.mean(frame_scores)),
                        "num_frames": len(frame_scores),
                    }
                )

        if not per_video:
            result = {
                "metric": "lpips",
                "status": "skipped",
                "message": "no valid frame pairs",
            }
            result_path = self.save_result(result)
            result["result_path"] = result_path
            logger.warning("LPIPS eval skipped: %s", result["message"])
            return result
        else:
            result = {
                "metric": "lpips",
                "status": "success",
                "mean_score": float(np.mean([item["score"] for item in per_video])),
                "num_videos": len(per_video),
                "per_video": per_video,
            }

        result_path = self.save_result(result)
        result["result_path"] = result_path
        logger.info("LPIPS eval done: %s", result_path)
        return result
