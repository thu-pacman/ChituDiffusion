from logging import getLogger

import numpy as np
from skimage.metrics import structural_similarity

from chitu_diffusion.eval.strategy.reference_base import ReferenceMetricStrategy
from chitu_diffusion.eval.utils.distributed import get_rank
from chitu_diffusion.eval.utils.reference_metrics import align_video_pair, load_video_frames

logger = getLogger(__name__)


class SsimStrategy(ReferenceMetricStrategy):
    def __init__(self, output_dir: str = "./eval_out"):
        super().__init__(metric_name="ssim", output_dir=output_dir)

    def evaluate(self, payload=None, args=None, max_frames: int = 16, **kwargs):
        if get_rank() != 0:
            return None

        pairs = (payload or {}).get("pairs", [])
        if not pairs:
            return {
                "metric": "ssim",
                "status": "skipped",
                "message": (payload or {}).get("skip_reason", "no valid pairs"),
            }

        per_video = []
        for pair in pairs:
            gen_frames = load_video_frames(pair["generated"], max_frames=-1)
            ref_frames = load_video_frames(pair["reference"], max_frames=-1)
            gen_frames, ref_frames = align_video_pair(gen_frames, ref_frames, max_frames=max_frames)
            if len(gen_frames) == 0 or len(ref_frames) == 0:
                continue

            frame_scores = []
            for gen_frame, ref_frame in zip(gen_frames, ref_frames):
                score = structural_similarity(
                    ref_frame,
                    gen_frame,
                    channel_axis=-1,
                    data_range=255.0,
                )
                frame_scores.append(float(score))

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
                "metric": "ssim",
                "status": "skipped",
                "message": "no valid frame pairs",
            }
        else:
            result = {
                "metric": "ssim",
                "status": "success",
                "mean_score": float(np.mean([item["score"] for item in per_video])),
                "num_videos": len(per_video),
                "per_video": per_video,
            }

        result_path = self.save_result(result)
        result["result_path"] = result_path
        logger.info("SSIM eval done: %s", result_path)
        return result
