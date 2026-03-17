import os
import tempfile
from logging import getLogger
import importlib

import imageio.v2 as imageio
import torch

from chitu_diffusion.eval.strategy.reference_base import ReferenceMetricStrategy
from chitu_diffusion.eval.utils.distributed import get_rank
from chitu_diffusion.eval.utils.reference_metrics import align_video_pair, load_video_frames

logger = getLogger(__name__)


class FidStrategy(ReferenceMetricStrategy):
    def __init__(self, output_dir: str = "./eval_out"):
        super().__init__(metric_name="fid", output_dir=output_dir)

    def evaluate(self, payload=None, args=None, max_frames: int = 16, batch_size: int = 32, **kwargs):
        if get_rank() != 0:
            return None

        pairs = (payload or {}).get("pairs", [])
        if not pairs:
            return {
                "metric": "fid",
                "status": "skipped",
                "message": (payload or {}).get("skip_reason", "no valid pairs"),
            }

        try:
            fid_module = importlib.import_module("pytorch_fid.fid_score")
            calculate_fid_given_paths = fid_module.calculate_fid_given_paths
        except ImportError:
            return {
                "metric": "fid",
                "status": "skipped",
                "message": "pytorch-fid package not installed",
            }

        device = "cuda" if torch.cuda.is_available() else "cpu"

        with tempfile.TemporaryDirectory(prefix="fid_gen_") as gen_dir, tempfile.TemporaryDirectory(
            prefix="fid_ref_"
        ) as ref_dir:
            written = 0
            for pair in pairs:
                gen_frames = load_video_frames(pair["generated"], max_frames=-1)
                ref_frames = load_video_frames(pair["reference"], max_frames=-1)
                gen_frames, ref_frames = align_video_pair(gen_frames, ref_frames, max_frames=max_frames)
                if len(gen_frames) == 0 or len(ref_frames) == 0:
                    continue

                for idx, (gen_frame, ref_frame) in enumerate(zip(gen_frames, ref_frames)):
                    imageio.imwrite(os.path.join(gen_dir, f"{pair['video_name']}_{idx:04d}.png"), gen_frame.astype("uint8"))
                    imageio.imwrite(os.path.join(ref_dir, f"{pair['video_name']}_{idx:04d}.png"), ref_frame.astype("uint8"))
                    written += 1

            if written == 0:
                return {
                    "metric": "fid",
                    "status": "skipped",
                    "message": "no valid frame pairs",
                }

            score = calculate_fid_given_paths(
                [gen_dir, ref_dir],
                batch_size=batch_size,
                device=device,
                dims=2048,
                num_workers=0,
            )

        result = {
            "metric": "fid",
            "status": "success",
            "score": float(score),
            "num_pairs": len(pairs),
        }
        result_path = self.save_result(result)
        result["result_path"] = result_path
        logger.info("FID eval done: %s", result_path)
        return result
