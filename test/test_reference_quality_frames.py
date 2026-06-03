import inspect
import json

import numpy as np

from chitu_diffusion.evaluation.strategy.fid import FidStrategy
from chitu_diffusion.evaluation.strategy.fvd import FvdStrategy
from chitu_diffusion.evaluation.strategy.lpips import LpipsStrategy
from chitu_diffusion.evaluation.strategy.psnr import PsnrStrategy
from chitu_diffusion.evaluation.strategy.reference_base import ReferenceMetricStrategy
from chitu_diffusion.evaluation.strategy.ssim import SsimStrategy
from chitu_diffusion.evaluation.utils.reference_metrics import align_video_pair


class DummyReferenceStrategy(ReferenceMetricStrategy):
    def evaluate(self, *args, **kwargs):
        return None


def test_align_video_pair_full_length_by_default_sentinel():
    gen = np.zeros((81, 4, 4, 3), dtype=np.float32)
    ref = np.zeros((77, 4, 4, 3), dtype=np.float32)

    gen_aligned, ref_aligned = align_video_pair(gen, ref, max_frames=-1)

    assert len(gen_aligned) == 77
    assert len(ref_aligned) == 77


def test_reference_metric_default_max_frames_is_full_video():
    strategies = [PsnrStrategy, LpipsStrategy, SsimStrategy, FidStrategy, FvdStrategy]

    for strategy_cls in strategies:
        signature = inspect.signature(strategy_cls.evaluate)
        assert signature.parameters["max_frames"].default == -1


def test_reference_pairs_match_recursive_origin_by_model_prompt_seed(tmp_path):
    generated_dir = tmp_path / "generated"
    reference_dir = tmp_path / "origin"
    generated_task_dir = generated_dir / "task-gen"
    reference_task_dir = reference_dir / "results" / "task-ref"
    generated_task_dir.mkdir(parents=True)
    reference_task_dir.mkdir(parents=True)

    generated_video = generated_task_dir / "a_cat_walking_on_grass._seed42_step50.mp4"
    correct_reference = reference_task_dir / "a_cat_walking_on_grass._seed42_step50.mp4"
    wrong_model_reference = reference_task_dir / "wrong_model_same_prompt_seed.mp4"
    generated_video.write_bytes(b"")
    correct_reference.write_bytes(b"")
    wrong_model_reference.write_bytes(b"")

    (reference_dir / "system_params.json").write_text(
        json.dumps({"model_name": "Wan2.1-T2V-1.3B"}),
        encoding="utf-8",
    )
    generated_video.with_suffix(".json").write_text(
        json.dumps(
            {
                "filename": generated_video.name,
                "prompt": "A cat walking on grass.",
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    correct_reference.with_suffix(".json").write_text(
        json.dumps(
            {
                "filename": correct_reference.name,
                "prompt": "A cat walking on grass.",
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    wrong_model_reference.with_suffix(".json").write_text(
        json.dumps(
            {
                "filename": wrong_model_reference.name,
                "model_name": "FLUX.2-klein-4B",
                "prompt": "A cat walking on grass.",
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    strategy = DummyReferenceStrategy(metric_name="dummy")
    pairs = strategy._build_video_pairs(
        {"task-gen/a_cat_walking_on_grass._seed42_step50.mp4": "A cat walking on grass."},
        str(generated_dir),
        str(reference_dir),
        model_name="Wan2.1-T2V-1.3B",
    )

    assert len(pairs) == 1
    assert pairs[0]["reference"] == str(correct_reference.resolve())
    assert pairs[0]["match_model"] == "wan2.1-t2v-1.3b"
    assert pairs[0]["match_prompt"] == "a_cat_walking_on_grass."
    assert pairs[0]["match_seed"] == "42"
