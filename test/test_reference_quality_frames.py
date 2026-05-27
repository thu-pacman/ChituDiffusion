import inspect

import numpy as np

from chitu_diffusion.evaluation.strategy.fid import FidStrategy
from chitu_diffusion.evaluation.strategy.fvd import FvdStrategy
from chitu_diffusion.evaluation.strategy.lpips import LpipsStrategy
from chitu_diffusion.evaluation.strategy.psnr import PsnrStrategy
from chitu_diffusion.evaluation.strategy.ssim import SsimStrategy
from chitu_diffusion.evaluation.utils.reference_metrics import align_video_pair


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
