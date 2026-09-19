from types import SimpleNamespace

import numpy as np
import pytest

from tools.benchmarks.wan_fpp import build_cases
from tools.benchmarks.wan_fpp_quality import paired_psnr


def test_legacy_matrix_and_warmup_sweep_are_reproducible(tmp_path):
    args = SimpleNamespace(
        case=None,
        mode="matrix",
        output_dir=tmp_path,
        model_path=tmp_path / "model",
        prompt="test",
        negative_prompt="",
        height=480,
        width=832,
        frames=81,
        steps=50,
        flow_shift=3,
        guidance_scale=7.5,
        warmup=1,
        cooldown=1,
        seeds=[42, 43],
        reference_stages=4,
        reference_context_degree=2,
        patches=7,
    )
    jobs = build_cases(args)
    assert len(jobs) == 22
    mixed = next(job for job in jobs if job["case"] == "2cfg2cp2pp")
    assert mixed["world_size"] == 8
    assert (
        mixed["command"][mixed["command"].index("--context-parallel-degree") + 1] == "2"
    )
    args.mode = "warmup"
    jobs = build_cases(args)
    assert len(jobs) == 42 and all(job["world_size"] == 1 for job in jobs)
    assert len({job["output"] for job in jobs}) == 42
    assert any(job["warmup"] == 5 and job["cooldown"] == 5 for job in jobs)
    assert {job["seed"] for job in jobs} == {42, 43}


def test_paired_psnr_has_explicit_range_and_shape_contract():
    zeros = np.zeros((2, 3, 4, 3), dtype=np.float32)
    assert paired_psnr(zeros, zeros)["identical"]
    assert paired_psnr(zeros, zeros + 0.1)["psnr_db"] == pytest.approx(20)
    with pytest.raises(ValueError, match="identical shapes"):
        paired_psnr(zeros, zeros[:1])
