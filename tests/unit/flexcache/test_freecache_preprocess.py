from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chitu_diffusion.flexcache.budget_profiles import BUDGET_PROFILES
from chitu_diffusion.flexcache.presets import preview_profile
from tools.freecache.collect import Recorder, observe
from tools.freecache.evaluate import quality, warmup_periodic
from tools.freecache.preprocess import (
    DATA,
    Objective,
    compile_ranking,
    main,
    normalized_setting,
    propagation_weights,
    reproduce,
    write_json,
)


def test_release_reproduces_all_150_profiles_from_canonical_inputs():
    result = reproduce()
    assert result["compiled"] == json.loads(json.dumps(BUDGET_PROFILES))
    for family, profiles in result["profiles"].items():
        for budget, record in enumerate(profiles, 1):
            assert record == json.loads(
                json.dumps(asdict(preview_profile(family, budget)))
            )
        # Presets are historical selected inputs, not new optimization results.
        legacy = json.loads((DATA / "legacy-selected-profiles.json").read_text())
        dense = json.loads((DATA / family / "searched_dense.json").read_text())
        for budget in (10, 17, 25):
            source = (
                legacy["models"]["zimage"]["profiles"][str(budget)]
                if family == "zimage" and budget != 17
                else dense["profiles"][str(budget)]
            )
            for field in ("fresh_steps", "proposal_coefficient"):
                assert profiles[budget - 1][field] == source[field]


@pytest.mark.parametrize("coherence", [0, 0.6, 1])
def test_gram_objective_matches_direct_velocity_defects(coherence):
    velocities = np.asarray([[1.0, 0.0], [1.2, 0.3], [0.8, 0.5], [0.4, 0.7]])
    sigmas = np.asarray([1.0, 0.8, 0.5, 0.1, 0.0])
    weights = np.asarray([3.0, 2.0, 1.0, 0.5])
    trace = {"sigmas": sigmas.tolist(), "gram": (velocities @ velocities.T).tolist()}
    objective = Objective([trace], coherence=coherence, weights=weights)
    defects = np.stack(
        [
            velocities[1]
            + 0.5
            * (sigmas[i] - sigmas[1])
            / (sigmas[1] - sigmas[0])
            * (velocities[1] - velocities[0])
            - velocities[i]
            for i in (2, 3)
        ]
    )
    increments = np.diff(sigmas)
    weighted = defects * (increments[2:] * weights[2:])[:, None]
    expected_sq = sum(
        coherence ** abs(i - j) * np.dot(weighted[i], weighted[j])
        for i in range(2)
        for j in range(2)
    )
    expected = np.sqrt(max(expected_sq, 0)) / np.linalg.norm(increments @ velocities)
    assert objective((0, 1), 0.5) == pytest.approx(expected)
    assert objective((0, 1, 2, 3), 0.5) == 0


def test_warmup_is_charged_and_never_deleted():
    velocities = np.arange(18, dtype=float).reshape(6, 3)
    trace = {
        "sigmas": [1, 0.8, 0.6, 0.4, 0.2, 0, 0],
        "gram": (velocities @ velocities.T).tolist(),
    }
    compiled = compile_ranking(
        Objective([trace], coherence=1, weights=np.ones(6)), warmup=4
    )
    assert compiled["choices"][:4] == [None] * 4
    for ranking in compiled["rankings"]:
        assert ranking[:4] == [0, 1, 2, 3]
        assert sorted(ranking) == list(range(6))
    for budget in range(4, 51):
        schedule = warmup_periodic(budget, 4)
        assert len(schedule) == len(set(schedule)) == budget
        assert schedule[:4] == [0, 1, 2, 3]
        assert schedule[-1] < 50


def test_invalid_inputs_and_no_overwrite(tmp_path):
    trace = {"sigmas": [1, 0.5, 0], "gram": [[1, 0], [0, 1]]}
    with pytest.raises(ValueError, match="sigma grid"):
        Objective([trace, dict(trace, sigmas=[1, 0.4, 0])], coherence=1, weights=[1, 1])
    with pytest.raises(ValueError, match="positive finite"):
        Objective([trace], coherence=1, weights=[1, float("nan")])
    dest = tmp_path / "one.json"
    write_json(dest, {"keep": True})
    with pytest.raises(FileExistsError):
        write_json(dest, {"keep": False})
    assert json.loads(dest.read_text()) == {"keep": True}


def test_legacy_weight_semantics_are_deviation_not_gain(tmp_path):
    path = tmp_path / "prop.csv"
    path.write_text("inject_step,final_deviation,gain\n0,8,1\n2,2,1\n")
    assert propagation_weights(path, 3) == pytest.approx([1.6, 0.8, 0.4])


def test_hook_captures_consumed_velocity_injects_after_update_and_restores():
    class Scheduler:
        def __init__(self):
            self.step_index = None
            self.sigmas = torch.tensor([1.0, 0.5, 0.0])

        def step(self, velocity, timestep, sample, return_dict=False):
            self.step_index = (self.step_index or 0) + 1
            return (sample - 0.5 * velocity,)

    pipeline = SimpleNamespace(
        diffusers_pipeline=SimpleNamespace(scheduler=Scheduler())
    )
    original = Scheduler.step
    recorder = Recorder(injection=(0, torch.tensor([0.25, -0.5])))
    with pytest.raises(RuntimeError, match="forced"):
        with observe(pipeline, recorder):
            scheduler = Scheduler()  # runtime clones the scheduler per request
            actual = scheduler.step(
                torch.tensor([2.0, 4.0]), 1, torch.tensor([1.0, 3.0])
            )[0]
            assert torch.equal(actual, torch.tensor([0.25, 0.5]))
            assert torch.equal(recorder.velocities[0], torch.tensor([2.0, 4.0]))
            assert recorder.actual_injection_norm == pytest.approx(
                np.sqrt(0.25**2 + 0.5**2)
            )
            raise RuntimeError("forced")
    assert Scheduler.step is original


def test_native_psnr_and_exact_reference():
    reference = np.zeros((1, 17, 29, 3), dtype=np.float32)
    assert quality(reference, reference)["exact"]
    result = quality(reference, reference + 0.1)
    assert result["native_shape"] == [1, 17, 29, 3]
    assert result["psnr"] == pytest.approx(20)


def test_default_key_order_does_not_change_scheduler_semantics():
    left = {"scheduler_config": {"shift": 3, "_use_default_values": ["a", "b"]}}
    right = {"scheduler_config": {"shift": 3, "_use_default_values": ["b", "a"]}}
    assert normalized_setting(left) == normalized_setting(right)
    right["scheduler_config"]["shift"] = 4
    assert normalized_setting(left) != normalized_setting(right)


@pytest.mark.parametrize("complete", [False, True])
def test_fit_rejects_partial_or_mismatched_collections(tmp_path, monkeypatch, complete):
    trace = tmp_path / "traces.json"
    trace.write_text(
        json.dumps(
            {
                "complete": complete,
                "protocol": {"setting": {"guidance": 5}, "sigmas": [1, 0]},
            }
        )
    )
    propagation = tmp_path / "probe"
    propagation.mkdir()
    (propagation / "protocol.json").write_text(
        json.dumps({"complete": complete, "setting": {"guidance": 4}, "sigmas": [1, 0]})
    )
    output = tmp_path / "candidate.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "preprocess",
            "fit",
            "--traces",
            str(trace),
            "--propagation",
            str(propagation),
            "--coherence",
            "0",
            "--warmup",
            "10",
            "--budgets",
            "25",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(
        ValueError, match="settings differ" if complete else "completed"
    ):
        main()
    assert not output.exists()


@pytest.mark.parametrize("steps", [40, 50, 60])
def test_evaluator_accepts_public_dict_stats_and_checks_full_fresh(
    tmp_path, monkeypatch, steps
):
    from tools.freecache import evaluate

    prompts = tmp_path / "holdout.json"
    prompts.write_text(json.dumps([{"id": "test", "prompt": "new prompt"}]))
    candidates = tmp_path / "candidates.json"
    selected = asdict(preview_profile("zimage", 25))
    selected.update(reference_steps=steps, fresh_steps=list(range(25)))
    candidates.write_text(
        json.dumps(
            {
                "protocol": {
                    "setting": {},
                    "prompts": [{"prompt": "training"}],
                    "seeds": [1],
                },
                "profiles": [selected],
                "warmup": 10,
            }
        )
    )
    pipeline = SimpleNamespace(last_cache_stats={}, close=lambda: None)

    def generate_stub(pipeline, request):
        budget = (
            len(request.cache.params.profile.fresh_steps)
            if request.cache.strategy == "freecache"
            else steps
        )
        pipeline.last_cache_stats = {"step_misses": budget}
        return SimpleNamespace(
            images=np.full(
                (1, 17, 29, 3), 0 if budget == steps else 0.1, dtype=np.float32
            )
        ), 1.0

    from chitu_diffusion import ZImageRequest

    monkeypatch.setattr(evaluate, "check_gpu_job", lambda: None)
    monkeypatch.setattr(evaluate, "load_model", lambda args: (pipeline, ZImageRequest))
    monkeypatch.setattr(
        evaluate, "protocol_for", lambda *args: {"complete": False, "setting": {}}
    )
    monkeypatch.setattr(evaluate, "generate", generate_stub)
    output = tmp_path / "validation"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--steps",
            str(steps),
            "--model",
            "zimage",
            "--model-path",
            "unused",
            "--prompts",
            str(prompts),
            "--seeds",
            "2",
            "--guidance",
            "5",
            "--output",
            str(output),
            "--candidates",
            str(candidates),
        ],
    )
    evaluate.main()
    assert json.loads((output / "protocol.json").read_text())["complete"]
    rows = json.loads((output / "rows.json").read_text())
    assert len(rows) == 2
    assert all(r["stats"]["step_misses"] == 25 for r in rows)


@pytest.mark.parametrize("steps", [4, 40, 60])
def test_fit_custom_grid_can_load_and_validate_in_runtime(tmp_path, monkeypatch, steps):
    from chitu_diffusion import CacheConfig, FreeCacheConfig, FreeCacheProfile
    from tools.freecache.collect import INJECTION_STEPS, injection_steps

    assert injection_steps(50) == list(INJECTION_STEPS)
    probes = injection_steps(steps)
    assert len(probes) >= 2 and min(probes) >= 2 and max(probes) < steps
    protocol = {
        "complete": True,
        "setting": {"steps": steps},
        "sigmas": np.linspace(1, 0, steps + 1).tolist(),
    }
    velocities = np.random.default_rng(17).normal(size=(steps, 3))
    trace = tmp_path / "traces.json"
    trace.write_text(
        json.dumps(
            {
                "complete": True,
                "steps": steps,
                "model_family": "zimage",
                "protocol": protocol,
                "traces": [
                    {
                        "sigmas": protocol["sigmas"],
                        "gram": (velocities @ velocities.T).tolist(),
                    }
                ],
            }
        )
    )
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "propagation.csv").write_text(
        f"inject_step,final_deviation\n2,1\n{steps - 1},0.5\n"
    )
    output = tmp_path / "candidates.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "preprocess",
            "fit",
            "--traces",
            str(trace),
            "--propagation",
            str(tmp_path),
            "--coherence",
            "0",
            "--warmup",
            "2",
            "--budgets",
            str(steps // 2),
            str(steps),
            "--output",
            str(output),
        ],
    )
    main()
    records = json.loads(output.read_text())["profiles"]
    for budget, record in zip((steps // 2, steps), records):
        profile = FreeCacheProfile(**record)
        assert profile.reference_steps == steps
        assert len(profile.fresh_steps) == budget
        assert profile.fresh_steps[:2] == (0, 1)
        CacheConfig(
            strategy="freecache", params=FreeCacheConfig(profile=profile)
        ).validate_steps(steps)
