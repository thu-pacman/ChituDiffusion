from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch import nn

from chitu_diffusion import CacheConfig, FreeCacheConfig, FreeCacheProfile
from chitu_diffusion.commands.cache_args import (
    add_cache_arguments,
    cache_config_from_args,
)
from chitu_diffusion.flexcache import (
    CacheCommonConfig,
    CacheSession,
    FlexCacheModelSpec,
    active_cache_session,
)
from chitu_diffusion.flexcache.budget_profiles import BUDGET_PROFILES
from chitu_diffusion.flexcache.strategies import create_cache_strategy
from chitu_diffusion.flexcache.strategies.freecache import (
    FreeCacheStrategy,
    VelocityAnchors,
)

MODELS = ("zimage", "qwen_image", "flux1")


class ParallelContext:
    world_size = 1

    @contextmanager
    def activate(self, lane_ranks):
        assert lane_ranks == (0,)
        yield SimpleNamespace(process_group=None)


def model_spec(family):
    model = nn.Module()
    model.transformer_blocks = nn.ModuleList([nn.Identity()])
    return replace(
        FlexCacheModelSpec.discover(model, family="qwen_image"), family=family
    )


def state():
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(50)
    return SimpleNamespace(
        scheduler=scheduler,
        timesteps=scheduler.timesteps,
        latents=torch.tensor([[0.25, -0.5]]),
        step_index=0,
    )


def full_step(s):
    i = s.step_index
    velocity = s.latents.square() + i / 50
    s.latents = s.scheduler.step(
        velocity, s.timesteps[i], s.latents, return_dict=False
    )[0]
    s.step_index += 1
    return s


@pytest.mark.parametrize("family", MODELS)
@pytest.mark.parametrize("budget", range(1, 51))
def test_every_budget_roundtrips_and_executes_exactly(family, budget):
    config = CacheConfig(
        strategy="freecache", params=FreeCacheConfig.preview(family, budget)
    )
    assert CacheConfig.from_mapping(json.loads(json.dumps(config.to_dict()))) == config
    config.validate_steps(50)
    with pytest.raises(ValueError):
        config.validate_steps(49)
    strategy = create_cache_strategy(config)
    strategy.begin(total_steps=50, model_spec=model_spec(family))
    s, fresh = state(), []
    pipeline = SimpleNamespace(parallel_context=ParallelContext())

    def fresh_step():
        fresh.append(s.step_index)
        return full_step(s)

    for _ in range(50):
        strategy.denoise_step(pipeline, s, lane_ranks=(0,), call_original=fresh_step)
        assert len(strategy.anchors) <= 2
    assert tuple(fresh) == config.params.profile.fresh_steps
    assert len(fresh) == budget
    assert s.step_index == 50 and torch.isfinite(s.latents).all()
    stats = strategy.end()
    assert stats.strategy["fresh_steps"] == budget
    assert stats.step_hits == 50 - budget
    assert len(strategy.anchors) == 0


@pytest.mark.parametrize("family", MODELS)
def test_default_and_integer_constructor_use_static_profiles(family):
    for budget in (None, 1, 13, 31, 50):
        strategy = FreeCacheStrategy(FreeCacheConfig(fresh_budget=budget))
        strategy.begin(total_steps=50, model_spec=model_spec(family))
        assert strategy.profile == FreeCacheConfig.preview(family, budget or 25).profile
        assert strategy.stats.strategy["budget_mode"] == "profile"


@pytest.mark.parametrize("budget", [0, -1, 51, 13.0, True, "13"])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):
        FreeCacheConfig.preview("zimage", budget)
    with pytest.raises(ValueError):
        FreeCacheConfig(fresh_budget=budget)


def test_no_feedback_repair_or_warmup_overrides():
    for params in (
        {"tolerance": 0.45},
        {"start_step": 1},
        {"profile": {"repair_coefficient": 1}},
    ):
        with pytest.raises(ValueError):
            CacheConfig.from_mapping({"strategy": "freecache", "params": params})
    with pytest.raises(ValueError):
        CacheConfig(strategy="freecache", common=CacheCommonConfig(warmup_steps=1))
    with pytest.raises(NotImplementedError):
        CacheConfig(strategy="freecache").require_serve_available()


@pytest.mark.parametrize("value", ["zimage:13", "qwen_image:31", "flux1:50"])
def test_cli_profile(value):
    parser = argparse.ArgumentParser()
    add_cache_arguments(parser)
    config = cache_config_from_args(
        parser.parse_args(
            ["--cache-strategy", "freecache", "--freecache-profile", value]
        )
    )
    family, budget = value.split(":")
    assert config.params == FreeCacheConfig.preview(family, int(budget))
    with pytest.raises(ValueError):
        cache_config_from_args(
            parser.parse_args(
                [
                    "--cache-strategy",
                    "freecache",
                    "--freecache-profile",
                    value,
                    "--freecache-fresh-budget",
                    "17",
                ]
            )
        )


def test_cli_default_and_budget():
    parser = argparse.ArgumentParser()
    add_cache_arguments(parser)
    assert (
        cache_config_from_args(
            parser.parse_args(["--cache-strategy", "freecache"])
        ).params
        == FreeCacheConfig()
    )
    assert (
        cache_config_from_args(
            parser.parse_args(
                ["--cache-strategy", "freecache", "--freecache-fresh-budget", "13"]
            )
        ).params.fresh_budget
        == 13
    )
    for value in ("zimage:51", "wan:13", "bad"):
        with pytest.raises(ValueError, match=r"\[1, 50\]"):
            cache_config_from_args(
                parser.parse_args(
                    ["--cache-strategy", "freecache", "--freecache-profile", value]
                )
            )


def test_rankings_are_permutations():
    for data in BUDGET_PROFILES.values():
        assert len(data["choices"]) == 51
        assert all(0 <= c < 5 for c in data["choices"])
        for ranking in data["rankings"]:
            assert sorted(ranking) == list(range(50))
            assert ranking[:2] == (0, 1)


def test_preview_schedules_and_coefficients_are_frozen():
    rows = [
        (family, budget, profile.fresh_steps, profile.proposal_coefficient)
        for family in MODELS
        for budget in range(1, 51)
        for profile in [FreeCacheConfig.preview(family, budget).profile]
    ]
    assert hashlib.sha256(json.dumps(rows).encode()).hexdigest() == (
        "4ad236dc3286488eb80237e87ab96f55ebe3422733d0f7a914e08b9c8c7b6b86"
    )


def test_model_mismatch_and_stochastic_sampling_rejected():
    strategy = FreeCacheStrategy(FreeCacheConfig.preview("zimage"))
    with pytest.raises(ValueError, match="match"):
        strategy.begin(total_steps=50, model_spec=model_spec("qwen_image"))
    strategy.begin(total_steps=50, model_spec=model_spec("zimage"))
    s = state()
    s.scheduler.register_to_config(stochastic_sampling=True)
    with pytest.raises(NotImplementedError, match="stochastic"):
        strategy.denoise_step(
            SimpleNamespace(parallel_context=ParallelContext()),
            s,
            lane_ranks=(0,),
            call_original=lambda: full_step(s),
        )


def test_failed_request_restores_session_and_releases_anchors():
    def fail(s, **kwargs):
        raise RuntimeError("forward failed")

    pipeline = SimpleNamespace(
        parallel_context=ParallelContext(),
        denoise_step=fail,
    )
    config = CacheConfig(strategy="freecache")
    strategy = create_cache_strategy(config)
    s = state()
    original_step = s.scheduler.step
    with pytest.raises(RuntimeError, match="forward failed"):
        with CacheSession(
            config=config,
            model_spec=model_spec("zimage"),
            strategy=strategy,
            total_steps=50,
            pipeline=pipeline,
        ):
            pipeline.denoise_step(s, lane_ranks=(0,))
    assert pipeline.denoise_step is fail
    assert s.scheduler.step == original_step
    assert active_cache_session() is None
    assert len(strategy.anchors) == 0


def test_velocity_blend_and_fallbacks_do_not_mutate_history():
    anchors = VelocityAnchors()
    args = dict(device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        anchors.extrapolate(sigma=torch.tensor(0.5), coefficient=1, **args)
    anchors.record(torch.tensor(1.0), torch.tensor([1.0]))
    assert (
        anchors.extrapolate(sigma=torch.tensor(0.5), coefficient=1, **args).item() == 1
    )
    anchors.record(torch.tensor(0.75), torch.tensor([2.0]))
    assert (
        anchors.extrapolate(sigma=torch.tensor(0.5), coefficient=0.5, **args).item()
        == 2.5
    )
    assert anchors.velocities[-1].item() == 2
    assert (
        anchors.extrapolate(
            sigma=torch.tensor(float("inf")), coefficient=1, **args
        ).item()
        == 2
    )
    anchors.record(torch.tensor(0.75), torch.tensor([3.0]))
    assert (
        anchors.extrapolate(sigma=torch.tensor(0.5), coefficient=1, **args).item() == 3
    )


def test_full_budget_is_identical_to_uncached_euler():
    s, reference = state(), state()
    strategy = FreeCacheStrategy(FreeCacheConfig(fresh_budget=50))
    strategy.begin(total_steps=50, model_spec=model_spec("qwen_image"))
    pipeline = SimpleNamespace(parallel_context=ParallelContext())
    for _ in range(50):
        strategy.denoise_step(
            pipeline, s, lane_ranks=(0,), call_original=lambda: full_step(s)
        )
        full_step(reference)
    torch.testing.assert_close(s.latents, reference.latents, rtol=0, atol=0)


def test_scheduler_hook_restored_after_failure():
    strategy = FreeCacheStrategy(FreeCacheConfig())
    strategy.begin(total_steps=50, model_spec=model_spec("zimage"))
    s = state()
    original = s.scheduler.step

    def fail():
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        strategy.denoise_step(
            SimpleNamespace(parallel_context=ParallelContext()),
            s,
            lane_ranks=(0,),
            call_original=fail,
        )
    assert s.scheduler.step == original


@pytest.mark.parametrize("family", MODELS)
def test_session_restores_pipeline_and_is_request_local(family):
    pipeline = SimpleNamespace(
        parallel_context=ParallelContext(), denoise_step=lambda s, **kw: full_step(s)
    )
    original = pipeline.denoise_step
    config = CacheConfig(
        strategy="freecache", params=FreeCacheConfig.preview(family, 13)
    )
    results = []
    for _ in range(2):
        s = state()
        with CacheSession(
            config=config,
            model_spec=model_spec(family),
            strategy=create_cache_strategy(config),
            total_steps=50,
            pipeline=pipeline,
        ) as session:
            for _ in range(50):
                pipeline.denoise_step(s, lane_ranks=(0,))
            assert session.steps_seen == 50
        assert pipeline.denoise_step is original
        assert active_cache_session() is None
        results.append(s.latents.clone())
    torch.testing.assert_close(*results, rtol=0, atol=0)


def test_unsupported_model_solver_and_parallelism_fail_explicitly():
    strategy = FreeCacheStrategy(FreeCacheConfig())
    with pytest.raises(ValueError):
        strategy.begin(total_steps=50, model_spec=model_spec("wan"))
    strategy.begin(total_steps=50, model_spec=model_spec("zimage"))
    pipeline = SimpleNamespace(parallel_context=ParallelContext())
    s = state()
    with pytest.raises(NotImplementedError, match="single-device"):
        strategy.denoise_step(
            pipeline, s, lane_ranks=(0, 1), call_original=lambda: full_step(s)
        )
    s.scheduler = SimpleNamespace()
    with pytest.raises(NotImplementedError, match="Euler"):
        strategy.denoise_step(
            pipeline, s, lane_ranks=(0,), call_original=lambda: full_step(s)
        )


@pytest.mark.parametrize("steps", [(0, 1.5), (0, True), (1,), (0, 0), (0, 50), ()])
def test_profile_rejects_invalid_nodes(steps):
    with pytest.raises(ValueError):
        FreeCacheProfile("test", "zimage", 50, steps)
