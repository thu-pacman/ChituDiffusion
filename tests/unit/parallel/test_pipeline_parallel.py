from __future__ import annotations

import pytest
import torch

from chitu_diffusion.parallel.pp import (
    FppAttentionContext,
    FppConfig,
    FppState,
    PipelineTopology,
    RollingKVCache,
    current_fpp_attention_context,
    fpp_attention_context,
    partition_bounds,
)


@pytest.mark.parametrize(("length", "parts"), [(7800, 21), (31, 3), (5, 5), (1, 1)])
def test_partitions_cover_every_token_exactly_once(length, parts):
    bounds = partition_bounds(length, parts)
    assert [index for start, stop in bounds for index in range(start, stop)] == list(
        range(length)
    )
    sizes = [stop - start for start, stop in bounds]
    assert min(sizes) >= 1 and max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize(("length", "parts"), [(0, 1), (5, 0), (2, 3), (5, -1)])
def test_partitions_reject_empty_patches(length, parts):
    with pytest.raises(ValueError):
        partition_bounds(length, parts)


def test_pipeline_neighbors_are_global_ranks():
    first = PipelineTopology((2, 4, 7), 2)
    middle = PipelineTopology((2, 4, 7), 4)
    last = PipelineTopology((2, 4, 7), 7)
    assert first.next_rank == 4
    assert (middle.previous_rank, middle.next_rank) == (2, 7)
    assert last.previous_rank == 4
    with pytest.raises(ValueError):
        _ = first.previous_rank
    with pytest.raises(ValueError):
        _ = last.next_rank


@pytest.mark.parametrize(
    "kwargs",
    [
        {"patches": 0},
        {"warmup_steps": 0},
        {"cooldown_steps": -1},
        {"refresh_interval": -1},
        {"patches": 2.5},
        {"rotate_patches": "yes"},
    ],
)
def test_invalid_fpp_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        FppConfig(schedule="step", **kwargs)


def test_refresh_and_short_requests_never_change_requested_step_count():
    config = FppConfig(
        schedule="step", warmup_steps=2, cooldown_steps=2, refresh_interval=4
    )
    assert [step for step in range(8) if config.full_step(step, 8)] == [0, 1, 4, 6, 7]
    assert [config.full_step(step, 1) for step in range(1)] == [True]
    assert FppConfig(schedule="step", patches=1).full_step(4, 9)
    assert {
        FppConfig(schedule="step", patches=3).order(step)[0] for step in range(1, 4)
    } == {0, 1, 2}


def test_kv_is_rolling_but_isolated_by_request_branch_and_layer():
    first, second = RollingKVCache(), RollingKVCache()
    zeros = torch.zeros(1, 5, 2, 4)
    for cache in (first, second):
        for branch in ("cond", "uncond"):
            for layer in (0, 1):
                cache.update(branch, layer, zeros, zeros, bounds=None, tokens=5)
    ones = torch.ones(1, 3, 2, 4)
    key, _ = first.update("cond", 0, ones, ones, bounds=(0, 3), tokens=5)
    assert torch.equal(key[:, :3], ones)
    assert not key[:, 3:].any()
    assert not zeros.any()
    assert not second.entries[("cond", 0)][0].any()
    assert not first.entries[("uncond", 0)][0].any()
    assert not first.entries[("cond", 1)][0].any()
    twos = torch.full((1, 2, 2, 4), 2.0)
    updated, _ = first.update("cond", 0, twos, twos, bounds=(3, 5), tokens=5)
    assert updated.data_ptr() == key.data_ptr()
    assert torch.equal(updated[:, :3], ones)


def test_missing_or_incompatible_cache_fails_before_attention():
    cache = RollingKVCache()
    patch = torch.zeros(1, 2, 1, 4)
    with pytest.raises(RuntimeError, match="warmup"):
        cache.update("cond", 0, patch, patch, bounds=(0, 2), tokens=5)
    full = torch.zeros(1, 5, 1, 4)
    cache.update("cond", 0, full, full, bounds=None, tokens=5)
    with pytest.raises(ValueError, match="bounds"):
        cache.update("cond", 0, patch, patch, bounds=(4, 6), tokens=5)
    with pytest.raises(ValueError, match="shape, dtype, or device"):
        cache.update("cond", 0, patch.double(), patch.double(), bounds=(0, 2), tokens=5)


def test_context_restores_after_an_interleaved_request_raises():
    first = FppAttentionContext(RollingKVCache(), "cond", None, 5)
    second = FppAttentionContext(RollingKVCache(), "uncond", (0, 2), 5)
    with fpp_attention_context(first):
        with pytest.raises(RuntimeError, match="test error"):
            with fpp_attention_context(second):
                assert current_fpp_attention_context() is second
                raise RuntimeError("test error")
        assert current_fpp_attention_context() is first
    with pytest.raises(RuntimeError, match="request"):
        current_fpp_attention_context()


def test_discontinuous_state_requires_a_full_refresh():
    state = FppState(FppConfig(schedule="step", cooldown_steps=0))
    assert state.needs_full_step(2, 8)
    full = torch.zeros(1, 5, 1, 4)
    state.cache.update("cond", 0, full, full, bounds=None, tokens=5)
    state.last_step = 1
    assert not state.needs_full_step(2, 8)
    assert state.needs_full_step(4, 8)
