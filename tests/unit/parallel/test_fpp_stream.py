import copy
import json
from pathlib import Path

import pytest
import torch

from chitu_diffusion.models.wan.fpp_scheduler import (
    fpp_scheduler_step,
    make_wan_scheduler,
)
from chitu_diffusion.parallel.pp import FppConfig, RollingKVCache
from chitu_diffusion.parallel.pp.layout import FppTokenLayout


def test_legacy_cp_padding_and_global_rope_indices():
    layout = FppTokenLayout(7800, 21, 2)
    assert (layout.patch_tokens, layout.local_tokens, layout.padded_tokens) == (
        186,
        3906,
        7812,
    )
    patches = [layout.indices(p, None, device="cpu") for p in range(21)]
    assert torch.equal(torch.cat(patches).sort().values, torch.arange(7812))
    assert torch.equal(
        patches[0], torch.cat([torch.arange(186), torch.arange(3906, 4092)])
    )


def test_legacy_order_and_refresh_do_not_restart_rotation():
    config = FppConfig(patches=7, warmup_steps=1, cooldown_steps=1, refresh_interval=3)
    assert config.order(1, 4) == (0, 1, 2, 3, 4, 5, 6)
    assert config.order(2, 4) == (3, 4, 5, 6, 2, 1, 0)
    assert config.order(4, 4) == (6, 2, 1, 0, 5, 4, 3)
    assert [s for s in range(8) if config.full_step(s, 8)] == [0, 3, 6, 7]
    legacy = FppConfig(patches=3, refresh_interval=50, cooldown_steps=0)
    assert [s for s in range(54) if legacy.full_step(s, 54)] == [0, 50]


def test_reference_scattered_patch_updates_only_its_cp_stripes():
    cache = RollingKVCache()
    full = torch.zeros(1, 12, 1, 2)
    cache.update("cond", 0, full, full, bounds=None, tokens=12)
    indices = torch.tensor([2, 3, 8, 9])
    values = torch.ones(1, 4, 1, 2)
    key, _ = cache.update(
        "cond", 0, values, values, bounds=None, tokens=12, indices=indices
    )
    assert torch.equal(
        key[0, :, 0, 0], torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    )


def _history(scheduler):
    return copy.deepcopy(
        {
            name: getattr(scheduler, name)
            for name in (
                "_step_index",
                "model_outputs",
                "timestep_list",
                "last_sample",
                "lower_order_nums",
            )
        }
    )


def _assert_history(actual, expected):
    for name, want in expected.items():
        got = getattr(actual, name)
        if isinstance(want, list):
            for a, b in zip(got, want, strict=True):
                if isinstance(b, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                else:
                    assert a == b
        elif isinstance(want, torch.Tensor):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        else:
            assert got == want


@pytest.mark.parametrize("case_index", range(4))
def test_unipc_matches_b64ddaa_golden_candidates_and_preserves_history(case_index):
    path = Path(__file__).parents[2] / "fixtures/smart_fpp_unipc_b64ddaa.json"
    case = json.loads(path.read_text())["cases"][case_index]
    scheduler = make_wan_scheduler("unipc", case["shift"])
    scheduler.set_timesteps(case["steps"])
    assert scheduler.timesteps.tolist() == case["timesteps"]
    assert scheduler.sigmas.tolist() == case["sigmas"]
    x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    for step, expected in enumerate(case["candidates"]):
        before = _history(scheduler)
        base = x.clone()
        for patch, target in enumerate(expected):
            prediction = torch.sin(
                torch.arange(4).reshape_as(x) * 0.3 + 0.1 * step + 0.2 * patch
            )
            commit = step == 0 or patch == 2
            candidate = fpp_scheduler_step(
                scheduler, prediction, scheduler.timesteps[step], x, commit=commit
            )
            torch.testing.assert_close(
                candidate.flatten(), torch.tensor(target), rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(x, base, rtol=0, atol=0)
            if not commit:
                _assert_history(scheduler, before)
        x = candidate
        assert scheduler.step_index == step + 1
