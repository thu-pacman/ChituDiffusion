import copy

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


# Reference values from SmartDiffusion exp/fpp at
# b64ddaa268516901a19372e421af56eaaf6f43c9, fm_solvers_unipc.py.
# Keep startup/final checkpoints for each shift; the 50-step case also covers
# the transition to second order, mid-run previews, and late-run candidates.
# These are outputs of the original solver, not this implementation.
@pytest.mark.parametrize(
    "steps,shift,grid,candidates",
    [
        (
            6,
            1,
            {
                0: (999, 0.9990000128746033),
                5: (166, 0.1665000021457672),
            },
            {
                (0, 0): [0.0, 0.95079589, 1.90598702, 2.86957598],
                (5, 0): [-0.542256, 0.24310052, 1.0960685, 2.02978206],
                (5, 1): [-0.56969386, 0.22243538, 1.08402205, 2.0274303],
                (5, 2): [-0.59285557, 0.2073558, 1.07837152, 2.03171372],
            },
        ),
        (
            6,
            3,
            {
                0: (999, 0.9996664524078369),
                5: (374, 0.37471869587898254),
            },
            {
                (0, 0): [0.0, 0.98152459, 1.96469951, 2.95102787],
                (5, 0): [-0.57687414, 0.21349534, 1.07412076, 2.01745272],
                (5, 1): [-0.63862485, 0.16698718, 1.04700959, 2.0121603],
                (5, 2): [-0.69075167, 0.13304961, 1.03429282, 2.02180028],
            },
        ),
        (
            50,
            3,
            {
                0: (999, 0.9996664524078369),
                1: (992, 0.9929074645042419),
                25: (749, 0.7496247887611389),
                48: (111, 0.11100822687149048),
                49: (57, 0.05763683468103409),
            },
            {
                (0, 0): [0.0, 0.99800259, 1.99618351, 2.99470544],
                (1, 0): [-0.00110403, 0.99491209, 1.9913826, 2.9886229],
                (1, 1): [-0.00326809, 0.99297428, 1.9898442, 2.98762131],
                (1, 2): [-0.00530185, 0.9912855, 1.98865128, 2.98703074],
                (25, 0): [-0.19654144, 0.83009076, 1.87190104, 2.92515326],
                (25, 1): [-0.19158964, 0.83570182, 1.87766993, 2.93056512],
                (25, 2): [-0.18614466, 0.84147573, 1.88325691, 2.93546629],
                (48, 0): [0.34550864, 1.40758729, 2.43325853, 3.4202261],
                (48, 1): [0.3415671, 1.39768612, 2.41828227, 3.40151262],
                (48, 2): [0.33357918, 1.38427293, 2.40064216, 3.38122106],
                (49, 0): [0.39020464, 1.43519247, 2.44130731, 3.40799928],
                (49, 1): [0.38694021, 1.42881262, 2.43238187, 3.39732575],
                (49, 2): [0.38154843, 1.42065716, 2.42219114, 3.38600993],
            },
        ),
        (
            6,
            5,
            {
                0: (999, 0.9997998476028442),
                5: (499, 0.4996998906135559),
            },
            {
                (0, 0): [0.0, 0.98862737, 1.97827065, 2.96985507],
                (5, 0): [-0.56841558, 0.21893263, 1.07605124, 2.01570368],
                (5, 1): [-0.6507622, 0.15691245, 1.03989756, 2.00864601],
                (5, 2): [-0.72027504, 0.11165556, 1.02293932, 2.02150106],
            },
        ),
    ],
)
def test_unipc_matches_b64ddaa_checkpoints_and_preserves_history(
    steps, shift, grid, candidates
):
    scheduler = make_wan_scheduler("unipc", shift)
    scheduler.set_timesteps(steps)
    for step, (timestep, sigma) in grid.items():
        assert scheduler.timesteps[step].item() == timestep
        assert scheduler.sigmas[step].item() == sigma
    assert scheduler.sigmas[-1].item() == 0
    x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    for step in range(steps):
        before = _history(scheduler)
        base = x.clone()
        for patch in range(1 if step == 0 else 3):
            prediction = torch.sin(
                torch.arange(4).reshape_as(x) * 0.3 + 0.1 * step + 0.2 * patch
            )
            commit = step == 0 or patch == 2
            candidate = fpp_scheduler_step(
                scheduler, prediction, scheduler.timesteps[step], x, commit=commit
            )
            if (step, patch) in candidates:
                torch.testing.assert_close(
                    candidate.flatten(),
                    torch.tensor(candidates[step, patch]),
                    rtol=1e-6,
                    atol=1e-6,
                )
            torch.testing.assert_close(x, base, rtol=0, atol=0)
            if not commit:
                _assert_history(scheduler, before)
        x = candidate
        assert scheduler.step_index == step + 1
