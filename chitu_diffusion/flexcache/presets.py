"""Bundled presets and calibrated budget extension for 50-step image samplers."""

from .config import FreeCacheProfile

PREVIEW_VERSION = "freecache-v2-preview-20260909"

# Offline-selected schedules, fixed before the preview benchmark.
_PROFILES = {
    "zimage": {
        25: (
            0.5,
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                19,
                23,
                27,
                31,
                35,
                39,
                43,
                47,
                48,
            ),
        ),
        17: (0.0, (0, 1, 2, 3, 4, 5, 7, 9, 11, 14, 18, 22, 28, 35, 42, 46, 48)),
        10: (0.0, (0, 1, 2, 4, 8, 13, 20, 28, 36, 44)),
    },
    "qwen_image": {
        25: (
            0.75,
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                15,
                17,
                19,
                22,
                25,
                28,
                32,
                37,
                42,
                46,
                48,
            ),
        ),
        17: (0.5, (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 27, 33, 41, 47)),
        10: (0.5, (0, 1, 2, 4, 7, 12, 19, 27, 34, 42)),
    },
    "flux1": {
        25: (
            1.0,
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                19,
                20,
                22,
                25,
                28,
                31,
                35,
                38,
                41,
                44,
                46,
                48,
            ),
        ),
        17: (0.75, (0, 1, 2, 3, 4, 5, 7, 9, 12, 15, 19, 23, 28, 34, 40, 44, 47)),
        10: (0.5, (0, 1, 2, 4, 7, 12, 19, 27, 35, 43)),
    },
}


def preview_profile(model_family: str, fresh_budget: int = 25) -> FreeCacheProfile:
    if model_family not in _PROFILES:
        raise ValueError("FreeCache preview supports zimage/qwen_image/flux1")
    if type(fresh_budget) is not int or not 1 <= fresh_budget <= 50:
        raise ValueError("FreeCache preview fresh_budget must be an integer in [1, 50]")
    preset = fresh_budget in _PROFILES[model_family]
    if preset:
        coefficient, steps = _PROFILES[model_family][fresh_budget]
    else:
        from .budget_profiles import BUDGET_PROFILES, COEFFICIENTS

        calibration = BUDGET_PROFILES[model_family]
        choice = calibration["choices"][fresh_budget]
        coefficient = COEFFICIENTS[choice]
        steps = tuple(sorted(calibration["rankings"][choice][:fresh_budget]))
    return FreeCacheProfile(
        profile_id=f"{PREVIEW_VERSION}{'' if preset else '-budget-v1'}-{model_family}-f{fresh_budget}",
        model_family=model_family,
        reference_steps=50,
        fresh_steps=steps,
        proposal_coefficient=coefficient,
    )
