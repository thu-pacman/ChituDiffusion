from __future__ import annotations

import pytest

from chitu_diffusion.models.minimax_h3.cost import (
    H3CostFeatures,
    MiniMaxH3StepCostModel,
)
from chitu_diffusion.models.minimax_h3.fl2va_packed_sequence import (
    build_fl2va_packed_sequence,
)


def _features(text_length: int) -> H3CostFeatures:
    packed = build_fl2va_packed_sequence(
        text_length=text_length,
        latent_t=2,
        latent_h=8,
        latent_w=8,
        audio_t=3,
    )
    return H3CostFeatures.from_packed(packed)


def test_h3_cost_features_preserve_live_and_padding_work() -> None:
    features = _features(17)

    assert features.used_tokens == (
        features.text_tokens
        + features.condition_tokens
        + features.audio_tokens
        + features.video_tokens
    )
    assert features.padded_tokens % 64 == 0
    assert features.attention_work == (
        features.used_tokens**2 + features.pad_tokens**2
    )


def test_h3_cost_features_distinguish_same_padding_bucket() -> None:
    shorter = _features(1)
    longer = _features(17)

    assert shorter.padded_tokens == longer.padded_tokens
    assert shorter.attention_work != longer.attention_work


def test_h3_cost_model_uses_only_sequence_length_and_cp_width() -> None:
    first = _features(1)
    second = _features(17)
    model = MiniMaxH3StepCostModel()
    model.initialize(
        [
            {
                "sequence_length": first.padded_tokens,
                "width": 1,
                "latency_ms": 10,
                "cost_features": first.to_dict(),
            },
            {
                "sequence_length": second.padded_tokens,
                "width": 1,
                "latency_ms": 14,
                "cost_features": second.to_dict(),
            },
        ]
    )

    first_prediction = model.predict_profile_step_ms(
        sequence_length=first.padded_tokens,
        width=1,
        cost_features=first.to_dict(),
    )
    second_prediction = model.predict_profile_step_ms(
        sequence_length=second.padded_tokens,
        width=1,
        cost_features=second.to_dict(),
    )

    assert first.padded_tokens == second.padded_tokens
    assert first_prediction == second_prediction


def test_h3_cost_model_selects_fit_with_validation_rows() -> None:
    model = MiniMaxH3StepCostModel()
    rows = []
    for length, split in (
        (1024, "train"),
        (2048, "validation"),
        (3072, "train"),
        (4096, "validation"),
        (5120, "train"),
    ):
        features = H3CostFeatures(
            text_tokens=0,
            condition_tokens=0,
            audio_tokens=0,
            video_tokens=length,
            used_tokens=length,
            padded_tokens=length,
            pad_tokens=0,
            synthetic=True,
        )
        rows.append(
            {
                "sequence_length": length,
                "width": 2,
                "latency_ms": 5.0 + 0.25 * length,
                "split": split,
                "cost_features": features.to_dict(),
            }
        )
    model.initialize(rows)

    predicted = model.predict_step_ms(sequence_length=2560, width=2)
    assert predicted == pytest.approx(645.0, rel=0.05)
    assert model.snapshot()["fits"]["2:1:1"]["validation"]["mape"] < 0.01
