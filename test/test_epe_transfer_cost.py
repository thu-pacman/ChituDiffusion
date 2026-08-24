from __future__ import annotations

import pytest

from chitu_diffusion.epac.cost import MeasuredTransferCostModel


def test_unmeasured_transfer_uses_nonzero_conservative_fallback() -> None:
    model = MeasuredTransferCostModel(
        fallback_bandwidth_gbps=10,
        fallback_setup_ms=1,
        fallback_per_tensor_ms=0.2,
        fallback_sync_ms=0.5,
    )

    predicted = model.predict_transfer_ms(
        bytes=10_000_000,
        source=0,
        destination=1,
        tensor_count=5,
    )

    assert predicted == pytest.approx(10.5)


def test_transfer_model_uses_measured_p95_and_bundle_overhead() -> None:
    model = MeasuredTransferCostModel()
    model.initialize(
        [
            {
                "bytes": 1_000,
                "source": 0,
                "destination": 1,
                "tensor_count": 2,
                "latency_ms": 2.0,
                "samples_ms": [1.0, 2.0, 3.0],
                "setup_ms": 0.25,
                "per_tensor_ms": 0.25,
                "sync_ms": 0.25,
            }
        ]
    )

    assert model.predict_transfer_ms(
        bytes=1_000,
        source=0,
        destination=1,
        tensor_count=2,
    ) == pytest.approx(2.9)
    assert model.predict_transfer_ms(
        bytes=2_000,
        source=0,
        destination=1,
        tensor_count=4,
    ) == pytest.approx(5.3)
