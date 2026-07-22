from __future__ import annotations

import math
from typing import Any, Iterable


class MeasuredStepCostModel:
    """Startup-measured DiT latency indexed by sequence length and lane width."""

    source = "startup_warmup"

    def __init__(self) -> None:
        self._latency_ms: dict[tuple[int, int, int, int], float] = {}
        self._rows: list[dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return bool(self._latency_ms)

    def initialize(self, rows: Iterable[dict[str, Any]]) -> None:
        latency: dict[tuple[int, int, int, int], float] = {}
        normalized = []
        for row in rows:
            sequence_length = row.get("sequence_length", row.get("image_tokens"))
            if sequence_length is None:
                raise ValueError("warmup row requires sequence_length or image_tokens")
            key = (
                int(sequence_length),
                int(row["width"]),
                int(row.get("batch_size", 1)),
                int(row.get("conditions", row.get("cfg_conditions", 1))),
            )
            value = float(row["latency_ms"])
            if min(key) <= 0 or value <= 0:
                raise ValueError(f"invalid warmup row: {row}")
            latency[key] = value
            normalized.append(
                {
                    "sequence_length": key[0],
                    "width": key[1],
                    "batch_size": key[2],
                    "conditions": key[3],
                    "latency_ms": value,
                }
            )
        if not latency:
            raise ValueError("warmup produced no latency rows")
        self._latency_ms = latency
        self._rows = sorted(
            normalized,
            key=lambda row: (
                row["sequence_length"],
                row["width"],
                row["batch_size"],
                row["conditions"],
            ),
        )

    def predict_step_ms(
        self,
        *,
        sequence_length: int | None = None,
        image_tokens: int | None = None,
        width: int,
        batch_size: int = 1,
        conditions: int | None = None,
        cfg_conditions: int | None = None,
    ) -> float:
        if not self._latency_ms:
            raise RuntimeError("cost model has not been initialized by warmup")
        selected_length = sequence_length
        if selected_length is None:
            selected_length = image_tokens
        if selected_length is None:
            raise ValueError("sequence_length or image_tokens is required")
        selected_conditions = conditions
        if selected_conditions is None:
            selected_conditions = cfg_conditions
        if selected_conditions is None:
            measured = [
                value
                for key, value in self._latency_ms.items()
                if key[0] == int(selected_length)
                and key[1] == int(width)
                and key[2] == int(batch_size)
            ]
            if measured:
                return measured[0]
            selected_conditions = 1
        target = (
            int(selected_length),
            int(width),
            int(batch_size),
            int(selected_conditions),
        )
        exact = self._latency_ms.get(target)
        if exact is not None:
            return exact
        candidates = [
            (key, value)
            for key, value in self._latency_ms.items()
            if key[0] == target[0] and key[1] == target[1]
        ]
        if not candidates:
            candidates = [
                (key, value)
                for key, value in self._latency_ms.items()
                if key[1] == target[1]
            ]
        if not candidates:
            candidates = list(self._latency_ms.items())
        key, value = min(
            candidates,
            key=lambda item: tuple(
                abs(candidate - expected)
                for candidate, expected in zip(item[0], target)
            ),
        )
        batch_scale = target[2] / key[2]
        condition_scale = target[3] / key[3]
        return value * batch_scale * condition_scale

    def snapshot(self) -> dict[str, Any]:
        return {"source": self.source, "ready": self.ready, "rows": self._rows}


class RuntimeCostCalibrator:
    """EWMA correction keyed by sequence length, lane width, batch, and conditions."""

    def __init__(
        self,
        model: MeasuredStepCostModel,
        *,
        enabled: bool,
        alpha: float = 0.25,
        warmup_skip: int = 1,
        min_samples: int = 2,
        outlier_clip: float = 4.0,
    ) -> None:
        self.model = model
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        self.warmup_skip = int(warmup_skip)
        self.min_samples = int(min_samples)
        self.outlier_clip = float(outlier_clip)
        self._skipped: dict[tuple[int, int, int, int], int] = {}
        self._count: dict[tuple[int, int, int, int], int] = {}
        self._factor: dict[tuple[int, int, int, int], float] = {}
        self._global_log_sum = 0.0
        self._global_count = 0

    @property
    def global_factor(self) -> float:
        if not self._global_count:
            return 1.0
        return math.exp(self._global_log_sum / self._global_count)

    def factor(
        self,
        sequence_length: int,
        width: int,
        batch_size: int,
        conditions: int,
    ) -> float:
        if not self.enabled:
            return 1.0
        key = (sequence_length, width, batch_size, conditions)
        if self._count.get(key, 0) >= self.min_samples:
            return self._factor[key]
        return self.global_factor

    def observe(
        self,
        *,
        sequence_length: int | None = None,
        image_tokens: int | None = None,
        width: int,
        batch_size: int,
        conditions: int | None = None,
        cfg_conditions: int | None = None,
        measured_step_ms: float,
    ) -> None:
        if not self.enabled or measured_step_ms <= 0:
            return
        selected_length = sequence_length
        if selected_length is None:
            selected_length = image_tokens
        selected_conditions = conditions
        if selected_conditions is None:
            selected_conditions = cfg_conditions
        if selected_length is None or selected_conditions is None:
            raise ValueError("calibration requires sequence length and conditions")
        key = (
            int(selected_length),
            int(width),
            int(batch_size),
            int(selected_conditions),
        )
        skipped = self._skipped.get(key, 0)
        if skipped < self.warmup_skip:
            self._skipped[key] = skipped + 1
            return
        predicted = self.model.predict_step_ms(
            sequence_length=key[0],
            width=key[1],
            batch_size=key[2],
            conditions=key[3],
        )
        ratio = measured_step_ms / predicted
        current = self._factor.get(key)
        if current is not None and (
            ratio > current * self.outlier_clip or ratio * self.outlier_clip < current
        ):
            return
        self._factor[key] = (
            ratio
            if current is None
            else (1.0 - self.alpha) * current + self.alpha * ratio
        )
        self._count[key] = self._count.get(key, 0) + 1
        self._global_log_sum += math.log(ratio)
        self._global_count += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "global_factor": self.global_factor,
            "per_key": {
                ":".join(str(value) for value in key): {
                    "factor": factor,
                    "samples": self._count.get(key, 0),
                }
                for key, factor in self._factor.items()
            },
        }


class CalibratedStepCostModel:
    """Cost-model view that applies online calibration to every prediction."""

    def __init__(
        self,
        model: MeasuredStepCostModel,
        calibrator: RuntimeCostCalibrator,
    ) -> None:
        self.model = model
        self.calibrator = calibrator

    @property
    def ready(self) -> bool:
        return self.model.ready

    @property
    def source(self) -> str:
        return self.model.source

    def predict_step_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
    ) -> float:
        base = self.model.predict_step_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
        )
        return base * self.calibrator.factor(
            sequence_length,
            width,
            batch_size,
            conditions,
        )
