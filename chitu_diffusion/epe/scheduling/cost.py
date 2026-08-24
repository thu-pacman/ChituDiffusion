from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _freeze_cost_features(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_cost_features(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cost_features(item) for item in value)
    return value


def cost_feature_signature(
    features: Mapping[str, Any] | None,
) -> tuple[tuple[str, Any], ...]:
    if not features:
        return ()
    frozen = _freeze_cost_features(features)
    assert isinstance(frozen, tuple)
    return frozen


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


class MeasuredStepCostModel:
    """Startup-measured DiT latency indexed by sequence length and lane width."""

    source = "startup_warmup"

    def __init__(self) -> None:
        self._latency_ms: dict[tuple[int, int, int, int], float] = {}
        self._terminal_ms: dict[tuple[int, int, int, int], float] = {}
        self._completion_ms: dict[tuple[int, int, int, int], float] = {}
        self._rows: list[dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return bool(self._latency_ms)

    def initialize(self, rows: Iterable[dict[str, Any]]) -> None:
        latency: dict[tuple[int, int, int, int], float] = {}
        terminal: dict[tuple[int, int, int, int], float] = {}
        completion: dict[tuple[int, int, int, int], float] = {}
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
            normalized_row = {
                "sequence_length": key[0],
                "width": key[1],
                "batch_size": key[2],
                "conditions": key[3],
                "latency_ms": value,
            }
            gpu_ms = float(row.get("terminal_gpu_ms", row.get("vae_latency_ms", 0.0)))
            d2h_ms = float(row.get("terminal_d2h_ms", row.get("d2h_latency_ms", 0.0)))
            cpu_ms = float(row.get("terminal_cpu_ms", 0.0))
            terminal_value = row.get("terminal_lane_ms", row.get("terminal_ms"))
            if terminal_value is None and (gpu_ms or d2h_ms):
                terminal_value = gpu_ms + d2h_ms
            if terminal_value is not None:
                terminal_value = float(terminal_value)
                if terminal_value < 0:
                    raise ValueError(f"invalid terminal warmup row: {row}")
                terminal[key] = terminal_value
                completion_value = float(
                    row.get("completion_tail_ms", terminal_value + cpu_ms)
                )
                if completion_value < terminal_value:
                    raise ValueError(
                        "completion_tail_ms cannot be less than lane terminal cost"
                    )
                completion[key] = completion_value
                normalized_row.update(
                    {
                        "terminal_ms": terminal_value,
                        "terminal_gpu_ms": gpu_ms,
                        "terminal_d2h_ms": d2h_ms,
                        "terminal_cpu_ms": cpu_ms,
                        "completion_tail_ms": completion_value,
                        "vae_latency_ms": float(row.get("vae_latency_ms", gpu_ms)),
                        "d2h_latency_ms": float(row.get("d2h_latency_ms", d2h_ms)),
                    }
                )
            normalized.append(normalized_row)
        if not latency:
            raise ValueError("warmup produced no latency rows")
        self._latency_ms = latency
        self._terminal_ms = terminal
        self._completion_ms = completion
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

    def predict_profile_step_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
        cost_features: Mapping[str, Any] | None = None,
    ) -> float:
        del cost_features
        return self.predict_step_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
        )

    def predict_terminal_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
    ) -> float:
        """Predict lane-critical VAE decode plus leader D2H latency."""
        return self._predict_tail(
            self._terminal_ms,
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
        )

    def predict_profile_terminal_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
        cost_features: Mapping[str, Any] | None = None,
    ) -> float:
        del cost_features
        return self.predict_terminal_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
        )

    def predict_completion_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
    ) -> float:
        """Predict the full terminal tail, including asynchronous CPU mux."""

        return self._predict_tail(
            self._completion_ms,
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
        )

    def predict_profile_completion_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
        cost_features: Mapping[str, Any] | None = None,
    ) -> float:
        del cost_features
        return self.predict_completion_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
        )

    @staticmethod
    def _predict_tail(
        values: dict[tuple[int, int, int, int], float],
        *,
        sequence_length: int,
        width: int,
        batch_size: int,
        conditions: int,
    ) -> float:
        if not values:
            return 0.0
        target = (
            int(sequence_length),
            int(width),
            int(batch_size),
            int(conditions),
        )
        exact = values.get(target)
        if exact is not None:
            return exact
        candidates = [
            (key, value)
            for key, value in values.items()
            if key[0] == target[0] and key[1] == target[1]
        ]
        if not candidates:
            candidates = [
                (key, value)
                for key, value in values.items()
                if key[1] == target[1]
            ]
        if not candidates:
            candidates = list(values.items())
        key, value = min(
            candidates,
            key=lambda item: tuple(
                abs(candidate - expected)
                for candidate, expected in zip(item[0], target)
            ),
        )
        return value * target[2] / key[2]

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
        min_samples: int = 1,
        outlier_clip: float = 4.0,
        tolerance: float = 0.05,
    ) -> None:
        self.model = model
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        self.warmup_skip = int(warmup_skip)
        self.min_samples = int(min_samples)
        self.outlier_clip = float(outlier_clip)
        self.tolerance = float(tolerance)
        if not 0.0 <= self.tolerance < 1.0:
            raise ValueError("calibration tolerance must be in [0, 1)")
        self._skipped: dict[tuple[Any, ...], int] = {}
        self._count: dict[tuple[Any, ...], int] = {}
        self._factor: dict[tuple[Any, ...], float] = {}
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
        cost_features: Mapping[str, Any] | None = None,
    ) -> float:
        if not self.enabled:
            return 1.0
        key = (
            sequence_length,
            width,
            batch_size,
            conditions,
            cost_feature_signature(cost_features),
        )
        if self._count.get(key, 0) >= self.min_samples:
            return self._factor[key]
        # An unseen shape starts from the measured warmup model. Reusing a
        # global factor across unrelated resolutions can badly distort an
        # extrapolated key before it has produced its first online sample.
        return 1.0

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
        cost_features: Mapping[str, Any] | None = None,
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
            cost_feature_signature(cost_features),
        )
        skipped = self._skipped.get(key, 0)
        if skipped < self.warmup_skip:
            self._skipped[key] = skipped + 1
            return
        predicted = self.model.predict_profile_step_ms(
            sequence_length=key[0],
            width=key[1],
            batch_size=key[2],
            conditions=key[3],
            cost_features=cost_features,
        )
        ratio = measured_step_ms / predicted
        current = self._factor.get(key)
        if current is not None and (
            ratio > current * self.outlier_clip or ratio * self.outlier_clip < current
        ):
            return
        reference = 1.0 if current is None else current
        relative_error = ratio / reference - 1.0
        self._factor[key] = (
            reference
            if abs(relative_error) <= self.tolerance
            else (
                ratio
                if current is None
                else (1.0 - self.alpha) * current + self.alpha * ratio
            )
        )
        self._count[key] = self._count.get(key, 0) + 1
        self._global_log_sum += math.log(ratio)
        self._global_count += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "global_factor": self.global_factor,
            "tolerance": self.tolerance,
            "per_key": {
                ":".join(str(value) for value in key[:4])
                + (f":{key[4]!r}" if key[4] else ""): {
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

    def predict_profile_step_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
        cost_features: Mapping[str, Any] | None = None,
    ) -> float:
        base = self.model.predict_profile_step_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
            cost_features=cost_features,
        )
        return base * self.calibrator.factor(
            sequence_length,
            width,
            batch_size,
            conditions,
            cost_features,
        )

    def predict_terminal_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
    ) -> float:
        return self.model.predict_terminal_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
        )

    def predict_profile_terminal_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
        cost_features: Mapping[str, Any] | None = None,
    ) -> float:
        return self.model.predict_profile_terminal_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
            cost_features=cost_features,
        )

    def predict_completion_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
    ) -> float:
        return self.model.predict_completion_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
        )

    def predict_profile_completion_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
        cost_features: Mapping[str, Any] | None = None,
    ) -> float:
        return self.model.predict_profile_completion_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
            cost_features=cost_features,
        )


class MeasuredTransferCostModel:
    """Conservative state-bundle transfer latency with measured P95 overrides."""

    source = "startup_warmup"

    def __init__(
        self,
        *,
        fallback_bandwidth_gbps: float = 12.0,
        fallback_setup_ms: float = 0.25,
        fallback_per_tensor_ms: float = 0.05,
        fallback_sync_ms: float = 0.25,
    ) -> None:
        if fallback_bandwidth_gbps <= 0:
            raise ValueError("fallback_bandwidth_gbps must be positive")
        if min(
            fallback_setup_ms,
            fallback_per_tensor_ms,
            fallback_sync_ms,
        ) < 0:
            raise ValueError("fallback transfer overheads must be non-negative")
        self.fallback_bandwidth_gbps = float(fallback_bandwidth_gbps)
        self.fallback_setup_ms = float(fallback_setup_ms)
        self.fallback_per_tensor_ms = float(fallback_per_tensor_ms)
        self.fallback_sync_ms = float(fallback_sync_ms)
        self._latency_ms: dict[tuple[int, int, int, int], float] = {}
        self._rows: list[dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return bool(self._latency_ms)

    def initialize(self, rows: Iterable[dict[str, Any]]) -> None:
        latency: dict[tuple[int, int, int], float] = {}
        normalized = []
        for row in rows:
            tensor_count = int(row.get("tensor_count", 1))
            key = (
                int(row["bytes"]),
                int(row["source"]),
                int(row["destination"]),
                tensor_count,
            )
            samples = row.get("samples_ms")
            p95_value = row.get("p95_latency_ms")
            if p95_value is None and isinstance(samples, (list, tuple)) and samples:
                p95_value = _percentile(samples, 0.95)
            value = float(
                row["latency_ms"] if p95_value is None else p95_value
            )
            if (
                key[0] <= 0
                or key[1] < 0
                or key[2] < 0
                or key[3] <= 0
                or value <= 0
            ):
                raise ValueError(f"invalid transfer warmup row: {row}")
            latency[key] = value
            normalized.append(
                {
                    "bytes": key[0],
                    "source": key[1],
                    "destination": key[2],
                    "tensor_count": key[3],
                    "latency_ms": value,
                    "median_latency_ms": float(row.get("latency_ms", value)),
                    "setup_ms": float(row.get("setup_ms", 0.0)),
                    "per_tensor_ms": float(row.get("per_tensor_ms", 0.0)),
                    "sync_ms": float(row.get("sync_ms", 0.0)),
                }
            )
        self._latency_ms = latency
        self._rows = sorted(
            normalized,
            key=lambda row: (
                row["bytes"],
                row["source"],
                row["destination"],
                row["tensor_count"],
            ),
        )

    def predict_transfer_ms(
        self,
        *,
        bytes: int,
        source: int,
        destination: int,
        tensor_count: int = 1,
    ) -> float:
        if source == destination or bytes <= 0:
            return 0.0
        if tensor_count <= 0:
            raise ValueError("tensor_count must be positive")
        if not self._latency_ms:
            payload_ms = int(bytes) * 8.0 / (
                self.fallback_bandwidth_gbps * 1_000_000.0
            )
            return (
                self.fallback_setup_ms
                + self.fallback_sync_ms
                + self.fallback_per_tensor_ms * int(tensor_count)
                + payload_ms
            )
        exact = self._latency_ms.get(
            (int(bytes), int(source), int(destination), int(tensor_count))
        )
        if exact is not None:
            return exact
        candidates = [
            (key, value)
            for key, value in self._latency_ms.items()
            if key[1:3] == (int(source), int(destination))
        ]
        if not candidates:
            candidates = list(self._latency_ms.items())
        key, value = min(
            candidates,
            key=lambda item: (
                abs(item[0][3] - int(tensor_count)),
                abs(item[0][0] - int(bytes)),
            ),
        )
        row = next(
            candidate
            for candidate in self._rows
            if (
                candidate["bytes"],
                candidate["source"],
                candidate["destination"],
                candidate["tensor_count"],
            )
            == key
        )
        fixed_ms = (
            float(row["setup_ms"])
            + float(row["sync_ms"])
            + float(row["per_tensor_ms"]) * int(tensor_count)
        )
        measured_fixed_ms = (
            float(row["setup_ms"])
            + float(row["sync_ms"])
            + float(row["per_tensor_ms"]) * key[3]
        )
        payload_ms = max(0.0, value - measured_fixed_ms)
        return fixed_ms + payload_ms * int(bytes) / key[0]

    def snapshot(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ready": self.ready,
            "fallback": {
                "bandwidth_gbps": self.fallback_bandwidth_gbps,
                "setup_ms": self.fallback_setup_ms,
                "per_tensor_ms": self.fallback_per_tensor_ms,
                "sync_ms": self.fallback_sync_ms,
            },
            "rows": self._rows,
        }
