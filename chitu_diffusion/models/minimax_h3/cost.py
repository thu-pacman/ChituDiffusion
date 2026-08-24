from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ...epac.cost import MeasuredStepCostModel


H3_COST_KIND = "minimax_h3"


@dataclass(frozen=True, slots=True)
class H3CostFeatures:
    text_tokens: int
    condition_tokens: int
    audio_tokens: int
    video_tokens: int
    used_tokens: int
    padded_tokens: int
    pad_tokens: int
    output_type: str = "latent"
    synthetic: bool = False

    def __post_init__(self) -> None:
        counts = (
            self.text_tokens,
            self.condition_tokens,
            self.audio_tokens,
            self.video_tokens,
            self.used_tokens,
            self.padded_tokens,
            self.pad_tokens,
        )
        if any(value < 0 for value in counts):
            raise ValueError("H3 cost token counts must be non-negative")
        if self.used_tokens != (
            self.text_tokens
            + self.condition_tokens
            + self.audio_tokens
            + self.video_tokens
        ):
            raise ValueError("H3 used token count does not match its segments")
        if self.padded_tokens != self.used_tokens + self.pad_tokens:
            raise ValueError("H3 padded token count does not match used plus pad")
        if self.padded_tokens <= 0 or self.padded_tokens % 64:
            raise ValueError("H3 padded token count must be a positive multiple of 64")

    @property
    def attention_work(self) -> int:
        # cu_seqlens=[0, used, padded] makes live and padding separate documents.
        return self.used_tokens**2 + self.pad_tokens**2

    @classmethod
    def from_packed(
        cls,
        packed: Any,
        *,
        output_type: str = "latent",
        synthetic: bool = False,
    ) -> "H3CostFeatures":
        text_tokens = int(packed.text_pos.numel())
        audio_tokens = int(packed.audio_pos.numel())
        video_tokens = int(packed.target_img_pos.numel())
        condition_tokens = int(packed.img_pos.numel()) - video_tokens
        used_tokens = int(packed.used_length)
        padded_tokens = int(packed.sequence_length)
        return cls(
            text_tokens=text_tokens,
            condition_tokens=condition_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
            used_tokens=used_tokens,
            padded_tokens=padded_tokens,
            pad_tokens=padded_tokens - used_tokens,
            output_type=str(output_type),
            synthetic=bool(synthetic),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "H3CostFeatures":
        if raw.get("kind") != H3_COST_KIND:
            raise ValueError("cost features are not MiniMax-H3 features")
        return cls(
            text_tokens=int(raw["text_tokens"]),
            condition_tokens=int(raw["condition_tokens"]),
            audio_tokens=int(raw["audio_tokens"]),
            video_tokens=int(raw["video_tokens"]),
            used_tokens=int(raw["used_tokens"]),
            padded_tokens=int(raw["padded_tokens"]),
            pad_tokens=int(raw["pad_tokens"]),
            output_type=str(raw.get("output_type", "latent")),
            synthetic=bool(raw.get("synthetic", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": H3_COST_KIND,
            "text_tokens": self.text_tokens,
            "condition_tokens": self.condition_tokens,
            "audio_tokens": self.audio_tokens,
            "video_tokens": self.video_tokens,
            "used_tokens": self.used_tokens,
            "padded_tokens": self.padded_tokens,
            "pad_tokens": self.pad_tokens,
            "attention_work": self.attention_work,
            "output_type": self.output_type,
            "synthetic": self.synthetic,
        }


class MiniMaxH3StepCostModel(MeasuredStepCostModel):
    """H3 step latency as a function of sequence length and discrete CP width."""

    source = "minimax_h3_profile"

    def __init__(self) -> None:
        super().__init__()
        self._h3_rows: list[tuple[H3CostFeatures, int, int, int, float]] = []
        self._h3_tail_rows: list[
            tuple[H3CostFeatures, int, int, int, float, float]
        ] = []
        self._sequence_rows: list[dict[str, Any]] = []
        self._fits: dict[tuple[int, int, int], dict[str, Any]] = {}

    def initialize(self, rows) -> None:
        materialized = [dict(row) for row in rows]
        super().initialize(materialized)
        parsed: list[tuple[H3CostFeatures, int, int, int, float]] = []
        tails: list[tuple[H3CostFeatures, int, int, int, float, float]] = []
        for row in materialized:
            raw = row.get("cost_features")
            if not isinstance(raw, Mapping) or raw.get("kind") != H3_COST_KIND:
                continue
            parsed.append(
                (
                    H3CostFeatures.from_mapping(raw),
                    int(row["width"]),
                    int(row.get("batch_size", 1)),
                    int(row.get("conditions", 1)),
                    float(row["latency_ms"]),
                )
            )
            if "terminal_ms" in row or "completion_tail_ms" in row:
                terminal = float(row.get("terminal_ms", 0.0))
                completion = float(row.get("completion_tail_ms", terminal))
                tails.append(
                    (
                        H3CostFeatures.from_mapping(raw),
                        int(row["width"]),
                        int(row.get("batch_size", 1)),
                        int(row.get("conditions", 1)),
                        terminal,
                        completion,
                    )
                )
        self._h3_rows = parsed
        self._h3_tail_rows = tails
        self._sequence_rows = [
            row
            for row in materialized
            if isinstance(row.get("cost_features"), Mapping)
            and row["cost_features"].get("kind") == H3_COST_KIND
        ]
        self._fit()

    def _fit(self) -> None:
        fits: dict[tuple[int, int, int], dict[str, Any]] = {}
        groups = {
            (
                int(row["width"]),
                int(row.get("batch_size", 1)),
                int(row.get("conditions", 1)),
            )
            for row in self._sequence_rows
        }
        for group in groups:
            rows = [
                row
                for row in self._sequence_rows
                if (
                    int(row["width"]),
                    int(row.get("batch_size", 1)),
                    int(row.get("conditions", 1)),
                )
                == group
            ]
            if len(rows) < 2:
                continue
            training = [row for row in rows if row.get("split") != "validation"]
            validation = [row for row in rows if row.get("split") == "validation"]
            if not validation:
                validation = rows[:: max(2, len(rows) // 3)]
                validation_ids = {id(row) for row in validation}
                training = [row for row in rows if id(row) not in validation_ids]
            candidates: dict[str, dict[str, Any]] = {}
            for degree in (1, 2):
                if len(training) < degree + 1:
                    continue
                coefficients = self._fit_polynomial(training, degree)
                predictions = [
                    self._polynomial(coefficients, int(row["sequence_length"]))
                    for row in validation
                ]
                candidates[f"polynomial_{degree}"] = {
                    "kind": "polynomial",
                    "degree": degree,
                    "validation": self._validation_metrics(validation, predictions),
                }
            if len(training) >= 2:
                points = self._points(training)
                predictions = [
                    self._piecewise(points, int(row["sequence_length"]))
                    for row in validation
                ]
                candidates["piecewise"] = {
                    "kind": "piecewise",
                    "validation": self._validation_metrics(validation, predictions),
                }
            if not candidates:
                continue
            selected = min(
                candidates.values(),
                key=lambda item: (
                    item["validation"]["p95_relative_error"],
                    item["validation"]["max_underestimate"],
                    item["validation"]["mape"],
                ),
            )
            fitted = dict(selected)
            fitted["candidates"] = candidates
            fitted["points"] = self._points(rows)
            if fitted["kind"] == "polynomial":
                fitted["coefficients"] = self._fit_polynomial(
                    rows, int(fitted["degree"])
                )
            fitted["safety_factor"] = 1.0 + min(
                0.25,
                float(fitted["validation"]["max_underestimate"]),
            )
            fits[group] = fitted
        self._fits = fits

    @staticmethod
    def _points(rows: list[dict[str, Any]]) -> tuple[tuple[int, float], ...]:
        by_length: dict[int, list[float]] = {}
        for row in rows:
            by_length.setdefault(int(row["sequence_length"]), []).append(
                float(row["latency_ms"])
            )
        return tuple(
            (length, float(np.median(values)))
            for length, values in sorted(by_length.items())
        )

    @staticmethod
    def _fit_polynomial(
        rows: list[dict[str, Any]], degree: int
    ) -> tuple[float, ...]:
        lengths = np.asarray(
            [float(row["sequence_length"]) / 10_000.0 for row in rows],
            dtype=np.float64,
        )
        design = np.stack(
            [lengths**power for power in range(degree + 1)], axis=1
        )
        target = np.asarray(
            [float(row["latency_ms"]) for row in rows], dtype=np.float64
        )
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return tuple(float(value) for value in coefficients)

    @staticmethod
    def _polynomial(coefficients: tuple[float, ...], length: int) -> float:
        normalized = float(length) / 10_000.0
        return float(
            sum(
                coefficient * normalized**power
                for power, coefficient in enumerate(coefficients)
            )
        )

    @staticmethod
    def _piecewise(points: tuple[tuple[int, float], ...], length: int) -> float:
        if len(points) == 1:
            return points[0][1]
        left, right = points[0], points[1]
        if length >= points[-1][0]:
            left, right = points[-2], points[-1]
        else:
            for candidate_left, candidate_right in zip(points, points[1:]):
                if candidate_left[0] <= length <= candidate_right[0]:
                    left, right = candidate_left, candidate_right
                    break
        if right[0] == left[0]:
            return left[1]
        ratio = (length - left[0]) / (right[0] - left[0])
        return float(left[1] + ratio * (right[1] - left[1]))

    @staticmethod
    def _validation_metrics(
        rows: list[dict[str, Any]], predictions: list[float]
    ) -> dict[str, float]:
        relative = [
            (prediction - float(row["latency_ms"])) / float(row["latency_ms"])
            for row, prediction in zip(rows, predictions)
        ]
        absolute = [abs(value) for value in relative]
        return {
            "count": float(len(rows)),
            "mape": float(np.mean(absolute)),
            "p95_relative_error": float(np.percentile(absolute, 95)),
            "max_underestimate": abs(min(0.0, min(relative))),
        }

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
        length = sequence_length if sequence_length is not None else image_tokens
        selected_conditions = (
            conditions if conditions is not None else cfg_conditions
        )
        if selected_conditions is None:
            selected_conditions = 1
        if length is not None:
            fit = self._fits.get(
                (int(width), int(batch_size), int(selected_conditions))
            )
            if fit is not None:
                exact = [
                    latency
                    for candidate_length, latency in fit["points"]
                    if candidate_length == int(length)
                ]
                if exact:
                    return float(exact[0])
                if fit["kind"] == "polynomial":
                    predicted = self._polynomial(
                        fit["coefficients"], int(length)
                    )
                else:
                    predicted = self._piecewise(fit["points"], int(length))
                return max(1e-6, float(predicted) * fit["safety_factor"])
        return super().predict_step_ms(
            sequence_length=sequence_length,
            image_tokens=image_tokens,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
            cfg_conditions=cfg_conditions,
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
        del cost_features
        return self.predict_step_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
        )

    def _predict_profile_tail(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int,
        conditions: int,
        cost_features: Mapping[str, Any] | None,
        completion: bool,
    ) -> float | None:
        if not cost_features or cost_features.get("kind") != H3_COST_KIND:
            return None
        features = H3CostFeatures.from_mapping(cost_features)
        candidates = [
            (candidate, complete if completion else terminal)
            for (
                candidate,
                candidate_width,
                candidate_batch,
                candidate_conditions,
                terminal,
                complete,
            ) in self._h3_tail_rows
            if candidate_width == int(width)
            and candidate_batch == int(batch_size)
            and candidate_conditions == int(conditions)
            and candidate.output_type == features.output_type
            and candidate.synthetic == features.synthetic
        ]
        if not candidates:
            return None
        candidate, latency = min(
            candidates,
            key=lambda item: (
                abs(item[0].video_tokens - features.video_tokens),
                abs(item[0].audio_tokens - features.audio_tokens),
                abs(item[0].padded_tokens - features.padded_tokens),
            ),
        )
        if candidate.padded_tokens <= 0:
            return float(latency)
        return float(latency * features.padded_tokens / candidate.padded_tokens)

    def predict_profile_terminal_ms(
        self,
        *,
        sequence_length: int,
        width: int,
        batch_size: int = 1,
        conditions: int = 1,
        cost_features: Mapping[str, Any] | None = None,
    ) -> float:
        predicted = self._predict_profile_tail(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
            cost_features=cost_features,
            completion=False,
        )
        if predicted is not None:
            return predicted
        return super().predict_terminal_ms(
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
        predicted = self._predict_profile_tail(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
            cost_features=cost_features,
            completion=True,
        )
        if predicted is not None:
            return predicted
        return super().predict_completion_ms(
            sequence_length=sequence_length,
            width=width,
            batch_size=batch_size,
            conditions=conditions,
        )

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        snapshot.update(
            {
                "source": self.source,
                "h3_rows": len(self._h3_rows),
                "fits": {
                    ":".join(str(value) for value in key): fit
                    for key, fit in self._fits.items()
                },
            }
        )
        return snapshot
