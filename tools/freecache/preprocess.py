"""CPU fitting and exact reproduction of the FreeCache v2 budget tables.

Run from a source checkout: python -m tools.freecache.preprocess --help.
The historical objective uses terminal deviations, not normalized gains.
It is an empirical proxy, not a terminal-error bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).with_name("data")
COEFFICIENTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_setting(setting):
    """Compare semantic settings, not the order of Diffusers' default-key set."""
    result = json.loads(json.dumps(setting))
    config = result.get("scheduler_config", {})
    if "_use_default_values" in config:
        config["_use_default_values"] = sorted(config["_use_default_values"])
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def propagation_weights(path, steps):
    """Reproduce v2: mean final_deviation, normalized and log-interpolated.

    Deliberately do not replace final_deviation with gain: doing so changes the
    fitted method. End slopes are extrapolated, including steps before knot 4.
    """
    samples = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            i, value = int(row["inject_step"]), float(row["final_deviation"])
            if not 0 <= i < steps or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "propagation rows require valid steps and positive finite deviations"
                )
            samples.setdefault(i, []).append(value)
    if len(samples) < 2:
        raise ValueError("at least two propagation positions are required")
    averaged = {i: sum(values) / len(values) for i, values in samples.items()}
    scale = sum(averaged.values()) / len(averaged)
    knots = sorted((i, math.log(v / scale)) for i, v in averaged.items())
    weights = []
    for step in range(steps):
        if step <= knots[0][0]:
            (x0, y0), (x1, y1) = knots[:2]
        elif step >= knots[-1][0]:
            (x0, y0), (x1, y1) = knots[-2:]
        else:
            (x0, y0), (x1, y1) = next(
                (a, b) for a, b in zip(knots, knots[1:]) if a[0] <= step <= b[0]
            )
        weights.append(math.exp(y0 + (y1 - y0) * (step - x0) / (x1 - x0)))
    return np.asarray(weights)


class Objective:
    """Pooled, displacement-normalized Gram objective from the v2 compiler."""

    def __init__(self, traces, *, coherence, weights):
        if not traces or not 0 <= coherence <= 1:
            raise ValueError("nonempty traces and coherence in [0,1] required")
        self.sigmas = np.asarray(traces[0]["sigmas"], dtype=np.float64)
        self.steps = len(traces[0]["gram"])
        if self.sigmas.shape != (self.steps + 1,) or not np.isfinite(self.sigmas).all():
            raise ValueError("expected N+1 finite sigmas")
        self.spans = np.diff(self.sigmas)
        if np.any(self.spans > 0) or not np.any(self.spans < 0):
            raise ValueError(
                "expected a nonincreasing Euler sigma grid with nonzero span"
            )
        self.weights = np.asarray(weights, dtype=np.float64)
        if (
            self.weights.shape != (self.steps,)
            or not np.isfinite(self.weights).all()
            or np.any(self.weights <= 0)
        ):
            raise ValueError("expected N positive finite propagation weights")
        self.coherence = coherence
        pooled = np.zeros((self.steps, self.steps))
        for trace in traces:
            if not np.array_equal(trace["sigmas"], self.sigmas):
                raise ValueError(
                    "calibration trajectories must share one exact sigma grid"
                )
            gram = np.asarray(trace["gram"], dtype=np.float64)
            if gram.shape != pooled.shape or not np.isfinite(gram).all():
                raise ValueError("invalid velocity Gram matrix")
            if not np.allclose(gram, gram.T, rtol=1e-5, atol=1e-7):
                raise ValueError("velocity Gram matrix must be symmetric")
            # Historical matrices were accumulated in float32. Do not silently
            # project them onto a different PSD matrix during reproduction.
            pooled += gram / max(self.spans @ gram @ self.spans, 1e-24)
        self.gram = pooled / len(traces)

    def __call__(self, fresh, coefficient):
        fresh = tuple(fresh)
        if (
            not fresh
            or fresh[0] != 0
            or fresh != tuple(sorted(set(fresh)))
            or fresh[-1] >= self.steps
        ):
            raise ValueError("Fresh steps must be sorted, unique and start at zero")
        if not math.isfinite(coefficient) or not 0 <= coefficient <= 1:
            raise ValueError("coefficient must be in [0,1]")
        anchors = {}
        for pos, anchor in enumerate(fresh):
            previous = fresh[max(0, pos - 1)]
            end = fresh[pos + 1] if pos + 1 < len(fresh) else self.steps
            anchors.update({i: (previous, anchor) for i in range(anchor + 1, end)})
        if not anchors:
            return 0.0
        reuse = np.asarray(sorted(anchors), dtype=np.int64)
        previous = np.asarray([anchors[i][0] for i in reuse])
        anchor = np.asarray([anchors[i][1] for i in reuse])
        span = self.sigmas[anchor] - self.sigmas[previous]
        alpha = np.divide(
            coefficient * (self.sigmas[reuse] - self.sigmas[anchor]),
            span,
            out=np.zeros(len(reuse)),
            where=np.abs(span) >= 1e-12,
        )
        rows = np.arange(len(reuse))
        coefficients = np.zeros((len(reuse), self.steps))
        np.add.at(coefficients, (rows, anchor), 1.0 + alpha)
        np.add.at(coefficients, (rows, previous), -alpha)
        np.add.at(coefficients, (rows, reuse), -1.0)
        scale = self.spans[reuse] * self.weights[reuse]
        retention = self.coherence ** np.abs(reuse[:, None] - reuse[None, :])
        cross = coefficients @ self.gram @ coefficients.T
        return float(
            np.sqrt(max(float((retention * np.outer(scale, scale) * cross).sum()), 0.0))
        )


def compile_ranking(objective, *, warmup):
    """Greedy node deletion. Prefix is explicit; no implicit warmup selection."""
    n = objective.steps
    if not 2 <= warmup <= n:
        raise ValueError("warmup must lie in [2,N]")
    rankings, curves = [], []
    for coefficient in COEFFICIENTS:
        current = list(range(n))
        removed, scores = [], {n: 0.0}
        while len(current) > warmup:
            score, drop = min(
                (
                    objective(tuple(i for i in current if i != candidate), coefficient),
                    candidate,
                )
                for candidate in current[warmup:]
            )
            current.remove(drop)
            removed.append(drop)
            scores[len(current)] = score
        rankings.append([*range(warmup), *reversed(removed)])
        curves.append(scores)
    choices = [None] * (n + 1)
    if warmup == 2:
        choices[0] = choices[1] = 0  # v2 F1 is explicitly ZOH
    for budget in range(warmup, n):
        choices[budget] = min(range(len(COEFFICIENTS)), key=lambda c: curves[c][budget])
    choices[n] = 0
    return {"rankings": rankings, "choices": choices}


def profile(family, budget, compiled, *, prefix="candidate"):
    choice = compiled["choices"][budget]
    if choice is None:
        raise ValueError("budget cannot be smaller than the required warmup")
    return {
        "profile_id": f"{prefix}-{family}-f{budget}",
        "model_family": family,
        "reference_steps": len(compiled["rankings"][choice]),
        "fresh_steps": sorted(compiled["rankings"][choice][:budget]),
        "proposal_coefficient": COEFFICIENTS[choice],
    }


def reproduce(data=DATA):
    """Verify canonical inputs, rebuild extension, apply historical overrides."""
    manifest = json.loads((data / "manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        if sha256(data / relative) != expected:
            raise ValueError(f"canonical input hash mismatch: {relative}")
    compiled, profiles = {}, {}
    presets = json.loads((data / "selected_presets.json").read_text())
    for family, config in manifest["models"].items():
        traces = json.loads((data / config["traces"]).read_text())["traces"]
        objective = Objective(
            traces,
            coherence=config["coherence"],
            weights=propagation_weights(data / config["propagation"], 50),
        )
        compiled[family] = compile_ranking(objective, warmup=2)
        profiles[family] = []
        for budget in range(1, 51):
            record = profile(
                family,
                budget,
                compiled[family],
                prefix="freecache-v2-preview-20260909-budget-v1",
            )
            if str(budget) in presets[family]:
                record = presets[family][str(budget)]
            profiles[family].append(record)
    return {
        "compiled": compiled,
        "profiles": profiles,
        "input_manifest_sha256": sha256(data / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser(
        "reproduce", help="CPU-only reconstruction of all 150 released profiles"
    )
    replay.add_argument("--data", type=Path, default=DATA)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--check-runtime", action="store_true")
    fit = commands.add_parser(
        "fit", help="fit experimental candidates from complete collection outputs"
    )
    fit.add_argument("--traces", type=Path, required=True)
    fit.add_argument(
        "--propagation",
        type=Path,
        required=True,
        help="directory containing propagation.csv and protocol.json",
    )
    fit.add_argument("--coherence", type=float, required=True)
    fit.add_argument(
        "--warmup",
        type=int,
        required=True,
        help="consecutive Fresh prefix, charged to budget",
    )
    fit.add_argument("--budgets", type=int, nargs="+", required=True)
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reproduce":
        result = reproduce(args.data)
        if args.check_runtime:
            from chitu_diffusion.flexcache.budget_profiles import BUDGET_PROFILES
            from chitu_diffusion.flexcache.presets import preview_profile

            if result["compiled"] != json.loads(json.dumps(BUDGET_PROFILES)):
                raise ValueError("compiled rankings differ from installed runtime")
            from dataclasses import asdict

            for family, records in result["profiles"].items():
                for budget, record in enumerate(records, 1):
                    if record != json.loads(
                        json.dumps(asdict(preview_profile(family, budget)))
                    ):
                        raise ValueError(
                            f"runtime profile mismatch: {family} F{budget}"
                        )
            result["runtime_checked"] = 150
    else:
        payload = json.loads(args.traces.read_text())
        protocol = json.loads((args.propagation / "protocol.json").read_text())
        if payload.get("complete") is not True or protocol.get("complete") is not True:
            raise ValueError("fit requires completed trace and propagation collections")
        if (
            normalized_setting(payload["protocol"]["setting"])
            != normalized_setting(protocol["setting"])
            or payload["protocol"]["sigmas"] != protocol["sigmas"]
        ):
            raise ValueError("trace and propagation model/grid/settings differ")
        steps = payload["steps"]
        if type(steps) is not int or steps < 1:
            parser.error("trace steps must be a positive integer")
        if len(set(args.budgets)) != len(args.budgets):
            parser.error("budgets must be unique")
        if any(b < args.warmup or b > steps for b in args.budgets):
            parser.error("every budget must be in [warmup,steps]")
        propagation = args.propagation / "propagation.csv"
        objective = Objective(
            payload["traces"],
            coherence=args.coherence,
            weights=propagation_weights(propagation, steps),
        )
        if objective.steps != steps:
            raise ValueError("trace step count does not match its metadata")
        compiled = compile_ranking(objective, warmup=args.warmup)
        result = {
            "status": "candidate_requires_heldout_validation",
            "fitter_sha256": sha256(Path(__file__)),
            "protocol": payload["protocol"],
            "warmup": args.warmup,
            "coherence": args.coherence,
            "weight_semantics": "normalized_final_deviation_legacy_v2",
            "sha256": {
                "traces": sha256(args.traces),
                "propagation": sha256(propagation),
                "propagation_protocol": sha256(args.propagation / "protocol.json"),
            },
            "profiles": [
                profile(payload["model_family"], b, compiled) for b in args.budgets
            ],
            "compiled": compiled,
        }
    write_json(args.output, result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
