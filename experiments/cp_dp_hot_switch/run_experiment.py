#!/usr/bin/env python3
"""Run the CP/DP hotswitch simulator across scenario traces and policies.

Loads a shared latency-profile library (traces/profiles.json), then for every
scenario trace file runs all scheduling policies and emits a compact markdown
comparison plus a machine-readable JSON dump.

Usage:
    python3 run_experiment.py                 # default traces/ dir, print markdown
    python3 run_experiment.py --output-dir out # also write markdown + json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from simulate import (  # noqa: E402
    RequestSpec,
    SimulationConfig,
    StepLatencyProfile,
    SwitchCostModel,
    simulate,
)

POLICIES = [
    "static_dp",
    "static_cp",
    "admission_control",
    "early_hot_switch",
    "shape_aware_ratio",
    "oracle",
]

METRIC_COLUMNS = [
    ("throughput_req_per_s", "thrpt(req/s)", True),
    ("mean_latency_ms", "mean(ms)", False),
    ("p95_latency_ms", "p95(ms)", False),
    ("p99_latency_ms", "p99(ms)", False),
    ("mean_queue_delay_ms", "queue(ms)", False),
    ("gpu_busy_ratio", "gpu_busy", True),
    ("switch_count", "switches", None),
]


def load_profiles(path: Path) -> dict[str, StepLatencyProfile]:
    data = json.loads(path.read_text())
    profiles: dict[str, StepLatencyProfile] = {}
    for item in data.get("profiles", []):
        profiles[item["name"]] = StepLatencyProfile(
            name=item["name"],
            dp_1gpu_step_ms=[float(v) for v in item["dp_1gpu_step_ms"]],
            cp_2gpu_step_ms=[float(v) for v in item["cp_2gpu_step_ms"]],
            cp_4gpu_step_ms=[float(v) for v in item["cp_4gpu_step_ms"]] if item.get("cp_4gpu_step_ms") else None,
            seq_len=int(item.get("seq_len", 0)),
            resolution=str(item.get("resolution", "unknown")),
        )
    return profiles


def load_scenario(path: Path) -> tuple[list[RequestSpec], SwitchCostModel, SimulationConfig]:
    data = json.loads(path.read_text())
    requests = [
        RequestSpec(
            request_id=str(item["request_id"]),
            arrival_ms=float(item["arrival_ms"]),
            num_steps=int(item["num_steps"]),
            profile=str(item["profile"]),
            priority=int(item.get("priority", 0)),
            deadline_ms=float(item["deadline_ms"]) if item.get("deadline_ms") is not None else None,
        )
        for item in data.get("requests", [])
    ]
    switch = SwitchCostModel(
        **{k: v for k, v in data.get("switch", {}).items() if k in SwitchCostModel.__dataclass_fields__}
    )
    config_data = data.get("config", {})
    config = SimulationConfig(
        total_gpus=int(config_data.get("total_gpus", 2)),
        admission_wait_ms=float(config_data.get("admission_wait_ms", 50.0)),
        long_request_seq_len=int(config_data.get("long_request_seq_len", 4096)),
        switch=switch,
    )
    return requests, switch, config


def _fmt(value: float, metric: str) -> str:
    if metric == "throughput_req_per_s":
        return f"{value:.3f}"
    if metric == "gpu_busy_ratio":
        return f"{value:.3f}"
    if metric == "switch_count":
        return f"{int(value)}"
    return f"{value:.0f}"


def _best_policy(results: dict[str, dict], metric: str, higher_is_better: bool) -> str | None:
    if higher_is_better is None:
        return None
    scored = [
        (name, res["metrics"][metric])
        for name, res in results.items()
        if res["metrics"].get("num_requests", 0) > 0 and res["metrics"][metric] is not None
    ]
    if not scored:
        return None
    chosen = max(scored, key=lambda kv: kv[1]) if higher_is_better else min(scored, key=lambda kv: kv[1])
    return chosen[0]


def render_scenario_markdown(name: str, config: SimulationConfig, results: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append(f"### {name}")
    lines.append("")
    lines.append(
        f"Config: total_gpus={config.total_gpus}, admission_wait_ms={config.admission_wait_ms:g}, "
        f"long_request_seq_len={config.long_request_seq_len}, switch_total_ms={config.switch.total_ms:g}, "
        f"switch_allowed_until_step={config.switch.switch_allowed_until_step}."
    )
    lines.append("")
    header = "| policy | " + " | ".join(label for _, label, _ in METRIC_COLUMNS) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in METRIC_COLUMNS) + " |"
    lines.append(header)
    lines.append(sep)
    for policy in POLICIES:
        if policy not in results:
            continue
        metrics = results[policy]["metrics"]
        row = [policy]
        for metric, _label, _hib in METRIC_COLUMNS:
            row.append(_fmt(metrics[metric], metric))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    best_bits = []
    for metric, label, hib in METRIC_COLUMNS:
        winner = _best_policy(results, metric, hib)
        if winner is not None:
            best_bits.append(f"{label}->{winner}")
    if best_bits:
        lines.append("Best per metric: " + ", ".join(best_bits) + ".")
        lines.append("")
    return "\n".join(lines)


def run(profiles_path: Path, trace_paths: list[Path]) -> tuple[str, dict]:
    profiles = load_profiles(profiles_path)
    md_sections: list[str] = ["# CP/DP Hotswitch Simulator Sweep", ""]
    md_sections.append(
        "Latency profiles are anchored to measured DiT-forward numbers "
        "(see traces/profiles.json). Arrivals are synthetic. All times in ms unless noted."
    )
    md_sections.append("")
    dump: dict = {"profiles": str(profiles_path), "scenarios": {}}

    for trace_path in trace_paths:
        requests, _switch, config = load_scenario(trace_path)
        scenario_name = trace_path.stem
        results: dict[str, dict] = {}
        for policy in POLICIES:
            result = simulate(requests, profiles, policy, config)
            results[policy] = result.to_jsonable()
        md_sections.append(render_scenario_markdown(scenario_name, config, results))
        dump["scenarios"][scenario_name] = {
            "config": {
                "total_gpus": config.total_gpus,
                "admission_wait_ms": config.admission_wait_ms,
                "long_request_seq_len": config.long_request_seq_len,
                "switch_total_ms": config.switch.total_ms,
            },
            "results": {policy: results[policy]["metrics"] for policy in results},
        }

    return "\n".join(md_sections), dump


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=_HERE / "traces" / "profiles.json")
    parser.add_argument(
        "--traces",
        type=Path,
        nargs="*",
        help="Scenario trace JSON files. Defaults to traces/*.json (excluding profiles.json).",
    )
    parser.add_argument("--output-dir", type=Path, help="If set, write summary.md and summary.json here.")
    args = parser.parse_args()

    if args.traces:
        trace_paths = sorted(args.traces)
    else:
        trace_paths = sorted(
            p for p in (_HERE / "traces").glob("*.json") if p.name != "profiles.json"
        )
    if not trace_paths:
        parser.error("no scenario trace files found")

    markdown, dump = run(args.profiles, trace_paths)
    print(markdown)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.md").write_text(markdown + "\n")
        (args.output_dir / "summary.json").write_text(json.dumps(dump, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
