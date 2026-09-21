#!/usr/bin/env python3
"""Plot native Fast CP latency across sequence and model workloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

CP_DEGREES = (2, 4, 8)
MODEL_DIRECTORIES = {
    "wan1.3b": "wan1.3b-scaling-node021",
    "wan14b": "wan14b-scaling-node040",
    "qwenimage": "qwenimage-scaling-node021",
    "zimage": "zimage-scaling-node021",
}
MODEL_SHAPES = {
    "wan1.3b": {
        "global_sequence": 32760,
        "heads": 12,
        "head_dim": 128,
    },
    "wan14b": {
        "global_sequence": 75600,
        "heads": 40,
        "head_dim": 128,
    },
    "qwenimage": {
        "global_image_sequence": 4096,
        "text_sequence": 512,
        "heads": 24,
        "head_dim": 128,
    },
    "zimage": {
        "global_image_sequence": 4096,
        "text_sequence": 512,
        "heads": 30,
        "head_dim": 128,
    },
}
NATIVE_SEQUENCE_WORKLOADS = (
    (4096, "S=4K"),
    (8192, "S=8K"),
    (16384, "S=16K"),
    (32768, "S=32K"),
)
NATIVE_WORKLOADS = NATIVE_SEQUENCE_WORKLOADS + (
    ("wan1.3b", "Wan 1.3B\n1x32760x12x128"),
    ("wan14b", "Wan 14B\n1x75600x40x128"),
    ("qwenimage", "Qwen-Image\n1x4608x24x128\n(4096I + 512T)"),
    ("zimage", "Z-Image*\n1x4608x30x128\n(4096I + 512T)"),
)
NATIVE_METHODS = (
    ("chitu", "fast_agkv", "Fast AGKV"),
    ("chitu", "fast_ulysses", "Fast Ulysses"),
    ("xdit", "ulysses", "xDiT Ulysses"),
    ("xdit", "usp", "xDiT USP"),
    ("sglang", "kv_gather", "SGLang KV gather"),
    ("sglang", "ulysses", "SGLang Ulysses"),
)
NATIVE_METHOD_STYLES = {
    "chitu:fast_agkv": ("#1565c0", "o", "-"),
    "chitu:fast_ulysses": ("#d97706", "s", "-"),
    "xdit:ulysses": ("#7e22ce", "D", ":"),
    "xdit:usp": ("#dc2626", "v", "--"),
    "sglang:kv_gather": ("#0f766e", "P", "-."),
    "sglang:ulysses": ("#52525b", "X", "-."),
}


def _matches_model_shape(model: str, shape: dict[str, Any]) -> bool:
    for name, expected in MODEL_SHAPES[model].items():
        actual = shape.get(name)
        if name == "global_sequence" and actual is None:
            actual = shape.get("global_image_sequence")
        if actual != expected:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/fast-cp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fast-cp/figures/workload-radar"),
    )
    parser.add_argument(
        "--native-sweep-dir",
        type=Path,
        help="Complete native QKV-to-O JSON sweep used for baseline polygons",
    )
    parser.add_argument(
        "--xdit-model-dir",
        type=Path,
        help="Native xDiT model-shaped sweep directory",
    )
    parser.add_argument(
        "--xdit-wan-dir",
        type=Path,
        help="Corrected native xDiT Wan sweep directory",
    )
    parser.add_argument(
        "--sglang-model-dir",
        type=Path,
        help="Native SGLang model-shaped sweep directory",
    )
    parser.add_argument(
        "--sglang-wan-dir",
        type=Path,
        help="Corrected native SGLang Wan sweep directory",
    )
    parser.add_argument(
        "--sglang-qwen-dir",
        type=Path,
        help="Same-node native SGLang Qwen-Image sweep directory",
    )
    parser.add_argument(
        "--chitu-joint-dir",
        type=Path,
        help="Updated Chitu Qwen-Image and Z-Image sweep directory",
    )
    parser.add_argument(
        "--chitu-zimage-dir",
        type=Path,
        help="Updated Chitu Z-Image sweep directory; overrides --chitu-joint-dir",
    )
    parser.add_argument(
        "--chitu-qwen-dir",
        type=Path,
        help="Same-node Chitu Qwen-Image sweep directory; overrides --chitu-joint-dir",
    )
    parser.add_argument(
        "--chitu-wan-dir",
        type=Path,
        help="Updated Chitu Wan 1.3B and Wan 14B sweep directory",
    )
    return parser.parse_args()


def load_native_sequence_latencies(
    directory: Path,
) -> dict[int, dict[int, dict[str, float]]]:
    latencies: dict[int, dict[int, dict[str, float]]] = {
        cp_degree: {} for cp_degree in CP_DEGREES
    }
    for path in sorted(directory.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if report.get("status") != "complete":
            continue
        shape = report.get("shape", {})
        cp_degree = int(shape.get("cp_degree", 0))
        sequence = int(shape.get("global_sequence", 0))
        if cp_degree not in CP_DEGREES or sequence not in dict(
            NATIVE_SEQUENCE_WORKLOADS
        ):
            continue
        method_key = f"{report['framework']}:{report['method']}"
        latencies[cp_degree].setdefault(sequence, {})[method_key] = float(
            report["timing"]["median_ms"]
        )

    for cp_degree in CP_DEGREES:
        for sequence, _ in NATIVE_SEQUENCE_WORKLOADS:
            missing = {
                f"{framework}:{method}"
                for framework, method, _ in NATIVE_METHODS
            } - latencies[cp_degree].get(sequence, {}).keys()
            if missing:
                raise RuntimeError(
                    f"{directory}: missing CP{cp_degree} S={sequence} latency cells: "
                    + ", ".join(sorted(missing))
                )
    return latencies


def load_native_model_latencies(
    results_root: Path,
    xdit_directory: Path,
    sglang_directory: Path,
    xdit_wan_directory: Path | None = None,
    sglang_wan_directory: Path | None = None,
    sglang_qwen_directory: Path | None = None,
    chitu_joint_directory: Path | None = None,
    chitu_zimage_directory: Path | None = None,
    chitu_qwen_directory: Path | None = None,
    chitu_wan_directory: Path | None = None,
) -> tuple[
    dict[int, dict[str, dict[str, float]]],
    dict[int, dict[str, dict[str, str]]],
] | None:
    reports: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    paths: list[tuple[Path, str | None]] = []
    for model, directory in MODEL_DIRECTORIES.items():
        if model.startswith("wan") and chitu_wan_directory is not None:
            chitu_directory = chitu_wan_directory
        elif model == "qwenimage" and chitu_qwen_directory is not None:
            chitu_directory = chitu_qwen_directory
        elif model == "zimage" and chitu_zimage_directory is not None:
            chitu_directory = chitu_zimage_directory
        elif not model.startswith("wan") and chitu_joint_directory is not None:
            chitu_directory = chitu_joint_directory
        else:
            chitu_directory = results_root / directory
        paths.extend((path, model) for path in sorted(chitu_directory.glob("*.json")))
    paths.extend((path, None) for path in sorted(xdit_directory.glob("*.json")))
    paths.extend(
        (path, None)
        for path in sorted(sglang_directory.glob("*.json"))
        if sglang_qwen_directory is None or "qwenimage" not in path.name
    )
    if xdit_wan_directory is not None:
        paths.extend(
            (path, None) for path in sorted(xdit_wan_directory.glob("*.json"))
        )
    if sglang_wan_directory is not None:
        paths.extend(
            (path, None) for path in sorted(sglang_wan_directory.glob("*.json"))
        )
    if sglang_qwen_directory is not None:
        paths.extend(
            (path, "qwenimage")
            for path in sorted(sglang_qwen_directory.glob("*.json"))
        )

    for path, expected_model in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if report.get("status") not in {"complete", "unsupported"}:
            continue
        if report.get("model") not in MODEL_DIRECTORIES:
            continue
        if expected_model is not None and report.get("model") != expected_model:
            continue
        if report.get("implementation", {}).get("schedule", "native") != "native":
            continue
        shape = report.get("shape", {})
        if not _matches_model_shape(str(report["model"]), shape):
            continue
        key = (
            str(report["framework"]),
            str(report["model"]),
            str(report["method"]),
            int(shape["cp_degree"]),
        )
        if key in reports:
            raise RuntimeError(f"duplicate native model result {key}")
        reports[key] = report

    expected: set[tuple[str, str, str, int]] = set()
    for model in MODEL_DIRECTORIES:
        for cp_degree in CP_DEGREES:
            for method in ("fast_agkv", "fast_ulysses"):
                expected.add(("chitu", model, method, cp_degree))
            expected.add(("xdit", model, "ulysses", cp_degree))
            expected.add(("sglang", model, "kv_gather", cp_degree))
            expected.add(("sglang", model, "ulysses", cp_degree))
    missing = expected - reports.keys()
    if missing:
        print(
            f"Native model radar pending: {len(expected) - len(missing)}/{len(expected)} "
            "terminal cells"
        )
        return None

    latencies: dict[int, dict[str, dict[str, float]]] = {
        cp_degree: {} for cp_degree in CP_DEGREES
    }
    statuses: dict[int, dict[str, dict[str, str]]] = {
        cp_degree: {} for cp_degree in CP_DEGREES
    }
    for model in MODEL_DIRECTORIES:
        for cp_degree in CP_DEGREES:
            latencies[cp_degree][model] = {}
            statuses[cp_degree][model] = {}
            for framework, method, _ in NATIVE_METHODS:
                method_key = f"{framework}:{method}"
                if (framework, method) == ("xdit", "usp"):
                    statuses[cp_degree][model][method_key] = "not_implemented"
                    latencies[cp_degree][model][method_key] = math.nan
                    continue
                report = reports[(framework, model, method, cp_degree)]
                status = str(report["status"])
                statuses[cp_degree][model][method_key] = status
                latencies[cp_degree][model][method_key] = (
                    float(report["timing"]["median_ms"])
                    if status == "complete"
                    else math.nan
                )
    return latencies, statuses


def save_native_workload_latency_radar(
    output_dir: Path,
    sequence_latencies: dict[int, dict[int, dict[str, float]]],
    model_latencies: dict[int, dict[str, dict[str, float]]],
    model_statuses: dict[int, dict[str, dict[str, str]]],
) -> None:
    count = len(NATIVE_WORKLOADS)
    angles = [2 * math.pi * index / count for index in range(count)]
    closed_angles = angles + [angles[0]]
    all_latencies: dict[int, dict[object, dict[str, float]]] = {}
    for cp_degree in CP_DEGREES:
        all_latencies[cp_degree] = {
            **sequence_latencies[cp_degree],
            **model_latencies[cp_degree],
        }

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(22.0, 8.2),
        subplot_kw={"polar": True},
    )
    for axis, cp_degree in zip(axes, CP_DEGREES):
        axis.set_theta_offset(math.pi / 2)
        axis.set_theta_direction(-1)
        axis.set_xticks(angles, [label for _, label in NATIVE_WORKLOADS])
        axis.tick_params(axis="x", labelsize=8.5, pad=21)
        finite = [
            value
            for methods in all_latencies[cp_degree].values()
            for value in methods.values()
            if math.isfinite(value)
        ]
        lower = 10 ** math.floor(math.log10(min(finite)))
        upper = 10 ** math.ceil(math.log10(max(finite)))
        axis.set_rscale("log")
        axis.set_ylim(lower, upper)
        ticks = []
        tick = lower
        while tick <= upper:
            ticks.append(tick)
            tick *= 10
        axis.set_yticks(ticks, [f"{value:g} ms" for value in ticks])
        axis.set_rlabel_position(22.5)
        axis.tick_params(axis="y", labelsize=8.5, colors="#5f6368")
        axis.grid(color="#b9c0c8", alpha=0.55, linewidth=0.8)
        axis.spines["polar"].set_color("#9aa0a6")
        axis.set_title(f"CP{cp_degree}", fontsize=16, fontweight="bold", pad=17)

        for framework, method, label in NATIVE_METHODS:
            method_key = f"{framework}:{method}"
            values = [
                all_latencies[cp_degree][workload][method_key]
                for workload, _ in NATIVE_WORKLOADS
            ]
            color, marker, linestyle = NATIVE_METHOD_STYLES[method_key]
            axis.plot(
                closed_angles,
                values + [values[0]],
                color=color,
                marker=marker,
                markersize=4.3,
                linewidth=1.8,
                linestyle=linestyle,
                label=label,
            )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    figure.suptitle(
        "Native CP latency across sequence lengths and models",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.009,
        "Radius is end-to-end latency in milliseconds on a logarithmic scale; smaller is faster; "
        "model shapes are global QKV BxSxHxD; I/T denote image/text tokens; gaps are unsupported; "
        "* Chitu/xDiT shard Z-Image text while SGLang replicates it",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    figure.subplots_adjust(top=0.78, bottom=0.2, left=0.02, right=0.98, wspace=0.25)
    for extension in ("png", "svg"):
        figure.savefig(
            output_dir / f"native-latency-workload-radar-all.{extension}",
            dpi=210,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)

    for cp_degree in CP_DEGREES:
        single, axis = plt.subplots(
            figsize=(9.5, 9.0),
            subplot_kw={"polar": True},
        )
        axis.set_theta_offset(math.pi / 2)
        axis.set_theta_direction(-1)
        axis.set_xticks(angles, [label for _, label in NATIVE_WORKLOADS])
        axis.tick_params(axis="x", labelsize=9.5, pad=29)
        finite = [
            value
            for methods in all_latencies[cp_degree].values()
            for value in methods.values()
            if math.isfinite(value)
        ]
        lower = 10 ** math.floor(math.log10(min(finite)))
        upper = 10 ** math.ceil(math.log10(max(finite)))
        axis.set_rscale("log")
        axis.set_ylim(lower, upper)
        ticks = []
        tick = lower
        while tick <= upper:
            ticks.append(tick)
            tick *= 10
        axis.set_yticks(ticks, [f"{value:g} ms" for value in ticks])
        axis.set_rlabel_position(22.5)
        axis.grid(color="#b9c0c8", alpha=0.55, linewidth=0.8)
        axis.spines["polar"].set_color("#9aa0a6")
        for framework, method, label in NATIVE_METHODS:
            method_key = f"{framework}:{method}"
            values = [
                all_latencies[cp_degree][workload][method_key]
                for workload, _ in NATIVE_WORKLOADS
            ]
            color, marker, linestyle = NATIVE_METHOD_STYLES[method_key]
            axis.plot(
                closed_angles,
                values + [values[0]],
                color=color,
                marker=marker,
                markersize=4.8,
                linewidth=1.9,
                linestyle=linestyle,
                label=label,
            )
        axis.set_title(
            f"CP{cp_degree} native latency",
            fontsize=17,
            fontweight="bold",
            pad=23,
        )
        single.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.09),
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        single.subplots_adjust(top=0.88, bottom=0.16, left=0.08, right=0.92)
        for extension in ("png", "svg"):
            single.savefig(
                output_dir
                / f"native-latency-workload-radar-cp{cp_degree}.{extension}",
                dpi=210,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(single)

    with (output_dir / "native-latency-workload-radar-values.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "cp_degree",
                "workload",
                "framework",
                "method",
                "latency_ms",
                "status",
            )
        )
        for cp_degree in CP_DEGREES:
            for workload, _ in NATIVE_WORKLOADS:
                for framework, method, _ in NATIVE_METHODS:
                    method_key = f"{framework}:{method}"
                    latency = all_latencies[cp_degree][workload][method_key]
                    status = (
                        model_statuses[cp_degree][workload][method_key]
                        if isinstance(workload, str)
                        else "complete"
                    )
                    writer.writerow(
                        (
                            cp_degree,
                            workload,
                            framework,
                            method,
                            f"{latency:.6f}" if math.isfinite(latency) else "",
                            status,
                        )
                    )


def save_native_workload_relative_radar(
    output_dir: Path,
    sequence_latencies: dict[int, dict[int, dict[str, float]]],
    model_latencies: dict[int, dict[str, dict[str, float]]],
    model_statuses: dict[int, dict[str, dict[str, str]]],
) -> None:
    count = len(NATIVE_WORKLOADS)
    angles = [2 * math.pi * index / count for index in range(count)]
    closed_angles = angles + [angles[0]]
    all_latencies: dict[int, dict[object, dict[str, float]]] = {}
    relative: dict[int, dict[object, dict[str, float]]] = {}
    for cp_degree in CP_DEGREES:
        all_latencies[cp_degree] = {
            **sequence_latencies[cp_degree],
            **model_latencies[cp_degree],
        }
        relative[cp_degree] = {}
        for workload, methods in all_latencies[cp_degree].items():
            finite = [value for value in methods.values() if math.isfinite(value)]
            best = min(finite)
            relative[cp_degree][workload] = {
                method: best / value if math.isfinite(value) else math.nan
                for method, value in methods.items()
            }

    finite_scores = [
        score
        for by_workload in relative.values()
        for methods in by_workload.values()
        for score in methods.values()
        if math.isfinite(score)
    ]
    lower = max(0.0, math.floor(min(finite_scores) * 10.0) / 10.0)
    ticks = [value / 10.0 for value in range(round(lower * 10), 11)]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(22.0, 8.2),
        subplot_kw={"polar": True},
    )
    for axis, cp_degree in zip(axes, CP_DEGREES):
        axis.set_theta_offset(math.pi / 2)
        axis.set_theta_direction(-1)
        axis.set_xticks(angles, [label for _, label in NATIVE_WORKLOADS])
        axis.tick_params(axis="x", labelsize=8.5, pad=21)
        axis.set_ylim(lower, 1.0)
        axis.set_yticks(ticks, [f"{value:.0%}" for value in ticks])
        axis.set_rlabel_position(22.5)
        axis.tick_params(axis="y", labelsize=8.5, colors="#5f6368")
        axis.grid(color="#b9c0c8", alpha=0.55, linewidth=0.8)
        axis.spines["polar"].set_color("#9aa0a6")
        axis.set_title(f"CP{cp_degree}", fontsize=16, fontweight="bold", pad=17)

        for framework, method, label in NATIVE_METHODS:
            method_key = f"{framework}:{method}"
            values = [
                relative[cp_degree][workload][method_key]
                for workload, _ in NATIVE_WORKLOADS
            ]
            color, marker, linestyle = NATIVE_METHOD_STYLES[method_key]
            axis.plot(
                closed_angles,
                values + [values[0]],
                color=color,
                marker=marker,
                markersize=4.3,
                linewidth=1.8,
                linestyle=linestyle,
                label=label,
            )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    figure.suptitle(
        "Native CP latency relative to the fastest method per workload",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.009,
        "Score = fastest latency / method latency; 100% is fastest; gaps are unsupported; "
        "model shapes are global QKV BxSxHxD; I/T denote image/text tokens; "
        "* Chitu/xDiT shard Z-Image text while SGLang replicates it",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    figure.subplots_adjust(top=0.78, bottom=0.2, left=0.02, right=0.98, wspace=0.25)
    for extension in ("png", "svg"):
        figure.savefig(
            output_dir / f"native-relative-workload-radar-all.{extension}",
            dpi=210,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)

    for cp_degree in CP_DEGREES:
        single, axis = plt.subplots(
            figsize=(9.5, 9.0),
            subplot_kw={"polar": True},
        )
        axis.set_theta_offset(math.pi / 2)
        axis.set_theta_direction(-1)
        axis.set_xticks(angles, [label for _, label in NATIVE_WORKLOADS])
        axis.tick_params(axis="x", labelsize=9.5, pad=29)
        axis.set_ylim(lower, 1.0)
        axis.set_yticks(ticks, [f"{value:.0%}" for value in ticks])
        axis.set_rlabel_position(22.5)
        axis.grid(color="#b9c0c8", alpha=0.55, linewidth=0.8)
        axis.spines["polar"].set_color("#9aa0a6")
        for framework, method, label in NATIVE_METHODS:
            method_key = f"{framework}:{method}"
            values = [
                relative[cp_degree][workload][method_key]
                for workload, _ in NATIVE_WORKLOADS
            ]
            color, marker, linestyle = NATIVE_METHOD_STYLES[method_key]
            axis.plot(
                closed_angles,
                values + [values[0]],
                color=color,
                marker=marker,
                markersize=4.8,
                linewidth=1.9,
                linestyle=linestyle,
                label=label,
            )
        axis.set_title(
            f"CP{cp_degree} latency relative to fastest",
            fontsize=17,
            fontweight="bold",
            pad=23,
        )
        single.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.09),
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        single.subplots_adjust(top=0.88, bottom=0.16, left=0.08, right=0.92)
        for extension in ("png", "svg"):
            single.savefig(
                output_dir
                / f"native-relative-workload-radar-cp{cp_degree}.{extension}",
                dpi=210,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(single)

    with (output_dir / "native-relative-workload-radar-values.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "cp_degree",
                "workload",
                "framework",
                "method",
                "latency_ms",
                "relative_to_fastest",
                "status",
            )
        )
        for cp_degree in CP_DEGREES:
            for workload, _ in NATIVE_WORKLOADS:
                for framework, method, _ in NATIVE_METHODS:
                    method_key = f"{framework}:{method}"
                    latency = all_latencies[cp_degree][workload][method_key]
                    score = relative[cp_degree][workload][method_key]
                    status = (
                        model_statuses[cp_degree][workload][method_key]
                        if isinstance(workload, str)
                        else "complete"
                    )
                    writer.writerow(
                        (
                            cp_degree,
                            workload,
                            framework,
                            method,
                            f"{latency:.6f}" if math.isfinite(latency) else "",
                            f"{score:.6f}" if math.isfinite(score) else "",
                            status,
                        )
                    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    native_sweep_dir = args.native_sweep_dir or args.results_root / "e2e-native-sweep"
    xdit_model_dir = args.xdit_model_dir or args.results_root / "xdit-model-native-sweep"
    sglang_model_dir = (
        args.sglang_model_dir or args.results_root / "sglang-joint-native-sweep"
    )
    sequence_latencies = load_native_sequence_latencies(native_sweep_dir)
    model_scores = load_native_model_latencies(
        args.results_root,
        xdit_model_dir,
        sglang_model_dir,
        args.xdit_wan_dir,
        args.sglang_wan_dir,
        args.sglang_qwen_dir,
        args.chitu_joint_dir,
        args.chitu_zimage_dir,
        args.chitu_qwen_dir,
        args.chitu_wan_dir,
    )
    if model_scores is not None:
        save_native_workload_latency_radar(
            args.output_dir,
            sequence_latencies,
            *model_scores,
        )
        save_native_workload_relative_radar(
            args.output_dir,
            sequence_latencies,
            *model_scores,
        )
    for stem in ("native-latency", "native-relative"):
        for extension in ("png", "svg", "csv"):
            path = args.output_dir / f"{stem}-workload-radar-all.{extension}"
            if extension == "csv":
                path = args.output_dir / f"{stem}-workload-radar-values.csv"
            if path.exists():
                print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
