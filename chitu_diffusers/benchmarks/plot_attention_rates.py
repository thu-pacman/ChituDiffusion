from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_runs(values: list[str]) -> dict[str, list[tuple[float, dict]]]:
    runs: dict[str, list[tuple[float, dict]]] = {}
    for value in values:
        try:
            label, raw_rate, raw_path = value.split(":", 2)
        except ValueError as exc:
            raise ValueError(
                f"invalid run {value!r}; expected LABEL:RATE:METRICS_JSON"
            ) from exc
        metrics = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        runs.setdefault(label, []).append((float(raw_rate), metrics["summary"]))
    for rows in runs.values():
        rows.sort(key=lambda item: item[0])
    return runs


def render(runs: dict[str, list[tuple[float, dict]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.3), constrained_layout=True)
    styles = {
        "Elastic AGKV": ("#1570ef", "o", "-"),
        "Static DP AGKV": ("#dc6803", "s", "--"),
        "Static CP AGKV": ("#d92d20", "^", "--"),
        "Elastic USP u2r2": ("#039855", "D", "-"),
    }
    for label, rows in runs.items():
        rates = [rate for rate, _ in rows]
        summaries = [summary for _, summary in rows]
        color, marker, linestyle = styles.get(label, (None, "o", "-"))
        axes[0].plot(
            rates,
            [float(row["throughput_req_per_s"]) for row in summaries],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=label,
            color=color,
        )
        axes[1].plot(
            rates,
            [float(row["p95_latency_ms"]) / 1000.0 for row in summaries],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=label,
            color=color,
        )
        axes[2].plot(
            rates,
            [float(row["mean_queue_delay_ms"]) / 1000.0 for row in summaries],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=label,
            color=color,
        )
    axes[0].plot(
        sorted({rate for rows in runs.values() for rate, _ in rows}),
        sorted({rate for rows in runs.values() for rate, _ in rows}),
        linestyle="--",
        linewidth=1,
        color="#98a2b3",
        label="offered rate",
    )
    axes[0].set_title("Achieved throughput")
    axes[0].set_ylabel("requests/s")
    axes[1].set_title("P95 end-to-end latency")
    axes[1].set_ylabel("seconds")
    axes[2].set_title("Mean queue delay")
    axes[2].set_ylabel("seconds")
    for axis in axes:
        axis.set_xlabel("offered arrival rate (requests/s)")
        axis.grid(color="#e4e7ec", linewidth=0.7)
        axis.legend(frameon=False)
    fig.suptitle("EPAC Z-Image serve benchmark", fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot EPAC serve variants across offered arrival rates"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL:RATE:METRICS_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(load_runs(args.run), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
