#!/usr/bin/env python3
"""Per-image end-to-end latency from ChituBench per-task timing JSONs.

Reconstructs a clean end-to-end (TextEncode + Denoise + VAEDecode) per generated
image from the ``stage_elapsed`` records each task writes, excluding warmup tasks
(``warmup_*``). Emits a stacked-stage bar chart plus a markdown table with the
end-to-end speedup vs the 1-GPU baseline and the parallel-sampler delta per pair.

Usage:
  plot_e2e_ablation.py <experiment_dir> [--order c1,c2,...] [--baseline baseline_1gpu]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["TextEncode", "Denoise", "VAEDecode"]
STAGE_COLORS = {"TextEncode": "#9aa7b8", "Denoise": "#1f9e89", "VAEDecode": "#e08214"}


def per_image_stages(run_dir: Path) -> dict[str, float] | None:
    """Mean per-image stage_elapsed (ms) for one run, warmup excluded."""
    timing = run_dir / "metrics" / "timing"
    if not timing.exists():
        return None
    sums: dict[str, list[float]] = defaultdict(list)
    n = 0
    for jf in sorted(timing.glob("*.json")):
        if jf.name == "summary.json" or jf.name.startswith("warmup_"):
            continue
        payload = json.loads(jf.read_text(encoding="utf-8"))
        recs = (payload.get("records") or {}).get("stage_elapsed") or []
        if not recs:
            continue
        by_stage = defaultdict(float)
        for r in recs:
            by_stage[r.get("stage")] += float(r.get("elapsed_ms", 0.0))
        for s in STAGES:
            sums[s].append(by_stage.get(s, 0.0))
        n += 1
    if n == 0:
        return None
    return {s: (sum(v) / len(v) if v else 0.0) for s, v in sums.items()}


def collect(experiment_dir: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for run_dir in sorted(experiment_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        stages = per_image_stages(run_dir)
        if stages is None:
            continue
        case = run_dir.name.split("-")[-1]
        out[case] = stages
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("--order", default="")
    ap.add_argument("--baseline", default="baseline_1gpu")
    ap.add_argument("--title", default="End-to-end latency (VAE + DiT + Sampler)")
    args = ap.parse_args()

    data = collect(args.experiment_dir)
    if not data:
        raise SystemExit(f"No stage_elapsed data under {args.experiment_dir}")

    cases = [c.strip() for c in args.order.split(",") if c.strip()] or list(data)
    cases = [c for c in cases if c in data]
    base_ms = sum(data[args.baseline][s] for s in STAGES) if args.baseline in data else None

    # --- stacked bar chart ---
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(cases)), 5))
    bottoms = [0.0] * len(cases)
    for s in STAGES:
        vals = [data[c][s] / 1000.0 for c in cases]
        ax.bar(cases, vals, bottom=[b / 1000.0 for b in bottoms], label=s, color=STAGE_COLORS[s])
        bottoms = [b + data[c][s] for b, c in zip(bottoms, cases)]
    for i, c in enumerate(cases):
        e2e = sum(data[c][s] for s in STAGES) / 1000.0
        label = f"{e2e:.3f}s"
        if base_ms:
            label += f"\n{base_ms / 1000.0 / e2e:.2f}x"
        ax.text(i, e2e, label, ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Per-image latency (s)")
    ax.set_title(args.title)
    ax.legend(title="stage")
    ax.margins(y=0.15)
    fig.tight_layout()
    out_png = args.experiment_dir / "plots" / "e2e_ablation.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")

    # --- markdown table ---
    lines = [
        "| case | TextEncode (s) | Denoise (s) | VAEDecode (s) | End-to-end (s) | E2E speedup vs 1 GPU |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in cases:
        te, dn, vd = (data[c][s] / 1000.0 for s in STAGES)
        e2e = te + dn + vd
        spd = f"{base_ms / 1000.0 / e2e:.2f}x" if base_ms else "-"
        lines.append(f"| {c} | {te:.3f} | {dn:.3f} | {vd:.3f} | {e2e:.3f} | {spd} |")
    md = "\n".join(lines) + "\n"
    out_md = args.experiment_dir / "e2e_summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {out_md}")
    print(md)

    # --- sampler delta per layout pair ---
    for cp in ("cp2", "cp4", "cp8"):
        only, samp = f"{cp}_dit_only", f"{cp}_dit_sampler"
        if only in data and samp in data:
            eo = sum(data[only][s] for s in STAGES)
            es = sum(data[samp][s] for s in STAGES)
            print(f"{cp}: only={eo/1000:.3f}s  +sampler={es/1000:.3f}s  E2E delta={100*(eo-es)/eo:+.2f}%")


if __name__ == "__main__":
    main()
