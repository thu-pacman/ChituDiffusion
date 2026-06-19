#!/usr/bin/env python3
"""Collect end-to-end vs DiT-forward timing to expose the per-image serial floor.

Unlike collect.py (which only reports the rank-0 `dit_forward` timer), this script
also reads the run-level `overall_elapsed_s` wall-clock written by
DiffusionTestRunContext.dump_timing_summary. For single-image runs (NUM_SEEDS=1,
WARMUP_RUNS=0) `overall_elapsed_s` is the per-case end-to-end generation time, so

    serial_floor_s = overall_elapsed_s - dit_forward_s

isolates the non-DiT cost (scheduler steps, per-step barrier/synchronize, CFG
all_gather, VAE decode, dispatch) that neither caching nor parallelism removes.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Optional


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_warmup(task_id: str) -> bool:
    return task_id.startswith("warmup_")


def dit_forward_s(run_dir: Path, task_id: str) -> Optional[float]:
    payload = read_json(run_dir / "metrics" / "timing" / f"{task_id}.json") or {}
    timers = payload.get("timers") or {}
    dit = timers.get("dit_forward_step") or timers.get("dit_forward") or {}
    total_ms = dit.get("total_ms")
    return None if total_ms is None else total_ms / 1000.0


def case_from_run(run_dir: Path) -> str:
    # run dir is "<tag>-<timestamp>-<task_id>"; the task_id is the case label.
    name = run_dir.name
    parts = name.rsplit("-", 1)
    return parts[-1] if len(parts) == 2 else name


def _timer_total_s(timers: dict[str, Any], *names: str) -> Optional[float]:
    for name in names:
        entry = timers.get(name)
        if entry and entry.get("total_ms") is not None:
            return entry["total_ms"] / 1000.0
    return None


def collect_run(run_dir: Path) -> Optional[dict[str, Any]]:
    timing_summary = read_json(run_dir / "metrics" / "timing" / "summary.json")
    request_payload = read_json(run_dir / "request_params.json")
    if timing_summary is None or request_payload is None:
        return None
    overall_s = timing_summary.get("overall_elapsed_s")
    # Prefer the serving latency (wall clock minus benchmark instrumentation) as the
    # end-to-end number; fall back to overall when older runs lack the field.
    serving_s = timing_summary.get("serving_elapsed_s")
    if serving_s is None:
        serving_s = overall_s
    bench_overhead_s = timing_summary.get("benchmark_overhead_s")
    timers = timing_summary.get("timers") or {}
    task_ids = [
        str(r.get("request_id") or "")
        for r in request_payload.get("requests", [])
        if not is_warmup(str(r.get("request_id") or ""))
    ]
    if not task_ids:
        return None
    dit_values = [v for v in (dit_forward_s(run_dir, tid) for tid in task_ids) if v is not None]
    if not dit_values:
        return None
    dit_s = sum(dit_values) / len(dit_values)
    floor_s = None if serving_s is None else serving_s - dit_s

    # Break the serving wall clock into named stages so the "floor" is attributable.
    text_encode_s = _timer_total_s(timers, "TextEncode")
    vae_decode_s = _timer_total_s(timers, "VaeDecode", "VAEDecode")
    denoise_s = _timer_total_s(timers, "denoise")
    # denoise loop overhead = scheduler step + per-step barrier/synchronize + CFG all_gather
    denoise_overhead_s = (
        None if denoise_s is None else denoise_s - dit_s
    )
    # everything not covered by encode/denoise/decode (dispatch, save, broadcasts, etc.)
    other_s = None
    if serving_s is not None and None not in (text_encode_s, denoise_s, vae_decode_s):
        other_s = serving_s - text_encode_s - denoise_s - vae_decode_s
    return {
        "case": case_from_run(run_dir),
        "run_name": run_dir.name,
        "overall_s": overall_s,
        "serving_s": serving_s,
        "bench_overhead_s": bench_overhead_s,
        "dit_forward_s": dit_s,
        "serial_floor_s": floor_s,
        "text_encode_s": text_encode_s,
        "denoise_s": denoise_s,
        "denoise_overhead_s": denoise_overhead_s,
        "vae_decode_s": vae_decode_s,
        "other_s": other_s,
        "num_tasks": len(task_ids),
    }


def pick_baseline(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for row in rows:
        if "baseline" in row["case"] or row["case"].endswith("1gpu"):
            return row
    return rows[0] if rows else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--baseline-case", default=None, help="case id to use as 1.0x baseline")
    args = parser.parse_args()

    root = Path(args.experiment_dir).resolve()
    run_dirs = [
        p for p in root.iterdir()
        if p.is_dir() and (p / "request_params.json").exists()
    ] if root.exists() else []
    rows = [r for r in (collect_run(p) for p in sorted(run_dirs)) if r is not None]
    if not rows:
        raise SystemExit(f"No floor-collectable runs under {root}")

    if args.baseline_case:
        baseline = next((r for r in rows if r["case"] == args.baseline_case), None)
    else:
        baseline = pick_baseline(rows)
    base_e2e = baseline["serving_s"] if baseline else None
    base_dit = baseline["dit_forward_s"] if baseline else None

    for row in rows:
        row["e2e_speedup"] = (
            None if not base_e2e or not row["serving_s"] else base_e2e / row["serving_s"]
        )
        row["dit_speedup"] = (
            None if not base_dit or not row["dit_forward_s"] else base_dit / row["dit_forward_s"]
        )

    out_json = root / "floor_summary.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def fmt(v: Any, spec: str = ".3f") -> str:
        return "-" if v is None else format(v, spec)

    header = (
        f"{'case':<26}{'serv_s':>9}{'bench_s':>9}{'dit_s':>9}{'floor_s':>9}"
        f"{'enc_s':>8}{'dnovh_s':>9}{'dec_s':>8}{'oth_s':>8}{'e2e_x':>7}{'dit_x':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in sorted(rows, key=lambda r: r["serving_s"] or 0.0, reverse=True):
        lines.append(
            f"{row['case']:<26}"
            f"{fmt(row['serving_s'], '.2f'):>9}"
            f"{fmt(row['bench_overhead_s'], '.2f'):>9}"
            f"{fmt(row['dit_forward_s'], '.2f'):>9}"
            f"{fmt(row['serial_floor_s'], '.2f'):>9}"
            f"{fmt(row['text_encode_s'], '.2f'):>8}"
            f"{fmt(row['denoise_overhead_s'], '.2f'):>9}"
            f"{fmt(row['vae_decode_s'], '.2f'):>8}"
            f"{fmt(row['other_s'], '.2f'):>8}"
            f"{fmt(row['e2e_speedup'], '.2f'):>7}"
            f"{fmt(row['dit_speedup'], '.2f'):>7}"
        )
    table = "\n".join(lines)
    (root / "floor_summary.txt").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
