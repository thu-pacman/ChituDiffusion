#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/ChituBench"
RUN_ID="${CHITUBENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ID="wan2_1_t2v_1_3b_flexcache"
RESULT_ROOT="$BENCH_DIR/results/$EXPERIMENT_ID/$RUN_ID"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
PYTHON_BIN="${CHITU_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG="$BENCH_DIR/configs/wan2_1_t2v_1_3b/flexcache/base_flash.yaml"
PLOT_TITLE="Wan2.1 T2V 1.3B FlexCache Trade-off"
CONTACT_TITLE="Wan2.1 T2V 1.3B FlexCache Visual Check"
REFERENCE_DIR="${CHITUBENCH_REFERENCE_DIR:-}"

PROMPT_FILE="${CHITUBENCH_PROMPT_FILE:-$BENCH_DIR/prompts/wan2_1_t2v_1_3b_attention.json}"
export CHITUBENCH_PROMPT_FILE="$PROMPT_FILE"
export CHITUBENCH_STEPS="${CHITUBENCH_STEPS:-50}"
export CHITUBENCH_NUM_SEEDS="${CHITUBENCH_NUM_SEEDS:-1}"
export CHITUBENCH_BASE_SEED="${CHITUBENCH_BASE_SEED:-42}"
export CHITUBENCH_WARMUP_RUNS="${CHITUBENCH_WARMUP_RUNS:-0}"
export CHITUBENCH_VIDEO_SIZE="${CHITUBENCH_VIDEO_SIZE:-832,480}"
export CHITUBENCH_FRAME_NUM="${CHITUBENCH_FRAME_NUM:-81}"
export CHITUBENCH_CASES="${CHITUBENCH_CASES:-origin_flash pab_s2c3 blockdance_b18g2 taylorseer_f3o1 cubic_s20_tau8_4x4_w8c2}"

if (( CHITUBENCH_STEPS <= 10 )); then
  FLEXCACHE_WARMUP="${CHITUBENCH_FLEXCACHE_WARMUP:-1}"
  FLEXCACHE_COOLDOWN="${CHITUBENCH_FLEXCACHE_COOLDOWN:-1}"
else
  FLEXCACHE_WARMUP="${CHITUBENCH_FLEXCACHE_WARMUP:-3}"
  FLEXCACHE_COOLDOWN="${CHITUBENCH_FLEXCACHE_COOLDOWN:-3}"
fi
CUBIC_WARMUP="${CHITUBENCH_CUBIC_WARMUP:-8}"
CUBIC_COOLDOWN="${CHITUBENCH_CUBIC_COOLDOWN:-2}"

mkdir -p "$RESULT_ROOT/configs"

render_config() {
  local output_config="$1"
  local tag="$2"
  "$PYTHON_BIN" - "$CONFIG" "$output_config" "$RESULT_ROOT" "$tag" <<'PY'
import sys
from pathlib import Path
import yaml

source, output, result_root, tag = sys.argv[1:5]
cfg = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
cfg.setdefault("launch", {})["tag"] = tag
cfg.setdefault("launch", {}).setdefault("srun", {})["job_name"] = tag
cfg.setdefault("output", {})["root_dir"] = result_root
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
}

should_run_case() {
  local case_id="$1"
  for item in $CHITUBENCH_CASES; do
    [[ "$item" == "$case_id" ]] && return 0
  done
  return 1
}

link_reference() {
  [[ -z "$REFERENCE_DIR" ]] && return 1
  "$PYTHON_BIN" - "$REFERENCE_DIR" "$RESULT_ROOT" <<'PY'
import sys
from pathlib import Path

reference = Path(sys.argv[1]).resolve()
result_root = Path(sys.argv[2]).resolve()
if not (reference / "request_params.json").exists():
    raise SystemExit(f"reference dir does not look like a ChituBench run: {reference}")
link = result_root / reference.name
if link.exists() or link.is_symlink():
    if link.resolve() != reference:
        raise SystemExit(f"reference link path already exists with a different target: {link}")
else:
    link.symlink_to(reference, target_is_directory=True)
print(link)
PY
}

run_case() {
  local case_id="$1"
  local flexcache_json="$2"
  if ! should_run_case "$case_id"; then
    echo "=== ChituBench [$EXPERIMENT_ID] skip $case_id ==="
    return 0
  fi
  if [[ "$case_id" == "origin_flash" && -n "$REFERENCE_DIR" ]]; then
    link_reference
    echo "=== ChituBench [$EXPERIMENT_ID] reuse reference $REFERENCE_DIR ==="
    return 0
  fi
  local rendered="$RESULT_ROOT/configs/${case_id}.yaml"
  render_config "$rendered" "chitubench-wan21-13b-flexcache-${case_id}"
  export CHITUBENCH_CASE_ID="$case_id"
  export CHITU_RUN_TASK_ID="$case_id"
  if [[ -n "$flexcache_json" ]]; then
    export CHITUBENCH_FLEXCACHE_PARAMS="$flexcache_json"
  else
    unset CHITUBENCH_FLEXCACHE_PARAMS || true
  fi
  echo "=== ChituBench [$EXPERIMENT_ID] $case_id ==="
  "$CHITU_BIN" run "$rendered" --gpus-per-node 1
  "$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
    --experiment-id "$EXPERIMENT_ID" \
    --title "$PLOT_TITLE" \
    --allow-partial
}

fc_json() {
  "$PYTHON_BIN" - "$FLEXCACHE_WARMUP" "$FLEXCACHE_COOLDOWN" "$@" <<'PY'
import json
import sys

warmup = int(sys.argv[1])
cooldown = int(sys.argv[2])
payload = json.loads(sys.argv[3])
payload["warmup"] = warmup
payload["cooldown"] = cooldown
print(json.dumps(payload, separators=(",", ":")))
PY
}

cubic_json() {
  "$PYTHON_BIN" - "$CUBIC_WARMUP" "$CUBIC_COOLDOWN" "$@" <<'PY'
import json
import sys

warmup = int(sys.argv[1])
cooldown = int(sys.argv[2])
payload = json.loads(sys.argv[3])
payload["warmup"] = warmup
payload["cooldown"] = cooldown
print(json.dumps(payload, separators=(",", ":")))
PY
}

run_case origin_flash ""
run_case pab_s2c3 "$(fc_json '{"strategy":"pab","skip_self_range":2,"skip_cross_range":3}')"
run_case blockdance_b18g2 "$(fc_json '{"strategy":"blockdance","boundary_block":18,"group_size":2,"start_fraction":0.25,"end_fraction":0.90}')"
run_case taylorseer_f3o1 "$(fc_json '{"strategy":"taylorseer","fresh_threshold":3,"max_order":1,"first_enhance":1}')"
run_case cubic_s20_tau8_4x4_w8c2 "$(cubic_json '{"strategy":"cubic","target_speedup":2.0,"tau_max":8,"block_size":16,"uniform_square_min_splits":4}')"

unset CHITUBENCH_FLEXCACHE_PARAMS || true

REFERENCE_DIR="$("$PYTHON_BIN" - "$RESULT_ROOT" "${REFERENCE_DIR:-}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
provided = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 and sys.argv[2] else None
if provided is not None:
    print(provided)
    raise SystemExit(0)
for run_dir in sorted(root.iterdir()):
    request_path = run_dir / "request_params.json"
    if not request_path.exists():
        continue
    payload = json.loads(request_path.read_text())
    if any((req.get("params") or {}).get("role") == "origin_flash" for req in payload.get("requests") or []):
        print(run_dir)
        break
else:
    raise SystemExit("failed to locate origin_flash reference run")
PY
)"

HPSV3_CONFIG="${CHITUBENCH_HPSV3_CONFIG:-$BENCH_DIR/results/hpsv3_assets/HPSv3_7B.local.yaml}"
HPSV3_CHECKPOINT="${CHITUBENCH_HPSV3_CHECKPOINT:-$BENCH_DIR/results/hpsv3_assets/HPSv3.chitu_compat.safetensors}"

run_quality_on_gpu() {
  local -a args=("$BENCH_DIR/scripts/evaluate_quality.py" "$RESULT_ROOT" "--origin-dir" "$REFERENCE_DIR")
  if [[ "${CHITUBENCH_SKIP_HPSV3:-0}" == "1" ]]; then
    args+=("--skip-hpsv3")
  else
    args+=(
      "--hpsv3-config-path" "$HPSV3_CONFIG"
      "--hpsv3-checkpoint-path" "$HPSV3_CHECKPOINT"
      "--hpsv3-frame-index" "-1"
    )
  fi
  CHITU_PROJECT_ROOT="$ROOT_DIR" \
  CHITU_PYTHON_BIN="$PYTHON_BIN" \
  SRUN_JOB_NAME="chitubench-wan21-flexcache-quality" \
  "$ROOT_DIR/script/srun_direct.sh" 1 1 "${args[@]}"
}

check_required_quality() {
  "$PYTHON_BIN" - "$RESULT_ROOT/quality/quality_summary.json" <<'PY'
import json
import math
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())
missing = []
for row in rows:
    case = row.get("case")
    for metric in ("psnr_mean", "one_minus_lpips_mean", "hpsv3_score_mean"):
        value = row.get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            missing.append(f"{case}:{metric}")
if missing:
    raise SystemExit("Missing required quality metrics: " + ", ".join(missing))
PY
}

if [[ "${CHITUBENCH_SKIP_HPSV3:-0}" == "1" ]]; then
  run_quality_on_gpu
  echo "HPSv3 skipped by CHITUBENCH_SKIP_HPSV3=1; this is a partial run." >&2
elif [[ -f "$HPSV3_CONFIG" && -f "$HPSV3_CHECKPOINT" ]]; then
  run_quality_on_gpu
  check_required_quality
else
  echo "Missing HPSv3 assets: config=$HPSV3_CONFIG checkpoint=$HPSV3_CHECKPOINT" >&2
  exit 1
fi

"$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --title "$PLOT_TITLE"
"$PYTHON_BIN" "$BENCH_DIR/scripts/make_contact_sheet.py" "$RESULT_ROOT" \
  --seed "$CHITUBENCH_BASE_SEED" \
  --title "$CONTACT_TITLE" \
  --experiment-id "$EXPERIMENT_ID" \
  --cases origin_flash pab_s2c3 blockdance_b18g2 taylorseer_f3o1 cubic_s20_tau8_4x4_w8c2 \
  --frame-index -1

mkdir -p "$BENCH_DIR/plots/$EXPERIMENT_ID"
cp "$RESULT_ROOT/plots/speed_quality.png" "$BENCH_DIR/plots/$EXPERIMENT_ID/speed_quality_${RUN_ID}.png"
cp "$RESULT_ROOT/visuals/contact_sheet.png" "$BENCH_DIR/plots/$EXPERIMENT_ID/contact_sheet_${RUN_ID}.png"

echo "Results: $RESULT_ROOT"
