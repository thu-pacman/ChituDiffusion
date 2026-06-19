#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/ChituBench"
RUN_ID="${CHITUBENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ID="flux1_dev_flexcache"
RESULT_ROOT="$BENCH_DIR/results/$EXPERIMENT_ID/$RUN_ID"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
PYTHON_BIN="${CHITU_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG="$BENCH_DIR/configs/flux1_dev/flexcache/base_flash.yaml"
PLOT_TITLE="Flux1-dev FlexCache Trade-off"
CONTACT_TITLE="Flux1-dev FlexCache Visual Check"

PROMPT_FILE="${CHITUBENCH_PROMPT_FILE:-$BENCH_DIR/prompts/flux1_attention.json}"
export CHITUBENCH_PROMPT_FILE="$PROMPT_FILE"
export CHITUBENCH_STEPS="${CHITUBENCH_STEPS:-50}"
export CHITUBENCH_NUM_SEEDS="${CHITUBENCH_NUM_SEEDS:-3}"
export CHITUBENCH_BASE_SEED="${CHITUBENCH_BASE_SEED:-42}"
export CHITUBENCH_WARMUP_RUNS="${CHITUBENCH_WARMUP_RUNS:-1}"
if (( CHITUBENCH_STEPS <= 10 )); then
  FLEXCACHE_WARMUP="${CHITUBENCH_FLEXCACHE_WARMUP:-1}"
  FLEXCACHE_COOLDOWN="${CHITUBENCH_FLEXCACHE_COOLDOWN:-1}"
else
  FLEXCACHE_WARMUP="${CHITUBENCH_FLEXCACHE_WARMUP:-3}"
  FLEXCACHE_COOLDOWN="${CHITUBENCH_FLEXCACHE_COOLDOWN:-3}"
fi
CUBIC_WARMUP="${CHITUBENCH_CUBIC_WARMUP:-8}"
CUBIC_COOLDOWN="${CHITUBENCH_CUBIC_COOLDOWN:-2}"
TEACACHE_WARMUP="${CHITUBENCH_TEACACHE_WARMUP:-1}"
TEACACHE_COOLDOWN="${CHITUBENCH_TEACACHE_COOLDOWN:-1}"

mkdir -p "$RESULT_ROOT/configs"

render_config() {
  local source_config="$1"
  local output_config="$2"
  local tag="$3"
  "$PYTHON_BIN" - "$source_config" "$output_config" "$RESULT_ROOT" "$tag" <<'PY'
import sys
from pathlib import Path
import yaml

source, output, result_root, tag = sys.argv[1:5]
cfg = yaml.safe_load(Path(source).read_text())
cfg.setdefault("launch", {})["tag"] = tag
cfg.setdefault("launch", {}).setdefault("srun", {})["job_name"] = tag
cfg.setdefault("output", {})["root_dir"] = result_root
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
}

should_run_case() {
  local case_id="$1"
  local selected="${CHITUBENCH_CASES:-}"
  [[ -z "$selected" ]] && return 0
  for item in $selected; do
    [[ "$item" == "$case_id" ]] && return 0
  done
  return 1
}

run_case() {
  local case_id="$1"
  local flexcache_json="$2"
  local rendered="$RESULT_ROOT/configs/${case_id}.yaml"
  if ! should_run_case "$case_id"; then
    echo "=== ChituBench [$EXPERIMENT_ID] skip $case_id ==="
    return 0
  fi
  render_config "$CONFIG" "$rendered" "chitubench-flux1-flexcache-${case_id}"
  export CHITUBENCH_CASE_ID="$case_id"
  export CHITUBENCH_FLEXCACHE_PARAMS="$flexcache_json"
  export CHITU_RUN_TASK_ID="$case_id"
  echo "=== ChituBench [$EXPERIMENT_ID] $case_id ==="
  "$CHITU_BIN" run "$rendered" --gpus-per-node 1
  "$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
    --experiment-id "$EXPERIMENT_ID" \
    --title "$PLOT_TITLE" \
    --allow-partial
}

run_case origin_flash ""
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

teacache_json() {
  "$PYTHON_BIN" - "$TEACACHE_WARMUP" "$TEACACHE_COOLDOWN" "$@" <<'PY'
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

run_case teacache_t025 "$(teacache_json '{"strategy":"teacache","teacache_thresh":0.25,"use_ref_steps":true}')"
run_case teacache_t040 "$(teacache_json '{"strategy":"teacache","teacache_thresh":0.40,"use_ref_steps":true}')"
run_case teacache_t060 "$(teacache_json '{"strategy":"teacache","teacache_thresh":0.60,"use_ref_steps":true}')"
run_case teacache_t080 "$(teacache_json '{"strategy":"teacache","teacache_thresh":0.80,"use_ref_steps":true}')"
run_case pab_s2c3 "$(fc_json '{"strategy":"pab","skip_self_range":2,"skip_cross_range":3}')"
run_case pab_s3c4 "$(fc_json '{"strategy":"pab","skip_self_range":3,"skip_cross_range":4}')"
run_case blockdance_b18g2 "$(fc_json '{"strategy":"blockdance","boundary_block":18,"group_size":2,"start_fraction":0.25,"end_fraction":0.90}')"
run_case blockdance_b24g3 "$(fc_json '{"strategy":"blockdance","boundary_block":24,"group_size":3,"start_fraction":0.30,"end_fraction":0.92}')"
run_case taylorseer_f2o1 "$(fc_json '{"strategy":"taylorseer","fresh_threshold":2,"max_order":1,"first_enhance":1}')"
run_case taylorseer_f3o1 "$(fc_json '{"strategy":"taylorseer","fresh_threshold":3,"max_order":1,"first_enhance":1}')"
run_case taylorseer_f4o1 "$(fc_json '{"strategy":"taylorseer","fresh_threshold":4,"max_order":1,"first_enhance":1}')"
run_case taylorseer_f5o2 "$(fc_json '{"strategy":"taylorseer","fresh_threshold":5,"max_order":2,"first_enhance":1}')"
run_case cubic_s20_tau8_4x4_w8c2 "$(cubic_json '{"strategy":"cubic","target_speedup":2.0,"tau_max":8,"block_size":16,"uniform_square_min_splits":4}')"
run_case cubic_s225_tau8_4x4_w8c2 "$(cubic_json '{"strategy":"cubic","target_speedup":2.25,"tau_max":8,"block_size":16,"uniform_square_min_splits":4}')"
run_case cubic_s25_tau8_4x4_w8c2 "$(cubic_json '{"strategy":"cubic","target_speedup":2.5,"tau_max":8,"block_size":16,"uniform_square_min_splits":4}')"
run_case cubic_s275_tau8_4x4_w8c2 "$(cubic_json '{"strategy":"cubic","target_speedup":2.75,"tau_max":8,"block_size":16,"uniform_square_min_splits":4}')"
run_case cubic_s30_tau8_4x4_w8c2 "$(cubic_json '{"strategy":"cubic","target_speedup":3.0,"tau_max":8,"block_size":16,"uniform_square_min_splits":4}')"
run_case cubic_s40_tau8_4x4_w8c2 "$(cubic_json '{"strategy":"cubic","target_speedup":4.0,"tau_max":8,"block_size":16,"uniform_square_min_splits":4}')"
run_case cubic_s50_tau8_4x4_w8c2 "$(cubic_json '{"strategy":"cubic","target_speedup":5.0,"tau_max":8,"block_size":16,"uniform_square_min_splits":4}')"

unset CHITUBENCH_FLEXCACHE_PARAMS

HPSV3_CONFIG="${CHITUBENCH_HPSV3_CONFIG:-}"
HPSV3_CHECKPOINT="${CHITUBENCH_HPSV3_CHECKPOINT:-}"
if [[ -n "$HPSV3_CONFIG" && -n "$HPSV3_CHECKPOINT" ]]; then
  "$PYTHON_BIN" "$BENCH_DIR/scripts/evaluate_quality.py" "$RESULT_ROOT" \
    --hpsv3-config-path "$HPSV3_CONFIG" \
    --hpsv3-checkpoint-path "$HPSV3_CHECKPOINT"
else
  "$PYTHON_BIN" "$BENCH_DIR/scripts/evaluate_quality.py" "$RESULT_ROOT" --skip-hpsv3
  echo "HPSv3 skipped. Set CHITUBENCH_HPSV3_CONFIG and CHITUBENCH_HPSV3_CHECKPOINT to enable it." >&2
fi

"$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --title "$PLOT_TITLE"
"$PYTHON_BIN" "$BENCH_DIR/scripts/make_contact_sheet.py" "$RESULT_ROOT" \
  --seed "$CHITUBENCH_BASE_SEED" \
  --title "$CONTACT_TITLE" \
  --experiment-id "$EXPERIMENT_ID"

mkdir -p "$BENCH_DIR/plots/$EXPERIMENT_ID"
cp "$RESULT_ROOT/plots/speed_quality.png" "$BENCH_DIR/plots/$EXPERIMENT_ID/speed_quality_${RUN_ID}.png"
cp "$RESULT_ROOT/visuals/contact_sheet.png" "$BENCH_DIR/plots/$EXPERIMENT_ID/contact_sheet_${RUN_ID}.png"

echo "Results: $RESULT_ROOT"
