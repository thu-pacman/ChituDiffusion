#!/usr/bin/env bash
# ChituBench Qwen-Image "serial floor" probe.
#
# Runs four single-image cases, each as its own `chitu run` so that the
# run-level `overall_elapsed_s` is a clean per-case end-to-end wall clock:
#   1. baseline_1gpu          : 1 GPU,  no cache              (reference)
#   2. mc17_cfp2_2gpu         : 2 GPU,  CFG parallel + MeanCache17
#   3. cfp2up4_8gpu           : 8 GPU,  CFG + image CP4, no cache
#   4. mc17_cfp2up4_8gpu      : 8 GPU,  CFG + image CP4 + MeanCache17  (missing combo)
#
# Then collect_floor.py reports e2e vs dit_forward and the floor = e2e - dit_forward.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/ChituBench"
RUN_ID="${CHITUBENCH_RUN_ID:-qwen_image_floor_$(date +%Y%m%d_%H%M)}"
EXPERIMENT_ID="qwen_image_floor"
RESULT_ROOT="$BENCH_DIR/results/$EXPERIMENT_ID/$RUN_ID"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
PYTHON_BIN="${CHITU_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
BASE_CONFIG="$BENCH_DIR/configs/qwen_image/attention/torch_sdpa.yaml"

export CHITUBENCH_PROMPT_FILE="${CHITUBENCH_PROMPT_FILE:-$BENCH_DIR/prompts/qwen_image_attention.json}"
export CHITUBENCH_STEPS="${CHITUBENCH_STEPS:-50}"
export CHITUBENCH_NUM_SEEDS="${CHITUBENCH_NUM_SEEDS:-1}"
export CHITUBENCH_BASE_SEED="${CHITUBENCH_BASE_SEED:-42}"
export CHITUBENCH_WARMUP_RUNS="${CHITUBENCH_WARMUP_RUNS:-0}"
export CHITUBENCH_IMAGE_SIZE="${CHITUBENCH_IMAGE_SIZE:-1328,1328}"
MC17_PARAMS='{"strategy":"meancache","fresh_steps":17,"warmup":0,"cooldown":0,"use_jvp":true}'

mkdir -p "$RESULT_ROOT/configs"

render_config() {
  local case_id="$1" gpus="$2" cfp="$3" up="$4"
  local output_config="$RESULT_ROOT/configs/${case_id}.yaml"
  "$PYTHON_BIN" - "$BASE_CONFIG" "$output_config" "$RESULT_ROOT" "$case_id" "$gpus" "$cfp" "$up" <<'PY'
import sys
from pathlib import Path
import yaml

source, output, result_root, case_id, gpus, cfp, up = sys.argv[1:8]
cfg = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
cfg.setdefault("launch", {})["tag"] = f"chitubench-qwen-image-floor-{case_id}"
cfg["launch"]["gpus_per_node"] = int(gpus)
cfg.setdefault("launch", {}).setdefault("srun", {})["job_name"] = f"chitubench-qwen-floor-{case_id}"
cfg.setdefault("parallel", {})["cfp"] = int(cfp)
cfg["parallel"]["up"] = int(up)
cfg.setdefault("infer", {})["attn_type"] = "torch_sdpa"
cfg.setdefault("output", {})["root_dir"] = result_root
cfg["output"]["log_ranks"] = [0]
cfg["output"]["timer"] = True
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  echo "$output_config"
}

run_case() {
  local case_id="$1" gpus="$2" cfp="$3" up="$4" flexcache="$5"
  local rendered
  rendered="$(render_config "$case_id" "$gpus" "$cfp" "$up")"
  export CHITUBENCH_CASE_ID="$case_id"
  export CHITU_RUN_TASK_ID="$case_id"
  if [[ -n "$flexcache" ]]; then
    export CHITUBENCH_FLEXCACHE_PARAMS="$flexcache"
  else
    unset CHITUBENCH_FLEXCACHE_PARAMS || true
  fi
  echo "=== ChituBench [$EXPERIMENT_ID] $case_id gpus=$gpus cfp=$cfp up=$up flexcache=${flexcache:-none} ==="
  "$CHITU_BIN" run "$rendered" --gpus-per-node "$gpus" --cfp "$cfp"
}

declare -A CASE_SELECTED=()
if [[ -n "${CHITUBENCH_CASES:-}" ]]; then
  IFS=',' read -ra REQUESTED_CASES <<< "$CHITUBENCH_CASES"
  for c in "${REQUESTED_CASES[@]}"; do
    c="${c// /}"; [[ -n "$c" ]] && CASE_SELECTED["$c"]=1
  done
fi
maybe_run_case() {
  local case_id="$1"; shift
  if [[ ${#CASE_SELECTED[@]} -gt 0 && -z "${CASE_SELECTED[$case_id]:-}" ]]; then
    return
  fi
  run_case "$case_id" "$@"
}

#               case_id                  gpus cfp up  flexcache
maybe_run_case  baseline_1gpu            1    1   1   ""
maybe_run_case  mc17_cfp2_2gpu           2    2   1   "$MC17_PARAMS"
maybe_run_case  cfp2up4_8gpu             8    2   4   ""
maybe_run_case  mc17_cfp2up4_8gpu        8    2   4   "$MC17_PARAMS"

"$PYTHON_BIN" "$BENCH_DIR/scripts/collect_floor.py" "$RESULT_ROOT" --baseline-case baseline_1gpu || true
echo "Results: $RESULT_ROOT"
