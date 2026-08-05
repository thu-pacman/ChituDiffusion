#!/usr/bin/env bash
# FreeCache vs MeanCache benchmark on Qwen-Image (50 step, cfp2).
# GPU via: chitu run -> script/srun_direct.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/ChituBench"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
PYTHON_BIN="${CHITU_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
RUN_ID="${CHITUBENCH_RUN_ID:-qwen_freecache_$(date +%Y%m%d_%H%M)}"
EXPERIMENT_ID="qwen_image_freecache"
RESULT_ROOT="$BENCH_DIR/results/$EXPERIMENT_ID/$RUN_ID"
BASE_CONFIG="$BENCH_DIR/configs/qwen_image/attention/torch_sdpa.yaml"
REFERENCE_DIR="${CHITUBENCH_REFERENCE_DIR:-}"
PLOT_TITLE="Qwen-Image FreeCache vs MeanCache"
CONTACT_TITLE="Qwen-Image FreeCache Visual Check"

export CHITUBENCH_PROMPT_FILE="${CHITUBENCH_PROMPT_FILE:-$BENCH_DIR/prompts/qwen_image_attention.json}"
export CHITUBENCH_STEPS="${CHITUBENCH_STEPS:-50}"
export CHITUBENCH_NUM_SEEDS="${CHITUBENCH_NUM_SEEDS:-3}"
export CHITUBENCH_BASE_SEED="${CHITUBENCH_BASE_SEED:-42}"
export CHITUBENCH_WARMUP_RUNS="${CHITUBENCH_WARMUP_RUNS:-0}"
export CHITUBENCH_IMAGE_SIZE="${CHITUBENCH_IMAGE_SIZE:-1328,1328}"
export CHITUBENCH_GPUS_PER_NODE="${CHITUBENCH_GPUS_PER_NODE:-2}"
export CHITUBENCH_CFP="${CHITUBENCH_CFP:-2}"
export CHITUBENCH_CASES="${CHITUBENCH_CASES:-torch_sdpa qwen_meancache25_50_cfp2 qwen_meancache17_50_cfp2 qwen_freecache_tol010 qwen_freecache_tol015 qwen_freecache_tol020 qwen_freecache_tol030}"

mkdir -p "$RESULT_ROOT/configs"

render_config() {
  local case_id="$1"
  local gpus="$2"
  local cfp="$3"
  local output_config="$RESULT_ROOT/configs/${case_id}.yaml"
  "$PYTHON_BIN" - "$BASE_CONFIG" "$output_config" "$RESULT_ROOT" "$case_id" "$gpus" "$cfp" <<'PY'
import sys
from pathlib import Path
import yaml

source, output, result_root, case_id, gpus, cfp = sys.argv[1:7]
cfg = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
cfg.setdefault("launch", {})["tag"] = f"chitubench-qwen-freecache-{case_id}"
cfg["launch"]["gpus_per_node"] = int(gpus)
cfg.setdefault("launch", {}).setdefault("srun", {})["job_name"] = f"qwen-freecache-{case_id}"
cfg.setdefault("parallel", {})["cfp"] = int(cfp)
cfg["parallel"]["up"] = 1
cfg.setdefault("output", {})["root_dir"] = result_root
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  echo "$output_config"
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
    raise SystemExit(f"reference dir invalid: {reference}")
link = result_root / reference.name
if not link.exists() and not link.is_symlink():
    link.symlink_to(reference, target_is_directory=True)
print(link)
PY
}

run_case() {
  local case_id="$1"
  local flexcache_json="$2"
  if ! should_run_case "$case_id"; then
    echo "=== skip $case_id ==="
    return 0
  fi
  if [[ "$case_id" == "torch_sdpa" && -n "$REFERENCE_DIR" ]]; then
    link_reference
    echo "=== reuse reference $REFERENCE_DIR as torch_sdpa ==="
    return 0
  fi
  local rendered
  rendered="$(render_config "$case_id" "$CHITUBENCH_GPUS_PER_NODE" "$CHITUBENCH_CFP")"
  export CHITUBENCH_CASE_ID="$case_id"
  export CHITU_RUN_TASK_ID="$case_id"
  if [[ -n "$flexcache_json" ]]; then
    export CHITUBENCH_FLEXCACHE_PARAMS="$flexcache_json"
  else
    unset CHITUBENCH_FLEXCACHE_PARAMS || true
  fi
  echo "=== [$EXPERIMENT_ID] $case_id gpus=$CHITUBENCH_GPUS_PER_NODE cfp=$CHITUBENCH_CFP ==="
  "$CHITU_BIN" run "$rendered" --gpus-per-node "$CHITUBENCH_GPUS_PER_NODE" --cfp "$CHITUBENCH_CFP"
  "$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
    --experiment-id "$EXPERIMENT_ID" \
    --title "$PLOT_TITLE" \
    --allow-partial
}

fc() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1]), separators=(",",":")))' "$1"
}

# Baseline (no flexcache)
run_case torch_sdpa ""

# Official MeanCache references
run_case qwen_meancache25_50_cfp2 "$(fc '{"strategy":"meancache","fresh_steps":25,"warmup":0,"cooldown":0,"use_jvp":true}')"
run_case qwen_meancache17_50_cfp2 "$(fc '{"strategy":"meancache","fresh_steps":17,"warmup":0,"cooldown":0,"use_jvp":true}')"

# FreeCache: MeanCache JVP extrapolation + train-free drift schedule
run_case qwen_freecache_tol010 "$(fc '{"strategy":"freecache","tol":0.10,"max_gap":8,"jvp_order":1,"warmup":0,"cooldown":0}')"
run_case qwen_freecache_tol015 "$(fc '{"strategy":"freecache","tol":0.15,"max_gap":8,"jvp_order":1,"warmup":0,"cooldown":0}')"
run_case qwen_freecache_tol020 "$(fc '{"strategy":"freecache","tol":0.20,"max_gap":8,"jvp_order":1,"warmup":0,"cooldown":0}')"
run_case qwen_freecache_tol030 "$(fc '{"strategy":"freecache","tol":0.30,"max_gap":8,"jvp_order":1,"warmup":0,"cooldown":0}')"

unset CHITUBENCH_FLEXCACHE_PARAMS

HPSV3_CONFIG="${CHITUBENCH_HPSV3_CONFIG:-$BENCH_DIR/results/hpsv3_assets/HPSv3_7B.local.yaml}"
HPSV3_CHECKPOINT="${CHITUBENCH_HPSV3_CHECKPOINT:-$BENCH_DIR/results/hpsv3_assets/HPSv3.chitu_compat.safetensors}"

run_quality_on_gpu() {
  local -a args=("$@")
  # HPSv3/accelerate 会重新 init_process_group；清掉上一轮 chitu run 残留的
  # MASTER_PORT / RANK 等环境变量，否则容易 EADDRINUSE。
  local fresh_port
  fresh_port="$("$PYTHON_BIN" -c 'import random; print(random.randint(52000, 62000))')"
  env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
    -u GROUP_RANK -u LOCAL_WORLD_SIZE -u TORCHELASTIC_RUN_ID \
    MASTER_PORT="$fresh_port" \
    CHITU_PROJECT_ROOT="$ROOT_DIR" \
    CHITU_PYTHON_BIN="$PYTHON_BIN" \
    SRUN_JOB_NAME="qwen-freecache-quality" \
    "$ROOT_DIR/script/srun_direct.sh" 1 1 "${args[@]}"
}

if [[ "${CHITUBENCH_SKIP_HPSV3:-1}" == "1" ]]; then
  "$PYTHON_BIN" "$BENCH_DIR/scripts/evaluate_quality.py" "$RESULT_ROOT" --skip-hpsv3 \
    --origin-dir "${REFERENCE_DIR:-$RESULT_ROOT}"
elif [[ -f "$HPSV3_CONFIG" && -f "$HPSV3_CHECKPOINT" ]]; then
  run_quality_on_gpu \
    "$BENCH_DIR/scripts/evaluate_quality.py" "$RESULT_ROOT" \
    --origin-dir "${REFERENCE_DIR:-$RESULT_ROOT}" \
    --hpsv3-config-path "$HPSV3_CONFIG" \
    --hpsv3-checkpoint-path "$HPSV3_CHECKPOINT"
else
  "$PYTHON_BIN" "$BENCH_DIR/scripts/evaluate_quality.py" "$RESULT_ROOT" --skip-hpsv3 \
    --origin-dir "${REFERENCE_DIR:-$RESULT_ROOT}"
  echo "HPSv3 skipped (missing assets). Set CHITUBENCH_HPSV3_* or CHITUBENCH_SKIP_HPSV3=1." >&2
fi

"$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --title "$PLOT_TITLE"
"$PYTHON_BIN" "$BENCH_DIR/scripts/make_contact_sheet.py" "$RESULT_ROOT" \
  --seed "$CHITUBENCH_BASE_SEED" \
  --title "$CONTACT_TITLE" \
  --experiment-id "$EXPERIMENT_ID"

mkdir -p "$BENCH_DIR/plots/$EXPERIMENT_ID"
cp "$RESULT_ROOT/plots/speed_quality.png" "$BENCH_DIR/plots/$EXPERIMENT_ID/speed_quality_${RUN_ID}.png" 2>/dev/null || true
cp "$RESULT_ROOT/visuals/contact_sheet.png" "$BENCH_DIR/plots/$EXPERIMENT_ID/contact_sheet_${RUN_ID}.png" 2>/dev/null || true

echo "Done. Results: $RESULT_ROOT"
