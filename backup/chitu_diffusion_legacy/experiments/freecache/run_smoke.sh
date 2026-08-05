#!/usr/bin/env bash
# FreeCache smoke: 4 steps, 1 seed, 1 GPU. Uses chitu run -> srun_direct.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/ChituBench"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
PYTHON_BIN="${CHITU_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
RUN_ID="${CHITUBENCH_RUN_ID:-freecache_smoke_$(date +%Y%m%d_%H%M)}"
RESULT_ROOT="$BENCH_DIR/results/qwen_image_freecache/$RUN_ID"
BASE_CONFIG="$BENCH_DIR/configs/qwen_image/attention/torch_sdpa.yaml"

export CHITUBENCH_PROMPT_FILE="${CHITUBENCH_PROMPT_FILE:-$BENCH_DIR/prompts/qwen_image_attention.json}"
export CHITUBENCH_STEPS="${CHITUBENCH_STEPS:-4}"
export CHITUBENCH_NUM_SEEDS="${CHITUBENCH_NUM_SEEDS:-1}"
export CHITUBENCH_BASE_SEED="${CHITUBENCH_BASE_SEED:-42}"
export CHITUBENCH_WARMUP_RUNS="${CHITUBENCH_WARMUP_RUNS:-0}"
export CHITUBENCH_IMAGE_SIZE="${CHITUBENCH_IMAGE_SIZE:-1328,1328}"
export CHITUBENCH_GPUS_PER_NODE="${CHITUBENCH_GPUS_PER_NODE:-1}"
export CHITUBENCH_CFP="${CHITUBENCH_CFP:-1}"

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
cfg.setdefault("launch", {})["tag"] = f"chitubench-qwen-freecache-smoke-{case_id}"
cfg["launch"]["gpus_per_node"] = int(gpus)
cfg.setdefault("launch", {}).setdefault("srun", {})["job_name"] = f"freecache-smoke-{case_id}"
cfg.setdefault("parallel", {})["cfp"] = int(cfp)
cfg["parallel"]["up"] = 1
cfg.setdefault("output", {})["root_dir"] = result_root
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(output)
PY
}

run_case() {
  local case_id="$1"
  local flexcache_json="$2"
  local rendered
  rendered="$(render_config "$case_id" "$CHITUBENCH_GPUS_PER_NODE" "$CHITUBENCH_CFP")"
  export CHITUBENCH_CASE_ID="$case_id"
  export CHITU_RUN_TASK_ID="$case_id"
  export CHITUBENCH_FLEXCACHE_PARAMS="$flexcache_json"
  echo "=== FreeCache smoke: $case_id (steps=$CHITUBENCH_STEPS, gpus=$CHITUBENCH_GPUS_PER_NODE) ==="
  "$CHITU_BIN" run "$rendered" --gpus-per-node "$CHITUBENCH_GPUS_PER_NODE" --cfp "$CHITUBENCH_CFP"
}

echo "=== CPU unit tests ==="
"$PYTHON_BIN" -m pytest "$ROOT_DIR/test/test_freecache_core.py" -q

run_case qwen_freecache_smoke '{"strategy":"freecache","tol":0.15,"max_gap":8,"jvp_order":1,"warmup":0,"cooldown":0}'

echo "Smoke OK. Results: $RESULT_ROOT"
