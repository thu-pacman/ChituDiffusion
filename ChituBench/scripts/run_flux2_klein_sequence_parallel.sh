#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/ChituBench"
RUN_ID="${CHITUBENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ID="flux2_klein_sequence_parallel"
RESULT_ROOT="$BENCH_DIR/results/$EXPERIMENT_ID/$RUN_ID"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
PYTHON_BIN="${CHITU_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
PLOT_TITLE="Flux2-klein-4B Sequence Parallel Scaling"
CONFIG_DIR="$BENCH_DIR/configs/flux2_klein/sequence_parallel"

PROMPT_FILE="${CHITUBENCH_PROMPT_FILE:-$BENCH_DIR/prompts/flux1_attention.json}"
export CHITUBENCH_PROMPT_FILE="$PROMPT_FILE"
export CHITUBENCH_STEPS="${CHITUBENCH_STEPS:-4}"
export CHITUBENCH_NUM_SEEDS="${CHITUBENCH_NUM_SEEDS:-3}"
export CHITUBENCH_BASE_SEED="${CHITUBENCH_BASE_SEED:-42}"
export CHITUBENCH_WARMUP_RUNS="${CHITUBENCH_WARMUP_RUNS:-1}"
export CHITUBENCH_CASES="${CHITUBENCH_CASES:-baseline_1gpu ring_2gpu ulysses_2gpu ring_4gpu usp_r2u2_4gpu ulysses_4gpu ring_8gpu usp_r4u2_8gpu ulysses_8gpu}"

mkdir -p "$RESULT_ROOT/configs"

render_config() {
  local source_config="$1"
  local output_config="$2"
  "$PYTHON_BIN" - "$source_config" "$output_config" "$RESULT_ROOT" <<'PY'
import sys
from pathlib import Path
import yaml

source, output, result_root = sys.argv[1:4]
cfg = yaml.safe_load(Path(source).read_text())
cfg.setdefault("output", {})["root_dir"] = result_root
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
}

run_case() {
  local case_id="$1"
  local gpus="$2"
  if [[ ! " $CHITUBENCH_CASES " =~ " $case_id " ]]; then
    return 0
  fi
  local config="$CONFIG_DIR/${case_id}.yaml"
  local rendered="$RESULT_ROOT/configs/${case_id}.yaml"
  render_config "$config" "$rendered"
  export CHITUBENCH_CASE_ID="$case_id"
  export CHITU_RUN_TASK_ID="$case_id"
  echo "=== ChituBench [$EXPERIMENT_ID] $case_id (${gpus} GPU) ==="
  "$CHITU_BIN" run "$rendered" --gpus-per-node "$gpus"
  "$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
    --experiment-id "$EXPERIMENT_ID" \
    --title "$PLOT_TITLE" \
    --allow-partial
}

run_case baseline_1gpu 1
run_case ring_2gpu 2
run_case ulysses_2gpu 2
run_case ring_4gpu 4
run_case usp_r2u2_4gpu 4
run_case ulysses_4gpu 4
run_case ring_8gpu 8
run_case usp_r4u2_8gpu 8
run_case ulysses_8gpu 8

"$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --title "$PLOT_TITLE"
"$PYTHON_BIN" "$BENCH_DIR/scripts/plot_parallel_scaling.py" "$RESULT_ROOT" \
  --title "$PLOT_TITLE"

mkdir -p "$BENCH_DIR/plots/$EXPERIMENT_ID"
cp "$RESULT_ROOT/plots/parallel_scaling.png" "$BENCH_DIR/plots/$EXPERIMENT_ID/parallel_scaling_${RUN_ID}.png"

echo "Results: $RESULT_ROOT"
