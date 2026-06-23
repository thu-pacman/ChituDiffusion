#!/usr/bin/env bash
# Flux2-klein-4B Parallel Sampler ablation: parallel DiT only vs parallel DiT +
# parallel sampler at the same GPU layout. Parallel sampler keeps each rank's
# latent shard local across scheduler steps (toggled with
# CHITU_FLUX2_PERSISTENT_CP_LATENTS=1/0). Flux2-klein has no CFG parallel, so the
# context-parallel layout is pure Ulysses (up=gpus). With only 4 denoise steps the
# DiT is tiny, so the per-step latent gather is a relatively larger share of the
# loop than on a 50-step model -- this experiment measures whether that makes the
# parallel-sampler saving more visible.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/ChituBench"
RUN_ID="${CHITUBENCH_RUN_ID:-flux2_klein_sampler_$(date +%Y%m%d_%H%M)}"
EXPERIMENT_ID="flux2_klein_sampler"
RESULT_ROOT="$BENCH_DIR/results/$EXPERIMENT_ID/$RUN_ID"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
PYTHON_BIN="${CHITU_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
BASE_CONFIG="$BENCH_DIR/configs/flux2_klein/sequence_parallel/baseline_1gpu.yaml"

export CHITUBENCH_PROMPT_FILE="${CHITUBENCH_PROMPT_FILE:-$BENCH_DIR/prompts/flux1_attention.json}"
export CHITUBENCH_STEPS="${CHITUBENCH_STEPS:-4}"
export CHITUBENCH_NUM_SEEDS="${CHITUBENCH_NUM_SEEDS:-3}"
export CHITUBENCH_BASE_SEED="${CHITUBENCH_BASE_SEED:-42}"
export CHITUBENCH_WARMUP_RUNS="${CHITUBENCH_WARMUP_RUNS:-1}"
export CHITUBENCH_ATTN_TYPE="${CHITUBENCH_ATTN_TYPE:-flash}"
CHITUBENCH_PARTITION="${CHITUBENCH_PARTITION:-debug}"

mkdir -p "$RESULT_ROOT/configs"

render_config() {
  local case_id="$1"
  local gpus="$2"
  local up="$3"
  local output_config="$RESULT_ROOT/configs/${case_id}.yaml"
  "$PYTHON_BIN" - "$BASE_CONFIG" "$output_config" "$RESULT_ROOT" "$case_id" "$gpus" "$up" "$CHITUBENCH_ATTN_TYPE" "$CHITUBENCH_PARTITION" <<'PY'
import sys
from pathlib import Path

import yaml

source, output, result_root, case_id, gpus, up, attn_type, partition = sys.argv[1:9]
cfg = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
gpus = int(gpus)
up = int(up)
cfg.setdefault("launch", {})["tag"] = f"chitubench-flux2-sampler-{case_id}"
cfg["launch"]["gpus_per_node"] = gpus
cfg.setdefault("launch", {}).setdefault("srun", {})["job_name"] = f"chitubench-flux2-sampler-{case_id}"
cfg["launch"]["srun"]["partition"] = partition
cfg.setdefault("parallel", {})["cfp"] = 1
cfg["parallel"]["up"] = up
cfg.setdefault("infer", {})["attn_type"] = attn_type
cfg.setdefault("output", {})["root_dir"] = result_root
cfg["output"]["memory"] = True
cfg["output"]["log_ranks"] = [0]
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  echo "$output_config"
}

run_case() {
  local case_id="$1"
  local gpus="$2"
  local up="$3"
  local sampler="$4"
  local rendered
  rendered="$(render_config "$case_id" "$gpus" "$up")"
  export CHITUBENCH_CASE_ID="$case_id"
  export CHITU_RUN_TASK_ID="$case_id"
  export CHITU_FLUX2_PERSISTENT_CP_LATENTS="$sampler"
  echo "=== ChituBench [$EXPERIMENT_ID] $case_id gpus=$gpus up=$up sampler=$sampler ==="
  "$CHITU_BIN" run "$rendered" --gpus-per-node "$gpus"
  "$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
    --experiment-id "$EXPERIMENT_ID" \
    --title "Flux2-klein Parallel Sampler Ablation" \
    --no-point-labels \
    --allow-partial
}

declare -A CASE_SELECTED=()
if [[ -n "${CHITUBENCH_CASES:-}" ]]; then
  IFS=',' read -ra REQUESTED_CASES <<< "$CHITUBENCH_CASES"
  for requested_case in "${REQUESTED_CASES[@]}"; do
    requested_case="${requested_case// /}"
    [[ -n "$requested_case" ]] && CASE_SELECTED["$requested_case"]=1
  done
fi

maybe_run_case() {
  local case_id="$1"
  shift
  if [[ ${#CASE_SELECTED[@]} -gt 0 && -z "${CASE_SELECTED[$case_id]:-}" ]]; then
    return
  fi
  run_case "$case_id" "$@"
}

#                case_id            gpus up sampler
maybe_run_case   baseline_1gpu      1    1  1
maybe_run_case   cp2_dit_only       2    2  0
maybe_run_case   cp2_dit_sampler    2    2  1
maybe_run_case   cp4_dit_only       4    4  0
maybe_run_case   cp4_dit_sampler    4    4  1
maybe_run_case   cp8_dit_only       8    8  0
maybe_run_case   cp8_dit_sampler    8    8  1

"$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --title "Flux2-klein Parallel Sampler Ablation" \
  --no-point-labels

echo "Results: $RESULT_ROOT"
