#!/usr/bin/env bash
# Qwen-Image Parallel Sampler ablation: compare "parallel DiT only" against
# "parallel DiT + parallel sampler" at the same GPU layout. Parallel sampler keeps
# the latent shards local across scheduler steps (Qwen `local_latent` plan), which
# is toggled with CHITU_QWEN_PERSISTENT_CP_LATENTS (1=on / 0=off). It only has an
# effect when cp_size > 1, so every ablation case uses a context-parallel layout.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/ChituBench"
RUN_ID="${CHITUBENCH_RUN_ID:-qwen_image_sampler_$(date +%Y%m%d_%H%M)}"
EXPERIMENT_ID="qwen_image_sampler"
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
export CHITUBENCH_ATTN_TYPE="${CHITUBENCH_ATTN_TYPE:-flash_attn}"
# AGCP vs UCP is a separate experiment; pin a single backend here so the only
# variable across each pair is the sampler toggle.
export CHITUBENCH_CP_BACKEND="${CHITUBENCH_CP_BACKEND:-agcp}"
CHITUBENCH_PARTITION="${CHITUBENCH_PARTITION:-ci}"

mkdir -p "$RESULT_ROOT/configs"

render_config() {
  local case_id="$1"
  local gpus="$2"
  local cfp="$3"
  local up="$4"
  local cp_backend="$5"
  local output_config="$RESULT_ROOT/configs/${case_id}.yaml"
  "$PYTHON_BIN" - "$BASE_CONFIG" "$output_config" "$RESULT_ROOT" "$case_id" "$gpus" "$cfp" "$up" "$CHITUBENCH_ATTN_TYPE" "$cp_backend" "$CHITUBENCH_PARTITION" <<'PY'
import sys
from pathlib import Path

import yaml

source, output, result_root, case_id, gpus, cfp, up, attn_type, cp_backend, partition = sys.argv[1:11]
cfg = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
gpus = int(gpus)
cfp = int(cfp)
up = int(up)
cfg.setdefault("launch", {})["tag"] = f"chitubench-qwen-image-sampler-{case_id}"
cfg["launch"]["gpus_per_node"] = gpus
cfg.setdefault("launch", {}).setdefault("srun", {})["job_name"] = f"chitubench-qwen-sampler-{case_id}"
cfg["launch"]["srun"]["partition"] = partition
cfg.setdefault("parallel", {})["cfp"] = cfp
cfg["parallel"]["up"] = up
cfg.setdefault("infer", {})["attn_type"] = attn_type
cfg["infer"]["cp_backend"] = cp_backend
cfg.setdefault("output", {})["root_dir"] = result_root
cfg["output"]["log_ranks"] = [0]
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  echo "$output_config"
}

run_case() {
  local case_id="$1"
  local gpus="$2"
  local cfp="$3"
  local up="$4"
  local sampler="$5"
  local rendered
  rendered="$(render_config "$case_id" "$gpus" "$cfp" "$up" "$CHITUBENCH_CP_BACKEND")"
  export CHITUBENCH_CASE_ID="$case_id"
  export CHITU_RUN_TASK_ID="$case_id"
  export CHITU_QWEN_PERSISTENT_CP_LATENTS="$sampler"
  echo "=== ChituBench [$EXPERIMENT_ID] $case_id gpus=$gpus cfp=$cfp up=$up sampler=$sampler backend=$CHITUBENCH_CP_BACKEND ==="
  "$CHITU_BIN" run "$rendered" --gpus-per-node "$gpus"
  "$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
    --experiment-id "$EXPERIMENT_ID" \
    --title "Qwen-Image Parallel Sampler Ablation" \
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

#                case_id                gpus cfp up sampler
maybe_run_case   baseline_1gpu          1    1   1  1
maybe_run_case   cp2_dit_only           2    1   2  0
maybe_run_case   cp2_dit_sampler        2    1   2  1
maybe_run_case   cfp2cp2_dit_only       4    2   2  0
maybe_run_case   cfp2cp2_dit_sampler    4    2   2  1
maybe_run_case   cfp2cp4_dit_only       8    2   4  0
maybe_run_case   cfp2cp4_dit_sampler    8    2   4  1

"$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --title "Qwen-Image Parallel Sampler Ablation" \
  --no-point-labels

echo "Results: $RESULT_ROOT"
