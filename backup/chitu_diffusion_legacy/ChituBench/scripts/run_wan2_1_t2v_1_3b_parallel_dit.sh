#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/ChituBench"
RUN_ID="${CHITUBENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ID="wan2_1_t2v_1_3b_parallel_dit"
RESULT_ROOT="$BENCH_DIR/results/$EXPERIMENT_ID/$RUN_ID"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
PYTHON_BIN="${CHITU_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG="$BENCH_DIR/configs/wan2_1_t2v_1_3b/parallel_dit/base_flash.yaml"
PLOT_TITLE="Wan2.1 T2V 1.3B Parallel DiT Scaling"

PROMPT_FILE="${CHITUBENCH_PROMPT_FILE:-$BENCH_DIR/prompts/wan2_1_t2v_1_3b_attention.json}"
export CHITUBENCH_PROMPT_FILE="$PROMPT_FILE"
export CHITUBENCH_STEPS="${CHITUBENCH_STEPS:-50}"
export CHITUBENCH_NUM_SEEDS="${CHITUBENCH_NUM_SEEDS:-1}"
export CHITUBENCH_BASE_SEED="${CHITUBENCH_BASE_SEED:-42}"
export CHITUBENCH_WARMUP_RUNS="${CHITUBENCH_WARMUP_RUNS:-0}"
export CHITUBENCH_VIDEO_SIZE="${CHITUBENCH_VIDEO_SIZE:-832,480}"
export CHITUBENCH_FRAME_NUM="${CHITUBENCH_FRAME_NUM:-81}"
export CHITUBENCH_CASES="${CHITUBENCH_CASES:-baseline_1gpu cfp2_2gpu up2_2gpu cfp2up2_4gpu}"

mkdir -p "$RESULT_ROOT/configs"

render_config() {
  local case_id="$1"
  local nodes="$2"
  local gpus_per_node="$3"
  local cfp="$4"
  local up="$5"
  local ring_cudagraph="${6:-false}"
  local output_config="$RESULT_ROOT/configs/${case_id}.yaml"
  "$PYTHON_BIN" - "$CONFIG" "$output_config" "$RESULT_ROOT" "$case_id" "$nodes" "$gpus_per_node" "$cfp" "$up" "$ring_cudagraph" <<'PY'
import sys
from pathlib import Path
import yaml

source, output, result_root, case_id, nodes, gpus_per_node, cfp, up, ring_cudagraph = sys.argv[1:10]
cfg = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
cfg.setdefault("launch", {})["tag"] = f"chitubench-wan21-13b-parallel-{case_id}"
cfg["launch"]["num_nodes"] = int(nodes)
cfg["launch"]["gpus_per_node"] = int(gpus_per_node)
cfg.setdefault("launch", {}).setdefault("srun", {})["job_name"] = f"chitubench-wan21-parallel-{case_id}"
cfg.setdefault("parallel", {})["cfp"] = int(cfp)
cfg["parallel"]["up"] = int(up)
cfg.setdefault("output", {})["root_dir"] = result_root
cfg["output"]["log_ranks"] = [0]
if str(ring_cudagraph).lower() in {"1", "true", "yes", "on"}:
    cfg["overrides"] = list(cfg.get("overrides") or [])
    cfg["overrides"].append("infer.diffusion.ring_cudagraph=true")
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

run_case() {
  local case_id="$1"
  local nodes="$2"
  local gpus_per_node="$3"
  local cfp="$4"
  local up="$5"
  local ring_cudagraph="${6:-false}"
  if ! should_run_case "$case_id"; then
    echo "=== ChituBench [$EXPERIMENT_ID] skip $case_id ==="
    return 0
  fi
  local rendered
  rendered="$(render_config "$case_id" "$nodes" "$gpus_per_node" "$cfp" "$up" "$ring_cudagraph")"
  export CHITUBENCH_CASE_ID="$case_id"
  export CHITU_RUN_TASK_ID="$case_id"
  echo "=== ChituBench [$EXPERIMENT_ID] $case_id nodes=$nodes gpus_per_node=$gpus_per_node cfp=$cfp up=$up ring_cudagraph=$ring_cudagraph ==="
  if [[ "$ring_cudagraph" == "true" ]]; then
    export NCCL_GRAPH_MIXING_SUPPORT=1
  fi
  "$CHITU_BIN" run "$rendered" --num-nodes "$nodes" --gpus-per-node "$gpus_per_node" --cfp "$cfp"
  "$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
    --experiment-id "$EXPERIMENT_ID" \
    --title "$PLOT_TITLE" \
    --allow-partial
}

run_case baseline_1gpu 1 1 1 1
run_case cfp2_2gpu 1 2 2 1
run_case up2_2gpu 1 2 1 2
run_case cfp2up2_4gpu 1 4 2 2
run_case up4_4gpu 1 4 1 4
run_case cfp2up4_8gpu 1 8 2 4
run_case ring2up4_8gpu 1 8 1 4
run_case cfp2ring2up4_16gpu 2 8 2 4
run_case graph_ring4_4gpu 1 4 1 1 true
run_case cfp2graph_ring4_8gpu 1 8 2 1 true
run_case cfp2graph_ring8_16gpu 2 8 2 1 true
run_case cfp2graph_ring2up4_16gpu 2 8 2 4 true

"$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --title "$PLOT_TITLE" \
  --allow-partial
"$PYTHON_BIN" "$BENCH_DIR/scripts/plot_parallel_scaling.py" "$RESULT_ROOT" \
  --title "$PLOT_TITLE"

mkdir -p "$BENCH_DIR/plots/$EXPERIMENT_ID"
cp "$RESULT_ROOT/plots/parallel_scaling.png" "$BENCH_DIR/plots/$EXPERIMENT_ID/parallel_scaling_${RUN_ID}.png"

echo "Results: $RESULT_ROOT"
