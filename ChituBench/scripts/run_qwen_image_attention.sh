#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT_DIR/ChituBench"
RUN_ID="${CHITUBENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ID="qwen_image_attention"
RESULT_ROOT="$BENCH_DIR/results/$EXPERIMENT_ID/$RUN_ID"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
PYTHON_BIN="${CHITU_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
PLOT_TITLE="Qwen-Image Attention Backend Performance"
CONTACT_TITLE="Qwen-Image Attention Backend Visual Check"

PROMPT_FILE="${CHITUBENCH_PROMPT_FILE:-$BENCH_DIR/prompts/qwen_image_attention.json}"
export CHITUBENCH_PROMPT_FILE="$PROMPT_FILE"
export CHITUBENCH_STEPS="${CHITUBENCH_STEPS:-50}"
export CHITUBENCH_NUM_SEEDS="${CHITUBENCH_NUM_SEEDS:-3}"
export CHITUBENCH_BASE_SEED="${CHITUBENCH_BASE_SEED:-42}"
export CHITUBENCH_WARMUP_RUNS="${CHITUBENCH_WARMUP_RUNS:-1}"
export CHITUBENCH_IMAGE_SIZE="${CHITUBENCH_IMAGE_SIZE:-1328,1328}"

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
  local config="$2"
  local rendered="$RESULT_ROOT/configs/${case_id}.yaml"
  render_config "$config" "$rendered"
  export CHITUBENCH_CASE_ID="$case_id"
  export CHITU_RUN_TASK_ID="$case_id"
  echo "=== ChituBench [$EXPERIMENT_ID] $case_id ==="
  "$CHITU_BIN" run "$rendered" --gpus-per-node 1
  "$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
    --experiment-id "$EXPERIMENT_ID" \
    --title "$PLOT_TITLE" \
    --allow-partial
}

run_case torch_sdpa "$BENCH_DIR/configs/qwen_image/attention/torch_sdpa.yaml"
run_case torch_sdpa_math "$BENCH_DIR/configs/qwen_image/attention/torch_sdpa_math.yaml"
run_case flashinfer "$BENCH_DIR/configs/qwen_image/attention/flashinfer.yaml"
run_case sage "$BENCH_DIR/configs/qwen_image/attention/sage.yaml"
run_case sparge "$BENCH_DIR/configs/qwen_image/attention/sparge.yaml"

REFERENCE_DIR="$("$PYTHON_BIN" - "$RESULT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for run_dir in sorted(root.iterdir()):
    request_path = run_dir / "request_params.json"
    if not request_path.exists():
        continue
    payload = json.loads(request_path.read_text())
    requests = payload.get("requests") or []
    if any((req.get("params") or {}).get("role") == "torch_sdpa" for req in requests):
        print(run_dir)
        break
else:
    raise SystemExit("failed to locate torch_sdpa reference run")
PY
)"

HPSV3_CONFIG="${CHITUBENCH_HPSV3_CONFIG:-}"
HPSV3_CHECKPOINT="${CHITUBENCH_HPSV3_CHECKPOINT:-}"
if [[ -n "$HPSV3_CONFIG" && -n "$HPSV3_CHECKPOINT" ]]; then
  "$PYTHON_BIN" "$BENCH_DIR/scripts/evaluate_quality.py" "$RESULT_ROOT" \
    --origin-dir "$REFERENCE_DIR" \
    --hpsv3-config-path "$HPSV3_CONFIG" \
    --hpsv3-checkpoint-path "$HPSV3_CHECKPOINT"
else
  "$PYTHON_BIN" "$BENCH_DIR/scripts/evaluate_quality.py" "$RESULT_ROOT" \
    --origin-dir "$REFERENCE_DIR" \
    --skip-hpsv3
  echo "HPSv3 skipped. Set CHITUBENCH_HPSV3_CONFIG and CHITUBENCH_HPSV3_CHECKPOINT to enable it." >&2
fi
"$PYTHON_BIN" "$BENCH_DIR/scripts/collect.py" "$RESULT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --title "$PLOT_TITLE"
"$PYTHON_BIN" "$BENCH_DIR/scripts/make_contact_sheet.py" "$RESULT_ROOT" \
  --seed "$CHITUBENCH_BASE_SEED" \
  --title "$CONTACT_TITLE" \
  --cases torch_sdpa torch_sdpa_math flashinfer sage sparge

mkdir -p "$BENCH_DIR/plots/$EXPERIMENT_ID"
cp "$RESULT_ROOT/plots/speed_quality.png" "$BENCH_DIR/plots/$EXPERIMENT_ID/speed_quality_${RUN_ID}.png"
cp "$RESULT_ROOT/visuals/contact_sheet.png" "$BENCH_DIR/plots/$EXPERIMENT_ID/contact_sheet_${RUN_ID}.png"

echo "Results: $RESULT_ROOT"
