#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
    echo "Usage: $0 <qwenimage|zimage> <method> <cp> <image_sequence> [text_sequence] [output_root]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL=$1
METHOD=$2
CP=$3
SEQUENCE=$4
TEXT_SEQUENCE=${5:-512}
OUTPUT_ROOT=${6:-$PROJECT_ROOT/outputs/fast-cp/joint-attention-same-node}
PYTHON_BIN=${CHITU_PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
source "$PROJECT_ROOT/tools/install/fast_cp_env.sh"
WARMUP=${FAST_CP_BENCH_WARMUP:-5}
ITERATIONS=${FAST_CP_BENCH_ITERATIONS:-15}
REPEATS=${FAST_CP_BENCH_REPEATS:-5}
EXTRA_ARGS=()
if [[ "${CHITU_Z_SHARDED_TEXT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--z-sharded-text)
fi

OUTPUT=$OUTPUT_ROOT/chitu-${MODEL}-${METHOD}-s${SEQUENCE}-t${TEXT_SEQUENCE}-cp${CP}.json
export NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-0}
export NCCL_GRAPH_REGISTER=${NCCL_GRAPH_REGISTER:-0}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}

"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$CP" \
    "$SCRIPT_DIR/chitu_joint.py" \
    --model "$MODEL" \
    --method "$METHOD" \
    --sequence "$SEQUENCE" \
    --text-sequence "$TEXT_SEQUENCE" \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS" \
    --repeats "$REPEATS" \
    --source-root "$PROJECT_ROOT" \
    "${EXTRA_ARGS[@]}" \
    --output "$OUTPUT"
