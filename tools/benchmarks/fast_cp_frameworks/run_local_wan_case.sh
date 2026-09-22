#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: $0 <wan1.3b|wan14b> <method> <cp> <sequence> <native|serialized_ce|sync> [output_root]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL=$1
METHOD=$2
CP=$3
SEQUENCE=$4
SCHEDULE=$5
OUTPUT_ROOT=${6:-$PROJECT_ROOT/outputs/fast-cp/model-attention-same-node}
PYTHON_BIN=${CHITU_PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
source "$PROJECT_ROOT/tools/install/fast_cp_env.sh"
WARMUP=${FAST_CP_BENCH_WARMUP:-5}
ITERATIONS=${FAST_CP_BENCH_ITERATIONS:-15}
REPEATS=${FAST_CP_BENCH_REPEATS:-5}

case "$SCHEDULE" in
    native) SERIAL_ARG=() ;;
    serialized_ce) SERIAL_ARG=(--serialized-async) ;;
    sync|serial) SERIAL_ARG=(--serial) ;;
    *) echo "schedule must be native, serialized_ce, or sync" >&2; exit 2 ;;
esac

OUTPUT=$OUTPUT_ROOT/chitu-${MODEL}-${METHOD}-${SCHEDULE}-s${SEQUENCE}-cp${CP}.json
export NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-0}
export NCCL_GRAPH_REGISTER=${NCCL_GRAPH_REGISTER:-0}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}

PROFILE_ARGS=()
if [[ "${FAST_CP_BENCH_PROFILE:-0}" == 1 ]]; then
    PROFILE_ARGS=(--profile)
fi

"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$CP" \
    "$SCRIPT_DIR/chitu_wan.py" \
    --model "$MODEL" \
    --method "$METHOD" \
    --sequence "$SEQUENCE" \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS" \
    --repeats "$REPEATS" \
    "${PROFILE_ARGS[@]}" \
    --source-root "$PROJECT_ROOT" \
    --output "$OUTPUT" \
    "${SERIAL_ARG[@]}"
