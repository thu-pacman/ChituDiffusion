#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: $0 <wan1.3b|wan14b|qwenimage|zimage> <ulysses|kv_gather> <cp> <image_sequence> <text_sequence> [output_root]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL=$1
METHOD=$2
CP=$3
SEQUENCE=$4
TEXT_SEQUENCE=$5
OUTPUT_ROOT=${6:-$PROJECT_ROOT/outputs/fast-cp/sglang-joint-same-node}
ENV_ROOT=${FAST_CP_BENCH_ENV_ROOT:-$PROJECT_ROOT/.venv-fast-cp-frameworks}
SOURCE_ROOT=${FAST_CP_BENCH_SOURCE_ROOT:-$PROJECT_ROOT/refs/fast-cp-frameworks}
PYTHON_BIN=$ENV_ROOT/sglang/bin/python
WARMUP=${FAST_CP_BENCH_WARMUP:-5}
ITERATIONS=${FAST_CP_BENCH_ITERATIONS:-15}
REPEATS=${FAST_CP_BENCH_REPEATS:-5}
OUTPUT=$OUTPUT_ROOT/sglang-${MODEL}-${METHOD}-s${SEQUENCE}-t${TEXT_SEQUENCE}-cp${CP}.json

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ENV_ROOT/sglang/bin:$PATH"
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=${TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN:-0}

"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$CP" \
    "$SCRIPT_DIR/sglang_joint.py" \
    --model "$MODEL" \
    --method "$METHOD" \
    --sequence "$SEQUENCE" \
    --text-sequence "$TEXT_SEQUENCE" \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS" \
    --repeats "$REPEATS" \
    --source-root "$SOURCE_ROOT/sglang" \
    --output "$OUTPUT"
