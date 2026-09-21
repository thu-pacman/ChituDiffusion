#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <wan1.3b|wan14b> <cp> <sequence> [output_root]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL=$1
CP=$2
SEQUENCE=$3
OUTPUT_ROOT=${4:-$PROJECT_ROOT/outputs/fast-cp/xdit-wan-same-node}
ENV_ROOT=${FAST_CP_BENCH_ENV_ROOT:-$PROJECT_ROOT/.venv-fast-cp-frameworks}
SOURCE_ROOT=${FAST_CP_BENCH_SOURCE_ROOT:-$PROJECT_ROOT/refs/fast-cp-frameworks}
PYTHON_BIN=$ENV_ROOT/xdit/bin/python
WARMUP=${FAST_CP_BENCH_WARMUP:-5}
ITERATIONS=${FAST_CP_BENCH_ITERATIONS:-15}
REPEATS=${FAST_CP_BENCH_REPEATS:-5}
OUTPUT=$OUTPUT_ROOT/xdit-${MODEL}-ulysses-s${SEQUENCE}-cp${CP}.json

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=${TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN:-0}

"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$CP" \
    "$SCRIPT_DIR/xdit_wan.py" \
    --model "$MODEL" \
    --sequence "$SEQUENCE" \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS" \
    --repeats "$REPEATS" \
    --source-root "$SOURCE_ROOT/xdit" \
    --output "$OUTPUT"
