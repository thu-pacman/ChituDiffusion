#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
    echo "Usage: $0 <framework> <method> <cp> <sequence> <heads> <head_dim> <component> [output_root]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
FRAMEWORK=$1
METHOD=$2
CP=$3
SEQUENCE=$4
HEADS=$5
HEAD_DIM=$6
COMPONENT=$7
OUTPUT_ROOT=${8:-$PROJECT_ROOT/outputs/fast-cp/qkvo-same-node}
WARMUP=${FAST_CP_BENCH_WARMUP:-5}
ITERATIONS=${FAST_CP_BENCH_ITERATIONS:-15}
REPEATS=${FAST_CP_BENCH_REPEATS:-5}
ENV_ROOT=${FAST_CP_BENCH_ENV_ROOT:-$PROJECT_ROOT/.venv-fast-cp-frameworks}
SOURCE_ROOT=${FAST_CP_BENCH_SOURCE_ROOT:-$PROJECT_ROOT/refs/fast-cp-frameworks}
EXTRA_PYTHONPATH=

case "$FRAMEWORK" in
    chitu)
        PYTHON_BIN=${CHITU_PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
        RUNNER=$SCRIPT_DIR/chitu.py
        FRAMEWORK_SOURCE=$PROJECT_ROOT
        FAST_CP_BUILD_ROOT=${FAST_CP_BUILD_ROOT:-$PROJECT_ROOT/refs/fast-cp-sm90-sm120}
        FLASH_ATTENTION_ROOT=${FLASH_ATTENTION_ROOT:-$PROJECT_ROOT/refs/kernels/flash-attention}
        EXTRA_PYTHONPATH=:$FAST_CP_BUILD_ROOT:$FLASH_ATTENTION_ROOT
        export LD_LIBRARY_PATH="$FAST_CP_BUILD_ROOT/.deps/nvidia/nvshmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ;;
    xdit)
        PYTHON_BIN=$ENV_ROOT/xdit/bin/python
        RUNNER=$SCRIPT_DIR/xdit.py
        FRAMEWORK_SOURCE=$SOURCE_ROOT/xdit
        ;;
    sglang)
        PYTHON_BIN=$ENV_ROOT/sglang/bin/python
        RUNNER=$SCRIPT_DIR/sglang_diffusion.py
        FRAMEWORK_SOURCE=$SOURCE_ROOT/sglang
        ;;
    *)
        echo "Unknown framework: $FRAMEWORK" >&2
        exit 2
        ;;
esac

test -x "$PYTHON_BIN"
SUFFIX=
if [[ "$COMPONENT" != e2e ]]; then
    SUFFIX=-$COMPONENT
fi
OUTPUT=$OUTPUT_ROOT/${FRAMEWORK}-${METHOD}${SUFFIX}-h${HEADS}-s${SEQUENCE}-cp${CP}.json
export PYTHONPATH="$PROJECT_ROOT$EXTRA_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-0}
export NCCL_GRAPH_REGISTER=${NCCL_GRAPH_REGISTER:-0}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=${TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN:-0}

PROFILE_ARGS=()
if [[ "${FAST_CP_BENCH_PROFILE:-0}" == 1 ]]; then
    PROFILE_ARGS=(--profile)
fi

"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$CP" \
    "$RUNNER" \
    --method "$METHOD" \
    --sequence "$SEQUENCE" \
    --heads "$HEADS" \
    --head-dim "$HEAD_DIM" \
    --component "$COMPONENT" \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS" \
    --repeats "$REPEATS" \
    "${PROFILE_ARGS[@]}" \
    --source-root "$FRAMEWORK_SOURCE" \
    --output "$OUTPUT"
