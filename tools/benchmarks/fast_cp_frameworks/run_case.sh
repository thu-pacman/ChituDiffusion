#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
    echo "Usage: $0 <chitu|xdit|sglang> <method> <cp> <sequence> [heads] [head_dim] [e2e|communication|attention]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
FRAMEWORK=$1
METHOD=$2
CP=$3
SEQUENCE=$4
HEADS=${5:-40}
HEAD_DIM=${6:-128}
COMPONENT=${7:-e2e}
ENV_ROOT=${FAST_CP_BENCH_ENV_ROOT:-$PROJECT_ROOT/.venv-fast-cp-frameworks}
SOURCE_ROOT=${FAST_CP_BENCH_SOURCE_ROOT:-$PROJECT_ROOT/refs/fast-cp-frameworks}
OUTPUT_ROOT=${FAST_CP_BENCH_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/fast-cp/qkvo}

case "$FRAMEWORK" in
    chitu)
        PYTHON_BIN=${CHITU_PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
        RUNNER=$SCRIPT_DIR/chitu.py
        FRAMEWORK_SOURCE=$PROJECT_ROOT
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
export CHITU_PYTHON_BIN=$PYTHON_BIN
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS:---exclusive}
export SRUN_JOB_NAME=${SRUN_JOB_NAME:-fast-cp-${FRAMEWORK}-${METHOD}}

"$PROJECT_ROOT/tools/cluster/srun_direct.sh" 1 "$CP" "$RUNNER" \
    --method "$METHOD" \
    --sequence "$SEQUENCE" \
    --heads "$HEADS" \
    --head-dim "$HEAD_DIM" \
    --component "$COMPONENT" \
    --source-root "$FRAMEWORK_SOURCE" \
    --output "$OUTPUT"
