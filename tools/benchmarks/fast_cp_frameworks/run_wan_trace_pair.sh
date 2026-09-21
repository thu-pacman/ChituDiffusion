#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL=${1:-wan14b}
SEQUENCE=${2:-4096}
CP=${3:-8}
TRACE_ROOT=${4:-$PROJECT_ROOT/outputs/fast-cp/traces}
mkdir -p "$TRACE_ROOT"

for schedule in native serialized_ce; do
    /usr/bin/env \
        FAST_CP_BENCH_WARMUP=3 \
        FAST_CP_BENCH_ITERATIONS=3 \
        FAST_CP_BENCH_REPEATS=1 \
        FAST_CP_BENCH_PROFILE=1 \
    nsys profile \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --sample=none \
        --cpuctxsw=none \
        --trace=cuda,nvtx,cudnn \
        --force-overwrite=true \
        --stats=true \
        --output "$TRACE_ROOT/${MODEL}-fast_ulysses-${schedule}-s${SEQUENCE}-cp${CP}" \
    "$SCRIPT_DIR/run_local_wan_case.sh" \
        "$MODEL" fast_ulysses "$CP" "$SEQUENCE" "$schedule" \
        "$PROJECT_ROOT/outputs/fast-cp/trace-timings"
done
