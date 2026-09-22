#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
    echo "Usage: $0 <framework> <method> <cp> <sequence> <heads> <head_dim> <component> [output_name]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PARTITION_ARGS=()
if [[ -n "${SRUN_PARTITION:-}" ]]; then
    PARTITION_ARGS=(--partition "$SRUN_PARTITION")
fi
FRAMEWORK=$1
METHOD=$2
CP=$3
SEQUENCE=$4
HEADS=$5
HEAD_DIM=$6
COMPONENT=$7
OUTPUT_NAME=${8:-${FRAMEWORK}-${METHOD}-${COMPONENT}-h${HEADS}-s${SEQUENCE}-cp${CP}}
OUTPUT_ROOT=$PROJECT_ROOT/outputs/fast-cp/traces
NODE_ARGS=()
if [[ -n "${FAST_CP_BENCH_NODE:-}" ]]; then
    NODE_ARGS=(--nodelist "$FAST_CP_BENCH_NODE")
fi
mkdir -p "$OUTPUT_ROOT"

exec srun \
    "${PARTITION_ARGS[@]}" \
    --exclusive \
    --job-name "${USER:-user}-fast-cp-trace" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task 192 \
    --mem 1937152 \
    --gres gpu:8 \
    "${NODE_ARGS[@]}" \
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
        --output "$OUTPUT_ROOT/$OUTPUT_NAME" \
    "$SCRIPT_DIR/run_local_case.sh" \
        "$FRAMEWORK" "$METHOD" "$CP" "$SEQUENCE" "$HEADS" "$HEAD_DIM" \
        "$COMPONENT" "$PROJECT_ROOT/outputs/fast-cp/trace-timings"
