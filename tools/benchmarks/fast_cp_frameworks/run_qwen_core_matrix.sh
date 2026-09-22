#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/qwen-core-same-node-20260918}

export FAST_CP_BENCH_WARMUP=${FAST_CP_BENCH_WARMUP:-3}
export FAST_CP_BENCH_ITERATIONS=${FAST_CP_BENCH_ITERATIONS:-10}
export FAST_CP_BENCH_REPEATS=${FAST_CP_BENCH_REPEATS:-3}

run_case() {
    local framework=$1
    local method=$2
    local cp=$3
    local component=$4
    "$SCRIPT_DIR/run_local_case.sh" \
        "$framework" "$method" "$cp" 4608 24 128 "$component" "$OUTPUT_ROOT"
}

for cp in 2 4 8; do
    run_case chitu fast_agkv "$cp" e2e
    run_case chitu fast_ulysses "$cp" e2e
    run_case sglang kv_gather "$cp" e2e
    run_case sglang ulysses "$cp" e2e
done

for component in communication attention; do
    run_case chitu fast_agkv 4 "$component"
    run_case chitu fast_ulysses 4 "$component"
    run_case sglang kv_gather 4 "$component"
    run_case sglang ulysses 4 "$component"
done
