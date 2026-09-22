#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/qwen-same-node-20260918}
WARMUP=${FAST_CP_BENCH_WARMUP:-3}
ITERATIONS=${FAST_CP_BENCH_ITERATIONS:-10}
REPEATS=${FAST_CP_BENCH_REPEATS:-3}

export FAST_CP_BENCH_WARMUP=$WARMUP
export FAST_CP_BENCH_ITERATIONS=$ITERATIONS
export FAST_CP_BENCH_REPEATS=$REPEATS

report_is_complete() {
    local report=$1
    [[ -s "$report" ]] || return 1
    "$PROJECT_ROOT/.venv/bin/python" -c \
        'import json, sys; sys.exit(json.load(open(sys.argv[1])).get("status") != "complete")' \
        "$report" >/dev/null 2>&1
}

run_case() {
    local report=$1
    shift
    if report_is_complete "$report"; then
        echo "Skipping complete report: $report"
        return
    fi
    "$@"
}

chitu_root=$OUTPUT_ROOT/chitu
sglang_root=$OUTPUT_ROOT/sglang

for cp in 2 4 8; do
    for method in fast_agkv fast_ulysses; do
        run_case \
            "$chitu_root/chitu-qwenimage-${method}-s4096-t512-cp${cp}.json" \
            "$SCRIPT_DIR/run_local_joint_case.sh" \
            qwenimage "$method" "$cp" 4096 512 "$chitu_root"
    done
    for method in kv_gather ulysses; do
        run_case \
            "$sglang_root/sglang-qwenimage-${method}-s4096-t512-cp${cp}.json" \
            "$SCRIPT_DIR/run_local_sglang_joint_case.sh" \
            qwenimage "$method" "$cp" 4096 512 "$sglang_root"
    done
done
