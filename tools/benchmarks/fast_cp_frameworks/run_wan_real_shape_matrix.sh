#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/wan-real-shape-20260918}
WARMUP=${FAST_CP_BENCH_WARMUP:-3}
ITERATIONS=${FAST_CP_BENCH_ITERATIONS:-10}
REPEATS=${FAST_CP_BENCH_REPEATS:-3}

export FAST_CP_BENCH_WARMUP=$WARMUP
export FAST_CP_BENCH_ITERATIONS=$ITERATIONS
export FAST_CP_BENCH_REPEATS=$REPEATS

report_is_terminal() {
    local report=$1
    [[ -s "$report" ]] || return 1
    "$PROJECT_ROOT/.venv/bin/python" -c \
        'import json, sys; sys.exit(json.load(open(sys.argv[1])).get("status") not in {"complete", "unsupported"})' \
        "$report" >/dev/null 2>&1
}

run_case() {
    local report=$1
    shift
    if report_is_terminal "$report"; then
        echo "Skipping terminal report: $report"
        return
    fi
    "$@"
}

for spec in "wan1.3b 32760" "wan14b 75600"; do
    read -r model sequence <<<"$spec"

    chitu_root=$OUTPUT_ROOT/chitu
    run_case \
        "$chitu_root/chitu-${model}-nccl_agkv-native-s${sequence}-cp1.json" \
        "$SCRIPT_DIR/run_local_wan_case.sh" \
        "$model" nccl_agkv 1 "$sequence" native "$chitu_root"
    for cp in 2 4 8; do
        for method in nccl_agkv fast_agkv nccl_ulysses fast_ulysses; do
            run_case \
                "$chitu_root/chitu-${model}-${method}-native-s${sequence}-cp${cp}.json" \
                "$SCRIPT_DIR/run_local_wan_case.sh" \
                "$model" "$method" "$cp" "$sequence" native "$chitu_root"
        done
    done

    xdit_root=$OUTPUT_ROOT/xdit
    for cp in 1 2 4 8; do
        run_case \
            "$xdit_root/xdit-${model}-ulysses-s${sequence}-cp${cp}.json" \
            "$SCRIPT_DIR/run_local_xdit_wan_case.sh" \
            "$model" "$cp" "$sequence" "$xdit_root"
    done

    sglang_root=$OUTPUT_ROOT/sglang
    run_case \
        "$sglang_root/sglang-${model}-ulysses-s${sequence}-t0-cp1.json" \
        "$SCRIPT_DIR/run_local_sglang_joint_case.sh" \
        "$model" ulysses 1 "$sequence" 0 "$sglang_root"
    for cp in 2 4 8; do
        for method in ulysses kv_gather; do
            run_case \
                "$sglang_root/sglang-${model}-${method}-s${sequence}-t0-cp${cp}.json" \
                "$SCRIPT_DIR/run_local_sglang_joint_case.sh" \
                "$model" "$method" "$cp" "$sequence" 0 "$sglang_root"
        done
    done
done
