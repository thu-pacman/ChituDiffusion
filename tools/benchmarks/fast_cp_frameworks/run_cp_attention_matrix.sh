#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${FAST_CP_BENCH_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT=$FAST_CP_BENCH_PROJECT_ROOT
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/tools/benchmarks/fast_cp_frameworks" ]]; then
    PROJECT_ROOT=$SLURM_SUBMIT_DIR
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
fi
SCRIPT_DIR=$PROJECT_ROOT/tools/benchmarks/fast_cp_frameworks
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/cp-attention-native-20260918}
HEADS=${FAST_CP_BENCH_HEADS:-40}
HEAD_DIM=${FAST_CP_BENCH_HEAD_DIM:-128}
SEQUENCES=(${FAST_CP_BENCH_SEQUENCES:-4096 8192 16384 32768})
CP_DEGREES=(${FAST_CP_BENCH_CP_DEGREES:-2 4 8})
FORCE=${FAST_CP_BENCH_FORCE:-0}

CP1_CASES=(
    "chitu nccl_agkv"
    "xdit ulysses"
    "sglang ulysses"
)
CP_CASES=(
    "chitu fast_agkv"
    "chitu fast_ulysses"
    "xdit ulysses"
    "xdit usp"
    "sglang kv_gather"
    "sglang ulysses"
)

mkdir -p "$OUTPUT_ROOT"

report_is_complete() {
    local report=$1
    [[ -s "$report" ]] || return 1
    "$PROJECT_ROOT/.venv/bin/python" -c \
        'import json, sys; sys.exit(json.load(open(sys.argv[1])).get("status") != "complete")' \
        "$report" >/dev/null 2>&1
}

run_case() {
    local framework=$1 method=$2 cp=$3 sequence=$4
    local report="$OUTPUT_ROOT/${framework}-${method}-h${HEADS}-s${sequence}-cp${cp}.json"
    if [[ "$FORCE" != 1 ]] && report_is_complete "$report"; then
        echo "Skipping complete report: $report"
        return
    fi
    "$SCRIPT_DIR/run_local_case.sh" \
        "$framework" "$method" "$cp" "$sequence" \
        "$HEADS" "$HEAD_DIM" e2e "$OUTPUT_ROOT"
}

for sequence in "${SEQUENCES[@]}"; do
    for case_spec in "${CP1_CASES[@]}"; do
        read -r framework method <<<"$case_spec"
        run_case "$framework" "$method" 1 "$sequence"
    done
    for cp in "${CP_DEGREES[@]}"; do
        for case_spec in "${CP_CASES[@]}"; do
            read -r framework method <<<"$case_spec"
            run_case "$framework" "$method" "$cp" "$sequence"
        done
    done
done

"$PROJECT_ROOT/.venv/bin/python" "$SCRIPT_DIR/summarize_cp_attention.py" \
    "$OUTPUT_ROOT" --output-md "$OUTPUT_ROOT/RESULTS.md" \
    --output-csv "$OUTPUT_ROOT/results.csv"
