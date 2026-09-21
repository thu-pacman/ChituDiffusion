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
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/sglang-joint-native-sweep}
SEQUENCE=${FAST_CP_MODEL_SEQUENCE:-4096}
TEXT_SEQUENCE=${FAST_CP_MODEL_TEXT_SEQUENCE:-512}
CP_DEGREES=(${FAST_CP_BENCH_CP_DEGREES:-2 4 8})

mkdir -p "$OUTPUT_ROOT/smoke"

report_is_terminal() {
    local report=$1
    [[ -s "$report" ]] || return 1
    "$PROJECT_ROOT/.venv/bin/python" -c \
        'import json, sys; sys.exit(json.load(open(sys.argv[1])).get("status") not in {"complete", "unsupported"})' \
        "$report" >/dev/null 2>&1
}

FAST_CP_BENCH_WARMUP=2 \
FAST_CP_BENCH_ITERATIONS=3 \
FAST_CP_BENCH_REPEATS=2 \
    "$SCRIPT_DIR/run_local_sglang_joint_case.sh" \
    qwenimage ulysses 2 512 64 "$OUTPUT_ROOT/smoke"
FAST_CP_BENCH_WARMUP=2 \
FAST_CP_BENCH_ITERATIONS=3 \
FAST_CP_BENCH_REPEATS=2 \
    "$SCRIPT_DIR/run_local_sglang_joint_case.sh" \
    zimage kv_gather 2 512 64 "$OUTPUT_ROOT/smoke"
FAST_CP_BENCH_WARMUP=2 \
FAST_CP_BENCH_ITERATIONS=3 \
FAST_CP_BENCH_REPEATS=2 \
    "$SCRIPT_DIR/run_local_sglang_joint_case.sh" \
    wan14b ulysses 2 512 0 "$OUTPUT_ROOT/smoke"

for model in wan1.3b wan14b qwenimage zimage; do
    baseline="$OUTPUT_ROOT/sglang-${model}-ulysses-s${SEQUENCE}-t${TEXT_SEQUENCE}-cp1.json"
    if ! report_is_terminal "$baseline"; then
        "$SCRIPT_DIR/run_local_sglang_joint_case.sh" \
            "$model" ulysses 1 "$SEQUENCE" "$TEXT_SEQUENCE" "$OUTPUT_ROOT"
    fi

    for cp in "${CP_DEGREES[@]}"; do
        for method in ulysses kv_gather; do
            report="$OUTPUT_ROOT/sglang-${model}-${method}-s${SEQUENCE}-t${TEXT_SEQUENCE}-cp${cp}.json"
            if report_is_terminal "$report"; then
                echo "Skipping terminal report: $report"
                continue
            fi
            "$SCRIPT_DIR/run_local_sglang_joint_case.sh" \
                "$model" "$method" "$cp" "$SEQUENCE" "$TEXT_SEQUENCE" "$OUTPUT_ROOT"
        done
    done
done
