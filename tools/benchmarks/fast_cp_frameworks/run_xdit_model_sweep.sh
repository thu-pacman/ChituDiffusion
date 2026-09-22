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
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/xdit-model-native-sweep}
SEQUENCE=${FAST_CP_MODEL_SEQUENCE:-4096}
TEXT_SEQUENCE=${FAST_CP_MODEL_TEXT_SEQUENCE:-512}
CP_DEGREES=(${FAST_CP_BENCH_CP_DEGREES:-1 2 4 8})

mkdir -p "$OUTPUT_ROOT"

report_is_terminal() {
    local report=$1
    [[ -s "$report" ]] || return 1
    "$PROJECT_ROOT/.venv/bin/python" -c \
        'import json, sys; sys.exit(json.load(open(sys.argv[1])).get("status") not in {"complete", "unsupported"})' \
        "$report" >/dev/null 2>&1
}

for model in wan1.3b wan14b; do
    for cp in "${CP_DEGREES[@]}"; do
        report="$OUTPUT_ROOT/xdit-${model}-ulysses-s${SEQUENCE}-cp${cp}.json"
        if report_is_terminal "$report"; then
            echo "Skipping terminal report: $report"
            continue
        fi
        "$SCRIPT_DIR/run_local_xdit_wan_case.sh" \
            "$model" "$cp" "$SEQUENCE" "$OUTPUT_ROOT"
    done
done

for model in qwenimage zimage; do
    for cp in "${CP_DEGREES[@]}"; do
        report="$OUTPUT_ROOT/xdit-${model}-ulysses-s${SEQUENCE}-t${TEXT_SEQUENCE}-cp${cp}.json"
        if report_is_terminal "$report"; then
            echo "Skipping terminal report: $report"
            continue
        fi
        "$SCRIPT_DIR/run_local_xdit_joint_case.sh" \
            "$model" "$cp" "$SEQUENCE" "$TEXT_SEQUENCE" "$OUTPUT_ROOT"
    done
done
