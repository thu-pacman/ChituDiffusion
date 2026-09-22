#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PARTITION_ARGS=()
if [[ -n "${SBATCH_PARTITION:-}" ]]; then
    PARTITION_ARGS=(--partition "$SBATCH_PARTITION")
fi
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/e2e-native-sweep}
NODE_ARGS=()
DEPENDENCY_ARGS=()

if [[ -n "${FAST_CP_BENCH_NODE:-}" ]]; then
    NODE_ARGS=(--nodelist "$FAST_CP_BENCH_NODE")
fi

if [[ -n "${FAST_CP_BENCH_DEPENDENCY:-}" ]]; then
    DEPENDENCY_ARGS=(--dependency "afterok:$FAST_CP_BENCH_DEPENDENCY")
fi

mkdir -p "$OUTPUT_ROOT"

exec sbatch \
    --parsable \
    "${PARTITION_ARGS[@]}" \
    --exclusive \
    --job-name "${FAST_CP_BENCH_JOB_NAME:-${USER:-user}-fast-cp-e2e}" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task 192 \
    --mem 1937152 \
    --gres gpu:8 \
    --time 01:00:00 \
    --chdir "$PROJECT_ROOT" \
    --output "$OUTPUT_ROOT/slurm-%j.out" \
    "${NODE_ARGS[@]}" \
    "${DEPENDENCY_ARGS[@]}" \
    "$SCRIPT_DIR/run_e2e_sweep.sh" "$OUTPUT_ROOT"
