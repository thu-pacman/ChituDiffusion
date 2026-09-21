#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PARTITION=${SBATCH_PARTITION:-debug}
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/xdit-model-native-sweep}
DEPENDENCY_ARGS=()
NODE_ARGS=()

if [[ -n "${FAST_CP_BENCH_DEPENDENCY:-}" ]]; then
    DEPENDENCY_ARGS=(--dependency "afterok:${FAST_CP_BENCH_DEPENDENCY}")
fi
if [[ -n "${FAST_CP_BENCH_NODE:-}" ]]; then
    NODE_ARGS=(--nodelist "$FAST_CP_BENCH_NODE")
fi

mkdir -p "$OUTPUT_ROOT"

exec sbatch \
    --parsable \
    --partition "$PARTITION" \
    --exclusive \
    --job-name "${USER:-user}-xdit-models" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task 192 \
    --mem 1937152 \
    --gres gpu:8 \
    --time 01:00:00 \
    --chdir "$PROJECT_ROOT" \
    --output "$OUTPUT_ROOT/slurm-%j.out" \
    "${DEPENDENCY_ARGS[@]}" \
    "${NODE_ARGS[@]}" \
    "$SCRIPT_DIR/run_xdit_model_sweep.sh" "$OUTPUT_ROOT"
