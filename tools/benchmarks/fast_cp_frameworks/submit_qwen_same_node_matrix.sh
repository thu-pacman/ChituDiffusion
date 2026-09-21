#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/qwen-same-node-20260918}

mkdir -p "$OUTPUT_ROOT"
exec sbatch \
    --partition debug \
    --nodes 1 \
    --gres gpu:8 \
    --exclusive \
    --job-name "${USER:-user}-qwen-cp" \
    --output "$OUTPUT_ROOT/slurm-%j.out" \
    --wrap "cd '$PROJECT_ROOT' && '$SCRIPT_DIR/run_qwen_same_node_matrix.sh' '$OUTPUT_ROOT'"
