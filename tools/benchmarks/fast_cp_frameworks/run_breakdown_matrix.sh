#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
SEQUENCE=${1:-4096}
HEADS=${2:-40}
HEAD_DIM=${3:-128}
CP=${4:-8}
OUTPUT_ROOT=${5:-$PROJECT_ROOT/outputs/fast-cp/breakdown-same-node}

CASES=(
    "chitu fast_agkv"
    "chitu fast_ulysses"
    "chitu nccl_agkv"
    "chitu nccl_ulysses"
    "xdit ulysses"
    "sglang kv_gather"
    "sglang ulysses"
)
COMPONENTS=(e2e communication attention)

for case_spec in "${CASES[@]}"; do
    read -r framework method <<<"$case_spec"
    for component in "${COMPONENTS[@]}"; do
        "$SCRIPT_DIR/run_local_case.sh" \
            "$framework" "$method" "$CP" "$SEQUENCE" \
            "$HEADS" "$HEAD_DIM" "$component" "$OUTPUT_ROOT"
    done
done
