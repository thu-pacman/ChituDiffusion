#!/bin/bash
# Launch the standalone CP-core async overlap harness under torchrun.
#
# Usage:
#   bash experiments/fast_cp/harness/run_harness.sh <num_gpus> [-- <harness args...>]
#
# Examples:
#   bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv --check
#   bash experiments/fast_cp/harness/run_harness.sh 8 -- --mode ulysses --bench
#   CHITU_TORCH_TRACE=$PWD/outputs/h20_async_comm_traces/agkv_overlap.json \
#       bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv
#
# On Slurm, wrap with srun (one task per node) and set MASTER_ADDR/NODE_RANK as
# in script/srun_direct.sh; here we default to a single-node standalone launch.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <num_gpus> [-- <harness args...>]" >&2
    exit 1
fi

NUM_GPUS=$1
shift
# allow an optional "--" separator before harness args
if [[ "${1:-}" == "--" ]]; then
    shift
fi
HARNESS_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
HARNESS="$SCRIPT_DIR/cp_async_harness.py"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$(expr $RANDOM % 10000 + 50000)}

# Keep NCCL defaults aligned with the project launchers.
export NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-0}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}

RUNTIME_PYTHON="${CHITU_PYTHON_BIN:-python}"

set -x
exec "$RUNTIME_PYTHON" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$NUM_GPUS" \
    "$HARNESS" "${HARNESS_ARGS[@]}"
