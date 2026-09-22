#!/usr/bin/env bash
# Run inside a single-node allocation with eight H20 NVLink GPUs.
set -euo pipefail

test_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # sbatch copies this script to its spool directory.
    project_root=${CHITU_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from the project root}}
else
    project_root=${CHITU_PROJECT_ROOT:-$(cd "$test_dir/../../../../.." && pwd)}
fi
test_dir=$project_root/chitu_diffusion/parallel/cp/fast/test
python_bin=${CHITU_PYTHON_BIN:-$project_root/.venv/bin/python}
source "$project_root/tools/install/fast_cp_env.sh"
output_dir=${1:-$project_root/outputs/fast-cp/wrapper}

cd "$project_root"
export OMP_NUM_THREADS=1 NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_GRAPH_REGISTER=0
export TORCH_NCCL_ENABLE_MONITORING=0

"$python_bin" -m pytest "$test_dir" -q
for cp in 2 4 8; do
    "$python_bin" -m torch.distributed.run --standalone --nproc_per_node "$cp" \
        "$test_dir/distributed_wrapper.py" \
        --models wan1.3b wan14b qwen zimage \
        --output "$output_dir/cp${cp}.json"
done
"$python_bin" "$test_dir/summarize.py" "$output_dir"/cp{2,4,8}.json \
    --summary "$output_dir/summary.csv" --samples "$output_dir/samples.csv"
