#!/bin/bash
#SBATCH --job-name=epac-phased-compare
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/epac-phased/slurm-%j.out

set -euo pipefail

ROOT_DIR=${EPAC_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
ROOT_DIR=$(cd "$ROOT_DIR" && pwd)
VENV=${EPAC_VENV:-$ROOT_DIR/../ChituDiffusion/.venv}
TRACE=${EPAC_TRACE:-$ROOT_DIR/chitu_diffusers/benchmarks/traces/phased_dp_cp_slo_n15_12step.json}
OUTPUT_BASE=${EPAC_COMPARISON_DIR:-$ROOT_DIR/outputs/epac-phased/job-${SLURM_JOB_ID}}
BASE_PORT=${EPAC_PORT:-$((24000 + SLURM_JOB_ID % 1000))}

mkdir -p "$OUTPUT_BASE"

strategies=(static_dp static_cp elastic)
for index in "${!strategies[@]}"; do
  strategy=${strategies[$index]}
  EPAC_ROOT="$ROOT_DIR" \
  EPAC_VENV="$VENV" \
  EPAC_TRACE="$TRACE" \
  EPAC_PORT=$((BASE_PORT + index)) \
  EPAC_RUN_DIR="$OUTPUT_BASE/$strategy" \
  EPAC_SCHEDULE_STRATEGY="$strategy" \
    bash "$ROOT_DIR/chitu_diffusers/benchmarks/run_benchmark_slurm.sh"
done

"$VENV/bin/python" -m chitu_diffusers.benchmarks.plot_strategy_timeline \
  --trace "$TRACE" \
  --static-dp-log "$OUTPUT_BASE/static_dp/service.log" \
  --static-dp-metrics "$OUTPUT_BASE/static_dp/metrics.json" \
  --static-cp-log "$OUTPUT_BASE/static_cp/service.log" \
  --static-cp-metrics "$OUTPUT_BASE/static_cp/metrics.json" \
  --elastic-log "$OUTPUT_BASE/elastic/service.log" \
  --elastic-metrics "$OUTPUT_BASE/elastic/metrics.json" \
  --output "$OUTPUT_BASE/strategy_timeline.png"

echo "EPAC_COMPARISON_DIR=$OUTPUT_BASE"
echo "EPAC_COMPARISON_TIMELINE=$OUTPUT_BASE/strategy_timeline.png"
