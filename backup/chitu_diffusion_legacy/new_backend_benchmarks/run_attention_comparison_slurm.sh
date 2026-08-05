#!/bin/bash
#SBATCH --job-name=epac-attention-compare
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/epac-attention/slurm-%j.out

set -euo pipefail

ROOT_DIR=${EPAC_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
ROOT_DIR=$(cd "$ROOT_DIR" && pwd)
VENV=${EPAC_VENV:-$ROOT_DIR/../ChituDiffusion/.venv}
TRACE=${EPAC_TRACE:-$ROOT_DIR/chitu_diffusers/benchmarks/traces/mixed_512_1024_2048_r3n12_12step.json}
OUTPUT_BASE=${EPAC_COMPARISON_DIR:-$ROOT_DIR/outputs/epac-attention/job-${SLURM_JOB_ID}}
BASE_PORT=${EPAC_PORT:-$((26000 + SLURM_JOB_ID % 1000))}
TIMELINE_RATE=${EPAC_TIMELINE_RATE:-0.12}
read -r -a RATES <<<"${EPAC_ARRIVAL_RATES:-0.06 0.12 0.24}"
read -r -a VARIANTS <<<"${EPAC_VARIANTS:-agkv static_dp static_cp usp}"

mkdir -p "$OUTPUT_BASE"
PLOT_ARGS=()
run_index=0
for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    agkv)
      mode=agkv
      strategy=elastic
      label="Elastic AGKV"
      ;;
    static_dp)
      mode=agkv
      strategy=static_dp
      label="Static DP AGKV"
      ;;
    static_cp)
      mode=agkv
      strategy=static_cp
      label="Static CP AGKV"
      ;;
    usp)
      mode=usp
      strategy=elastic
      label="Elastic USP u2r2"
      ;;
    *)
      echo "Unknown EPAC variant: $variant" >&2
      exit 2
      ;;
  esac
  for rate in "${RATES[@]}"; do
    rate_key=${rate//./p}
    run_dir="$OUTPUT_BASE/$variant/rate-$rate_key"
    EPAC_ROOT="$ROOT_DIR" \
    EPAC_VENV="$VENV" \
    EPAC_TRACE="$TRACE" \
    EPAC_PORT=$((BASE_PORT + run_index)) \
    EPAC_RUN_DIR="$run_dir" \
    EPAC_SCHEDULE_STRATEGY="$strategy" \
    EPAC_ATTENTION_MODE="$mode" \
    EPAC_ULYSSES_DEGREE=2 \
    EPAC_ARRIVAL_RATE="$rate" \
      bash "$ROOT_DIR/chitu_diffusers/benchmarks/run_benchmark_slurm.sh"
    PLOT_ARGS+=(--run "$label:$rate:$run_dir/metrics.json")
    run_index=$((run_index + 1))
  done
done

"$VENV/bin/python" -m chitu_diffusers.benchmarks.plot_attention_rates \
  "${PLOT_ARGS[@]}" \
  --output "$OUTPUT_BASE/attention_rate.png"

if [[ -d "$OUTPUT_BASE/agkv" ]]; then
  timeline_key=${TIMELINE_RATE//./p}
  "$VENV/bin/python" -m chitu_diffusers.benchmarks.plot_strategy_timeline \
    --trace "$TRACE" \
    --arrival-rate "$TIMELINE_RATE" \
    --run "Elastic AGKV=$OUTPUT_BASE/agkv/rate-$timeline_key" \
    --output "$OUTPUT_BASE/attention_timeline.png"
fi

echo "EPAC_ATTENTION_COMPARISON_DIR=$OUTPUT_BASE"
