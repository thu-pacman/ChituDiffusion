#!/bin/bash
#SBATCH --job-name=epac-zimage-serve
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/epac-serve/slurm-%j.out

set -euo pipefail

ROOT_DIR=${EPAC_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
ROOT_DIR=$(cd "$ROOT_DIR" && pwd)
VENV=${EPAC_VENV:-$ROOT_DIR/../ChituDiffusion/.venv}
MODEL_PATH=${ZIMAGE_MODEL_PATH:?Set ZIMAGE_MODEL_PATH to the Z-Image checkpoint}
PORT=${EPAC_PORT:-18200}
GPUS_PER_NODE=${EPAC_GPUS_PER_NODE:-4}
OUTPUT_ROOT=${EPAC_OUTPUT_ROOT:-$ROOT_DIR/outputs/epac-serve/job-${SLURM_JOB_ID}}
SCHEDULE_STRATEGY=${EPAC_SCHEDULE_STRATEGY:-elastic}
ATTENTION_MODE=${EPAC_ATTENTION_MODE:-agkv}
ULYSSES_DEGREE=${EPAC_ULYSSES_DEGREE:-2}
CFG_PARALLEL=${EPAC_CFG_PARALLEL:-1}
PARALLEL_VAE=${EPAC_PARALLEL_VAE:-1}
VAE_PARALLEL_HALO=${EPAC_VAE_PARALLEL_HALO:-8}

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-0}
export NCCL_GRAPH_REGISTER=${NCCL_GRAPH_REGISTER:-0}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
mkdir -p "$OUTPUT_ROOT"

SERVICE_ARGS=()
if [[ "$CFG_PARALLEL" == "0" ]]; then
  SERVICE_ARGS+=(--no-cfg-parallel)
fi
if [[ "$PARALLEL_VAE" == "0" ]]; then
  SERVICE_ARGS+=(--no-parallel-vae)
fi

exec "$VENV/bin/torchrun" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$GPUS_PER_NODE" \
  --module chitu_diffusers.examples.zimage_serve \
  --model-path "$MODEL_PATH" \
  --local-files-only \
  --host 0.0.0.0 \
  --advertise-host "$(hostname)" \
  --port "$PORT" \
  --warmup-resolutions 512 1024 2048 \
  --warmup-steps 5 \
  --pulse-steps 5 \
  --attention-mode "$ATTENTION_MODE" \
  --ulysses-degree "$ULYSSES_DEGREE" \
  --vae-parallel-halo "$VAE_PARALLEL_HALO" \
  --schedule-strategy "$SCHEDULE_STRATEGY" \
  --default-steps 50 \
  --max-inflight-requests 4 \
  --max-pending-requests 64 \
  --output-root "$OUTPUT_ROOT" \
  "${SERVICE_ARGS[@]}"
