#!/bin/bash
#SBATCH --job-name=epac-zimage-bench
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/epac-benchmark/slurm-%j.out

set -euo pipefail

ROOT_DIR=${EPAC_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
ROOT_DIR=$(cd "$ROOT_DIR" && pwd)
VENV=${EPAC_VENV:-$ROOT_DIR/../ChituDiffusion/.venv}
MODEL_PATH=${ZIMAGE_MODEL_PATH:?Set ZIMAGE_MODEL_PATH to the Z-Image checkpoint}
TRACE=${EPAC_TRACE:-$ROOT_DIR/chitu_diffusers/benchmarks/traces/mixed_512_1024_2048_r3n12_12step.json}
PORT=${EPAC_PORT:-$((19000 + SLURM_JOB_ID % 1000))}
GPUS_PER_NODE=${EPAC_GPUS_PER_NODE:-4}
RUN_DIR=${EPAC_RUN_DIR:-$ROOT_DIR/outputs/epac-benchmark/job-${SLURM_JOB_ID}}
SCHEDULE_STRATEGY=${EPAC_SCHEDULE_STRATEGY:-elastic}
ATTENTION_MODE=${EPAC_ATTENTION_MODE:-agkv}
ULYSSES_DEGREE=${EPAC_ULYSSES_DEGREE:-2}
ARRIVAL_RATE=${EPAC_ARRIVAL_RATE:-}
DEFAULT_DEADLINE_MS=${EPAC_DEFAULT_DEADLINE_MS:-}
ENDPOINT="http://127.0.0.1:$PORT"
SERVICE_LOG="$RUN_DIR/service.log"
METRICS="$RUN_DIR/metrics.json"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-0}
export NCCL_GRAPH_REGISTER=${NCCL_GRAPH_REGISTER:-0}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
mkdir -p "$RUN_DIR"

SERVICE_ARGS=()
if [[ -n "$DEFAULT_DEADLINE_MS" ]]; then
  SERVICE_ARGS+=(--default-deadline-ms "$DEFAULT_DEADLINE_MS")
fi

"$VENV/bin/torchrun" \
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
  --schedule-strategy "$SCHEDULE_STRATEGY" \
  --default-steps 50 \
  --max-inflight-requests 4 \
  --max-pending-requests 64 \
  --record-timeline \
  --output-root "$RUN_DIR" \
  "${SERVICE_ARGS[@]}" \
  >"$SERVICE_LOG" 2>&1 &
SERVICE_PID=$!

shutdown_service() {
  rank0_pid_file="$RUN_DIR/worker-rank0.pid"
  if [[ -f "$rank0_pid_file" ]]; then
    rank0_pid=$(cat "$rank0_pid_file")
    kill -TERM "$rank0_pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "$SERVICE_PID" 2>/dev/null; then
    kill -TERM "$SERVICE_PID" 2>/dev/null || true
  fi
  wait "$SERVICE_PID" 2>/dev/null || true
}
trap shutdown_service EXIT INT TERM

ready=0
for _ in $(seq 1 1800); do
  if curl --silent --fail --max-time 2 "$ENDPOINT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
    echo "EPAC service exited before becoming ready" >&2
    tail -200 "$SERVICE_LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "$ready" != "1" ]]; then
  echo "EPAC service did not become ready" >&2
  tail -200 "$SERVICE_LOG" >&2 || true
  exit 1
fi

CLIENT_ARGS=(
  --endpoint "$ENDPOINT"
  --trace "$TRACE"
  --output "$METRICS"
  --save-images-dir "$RUN_DIR/images"
  --warmup-steps 0
  --ready-timeout-s 30
  --timeout-s 1800
  --poll-interval-s 0.25
)
if [[ -n "$ARRIVAL_RATE" ]]; then
  CLIENT_ARGS+=(--arrival-rate "$ARRIVAL_RATE")
fi
"$VENV/bin/python" -m chitu_diffusers.benchmarks.benchmark_client "${CLIENT_ARGS[@]}"

echo "EPAC_BENCHMARK_RUN_DIR=$RUN_DIR"
echo "EPAC_BENCHMARK_METRICS=$METRICS"
