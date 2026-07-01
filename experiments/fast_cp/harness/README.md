# CP-core async overlap harness

A standalone driver for the model-agnostic CP attention core
([`image_cp_attention.py`](../../../chitu_diffusion/modules/attention/image_cp_attention.py)),
feeding it **synthetic** packed Q/K/V instead of a real Z-Image forward pass.

Purpose: debug, bit-equivalence-check, profile, and A/B the async QKV-comm
overlap (`attention_fused`) **in isolation**, then migrate the proven path into
`z_image_attention.py`. See section 1.5 of
[`../async_comm_h20_nvlink_plan.md`](../async_comm_h20_nvlink_plan.md) for the
rationale and the migration gate.

## Why a harness

The CP core depends only on:

- a comm `group` (`group_size` / `gpu_group` / `rank_in_group` / `all_to_all`),
  which the harness wraps around a raw `torch.distributed` process group
  (`HarnessGroup`); and
- an `attn_backend(q, k, v, causal=False) -> (out, ...)` callable, which the
  harness fills with plain SDPA (`sdpa_backend`).

No model weights, scheduler, config, or 50-step denoise loop. One attention call
per iteration -> seconds per A/B, and a clean numerical signal (any stream /
`record_stream` / event bug shows up as a serial-vs-overlap mismatch with no
model noise on top).

## Run

```bash
# correctness: serial (cp.attention) vs overlap (cp.attention_fused), same seed
bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv --check
bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode ulysses --check

# wall-clock A/B (median over --iters)
bash experiments/fast_cp/harness/run_harness.sh 8 -- --mode agkv --bench
bash experiments/fast_cp/harness/run_harness.sh 8 -- --mode ulysses --bench

# chrome trace of the overlap path (verify NCCL on a side stream vs proj GEMM)
CHITU_TORCH_TRACE=$PWD/outputs/h20_async_comm_traces/agkv_overlap.json \
  bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv

# chrome trace of the captured CUDA graph replay path (AGKV only)
CHITU_TORCH_TRACE=$PWD/outputs/h20_async_comm_traces/agkv_overlap_graph.json \
CHITU_TRACE_MODE=overlap_graph \
  bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv

# optional: emit serial-graph and overlap-graph traces side by side
CHITU_TORCH_TRACE=$PWD/outputs/h20_async_comm_traces/agkv_graph_compare.json \
CHITU_TRACE_MODE=both_graph \
  bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv
```

On Slurm/H20, run the same script under an `srun` allocation (one task per node,
export `MASTER_ADDR`/`MASTER_PORT`/`NODE_RANK` as in
[`script/srun_direct.sh`](../../../script/srun_direct.sh)). Keep the harness and
the real Z-Image A/B on the **same partition** so the overlap ceiling is
topology-faithful.

## Key flags

| flag | meaning |
| --- | --- |
| `--mode {agkv,ulysses}` | CP strategy under test (phase-1 overlap targets) |
| `--check` | assert overlap output == serial output (max-abs-diff / PSNR) |
| `--bench` | report serial vs overlap median ms/call, and the delta |
| `--img-tokens` | **global** image tokens, sharded across the CP group |
| `--txt-tokens` | replicated text tokens |
| `--heads` / `--head-dim` | attention shape (Ulysses needs `heads % cp == 0`) |
| `--iters` / `--warmup` | benchmark sample / warmup counts |
| `--dtype {bf16,fp16,fp32}` | compute dtype (phase-1 overlap is bf16) |

Defaults approximate Z-Image 1024x1024 (`img-tokens=4096`, `txt-tokens=512`,
`heads=24`, `head-dim=128`); override to match the exact config under test.

## Lifecycle vs the plan

- **Step 1-2 (scaffolding):** `attention_fused` does not exist yet, or is a
  fallback-only no-op. `--check` reports "skipped / fused unavailable" and
  `--bench` times the serial path only. Useful as a baseline immediately.
- **Step 3 (AGKV) / Step 4 (Ulysses):** `--check` must pass (PSNR > 60 dB,
  effectively bit-identical) and the trace must show overlap before porting.
- **Step 8 (CUDA graph, later):** when a `CHITU_CP_OVERLAP_GRAPH` capture path is
  added, validate it here (bit-equiv + trace) before touching Z-Image. Current
  harness graph support is AGKV-only via `--graph` for correctness/bench and
  `CHITU_TRACE_MODE=overlap_graph` / `both_graph` for traces.

## Notes

- The harness uses the **full world** as a single CP group. For multi-CP-group
  topologies (e.g. cfp2 x cp4) launch with the matching `--nproc-per-node` and
  treat each run as one CP group's worth of work.
- `--check` fixes the per-rank seed so serial and overlap consume identical
  synthetic Q/K/V; the only difference is the schedule.
- Benchmark numbers are core-level (one attention call), not end-to-end denoise.
  Use them to confirm the overlap direction/ceiling, not as the final speedup.
