# Chitu API Z-Image Service Smoke Result

Date: 2026-07-18

## Standalone Diffusers Runtime

The package under `experiments/chitu_api/zimage` imports Diffusers and PyTorch
directly. It does not depend on `chitu_diffusion.runtime`, the process-global
`DiffusionBackend`, or the Chitu stage-level pipeline.

H20 validation with checkpoint `/home/chenyy/WORK/models/Z-Image`:

- 1-GPU native pipeline: completed, valid `512 x 512` PNG.
- 2-GPU AGKV CP pipeline: completed, valid `512 x 512` PNG.
- 1-GPU versus 2-GPU output: PSNR `44.28 dB` for the two-step smoke.
- 2-GPU HTTP, one request: planner selected `cp2@[0,1]`, E2E about `690 ms`.
- 2-GPU HTTP, two concurrent requests: planner selected
  `cp1@[0] + cp1@[1]`; both completed in about `1.61 s`.

## Denoise Hot Switch

A six-step, 2-GPU HTTP smoke used `phase_max_steps=1` and lane widths `[1,2]`.
The observed layout sequence was:

```text
hot-first:  step 0-3  cp2@[0,1]
hot-first:  step 4-5  cp1@[0]    (switch)
hot-second: step 0-1  cp1@[1]
hot-second: step 2-5  cp2@[0,1]  (switch)
```

Both requests reached `completed` and returned valid `512 x 512` RGB PNGs
(288,489 and 234,534 bytes). Measured E2E was `15.51 s` for the cold first
request and `1.87 s` for the second request. The first value includes cold
request preparation and is not a throughput benchmark.

This verifies both shrinking and expanding a running request at denoise-step
boundaries. No process group was created during the switches.

## Local Verification

The targeted suite passed with `43 passed`. Ruff, compileall,
dependency-boundary, and whitespace checks also passed. The distributed test
covers `cp2 -> cp1`, canonical latent replication, scheduler cursor alignment,
and rank consensus after each phase.

## Four-GPU Poisson Benchmark

Workload: 12 Poisson arrivals at 3 req/s, `1024 x 1024`, 50 denoise steps,
single-node 4 x H20. Both arms used the standalone HTTP runtime and the same
fixed-seed trace, with a two-step same-shape warmup before measurement.

| metric | static CP | EPE | EPE change |
| --- | ---: | ---: | ---: |
| completed | 12/12 | 12/12 | - |
| makespan | 199.65s | 158.44s | -20.6% |
| throughput | 0.0601 req/s | 0.0757 req/s | +26.0% |
| mean latency | 106.23s | 102.75s | -3.3% |
| p95 latency | 186.15s | 153.87s | -17.3% |
| mean queue delay | 89.59s | 50.71s | -43.4% |
| p95 queue delay | 169.56s | 102.09s | -39.8% |

Static CP serialized requests on `cp4` and completed each 50-step request in a
single phase. EPE used 152 formal phases and 600 request-step lane assignments:
598 `cp1`, two `cp4`, and 21 placement switches. It improved aggregate
throughput and tail latency, but the first request regressed from `16.53s` to
`50.27s` because the burst caused it to shrink from `cp4` to `cp1`.

The arms ran on different nodes of the same H20 partition. No failed request,
OOM, CUDA error, or NCCL error appeared. This is a runtime baseline for the
current count-based planner, not a result from a calibrated cost/SLO policy.

## Historical Shared Fixes

An earlier four-GPU smoke of the now-retired Chitu-runtime adapter found two
general EPE defects. The adapter itself is no longer used by this service, but
the shared fixes remain valid:

- `parallel_tiled_vae_decode()` selected global rank 0 in its fallback. It now
  selects `rank_in_group == 0`, so a width-1 lane at a nonzero offset can decode.
- Decoded results used NCCL `send/recv`, which lazily created pair
  communicators. Retirement now reuses the precreated world CPU group.

## Current Limits

- Running-request cancellation is only supported before first scheduling.
- The CP backend is simplified AGKV; Ulysses and LLaDA2-Uni VQ conditioning are
  not implemented.
- Canonical latent state is replicated to the full stage world after every
  phase; selective migration to only the next lane is not implemented.
- Internally padded sequences are numerically approximate under simplified
  AGKV, so production needs a fixed-prompt quality regression.
- Completion metadata and PNG bytes remain in leader memory without TTL or
  capacity eviction.
