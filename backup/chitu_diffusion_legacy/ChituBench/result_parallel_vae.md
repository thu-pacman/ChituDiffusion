# ChituBench Results: Parallel VAE

VAE decode normally runs on rank 0 only while every other GPU sits idle. As the
denoise stage gets cheaper (parallelism, caching, and especially distilled
few-step models), this single-rank decode becomes a growing share of end-to-end
latency. This module splits VAE decode spatially across all ranks, plus two
supporting optimizations (adaptive model residency, and separating benchmark I/O
from serving latency).

Implementation: `chitu_diffusion/runtime/parallel_vae.py::parallel_tiled_vae_decode`,
wired into `decode_latents` of Qwen-Image, Flux.1-dev, and Flux2-klein-4B. It
treats `vae.decode` as a black box, so it generalizes across VAE families.

Back to [index](result.md).

## parallel_vae_decode

Each rank decodes its contiguous spatial slice plus an interior `halo` of
neighbour latent rows; a VAE decoder is convolutional, so a halo >= the decoder
receptive field makes the core output bit-faithful (no convolution seams). The
outer image border keeps the VAE's natural boundary. Toggles:
`CHITU_VAE_PARALLEL_DECODE` (default `1`, set `0` for rank-0-only) and
`CHITU_VAE_DECODE_HALO` (default `8` latent rows).

### Summary: decode compute speedup (parallel vs rank-0)

| model / config | resolution | decode off -> on (ms) | speedup | PSNR vs rank-0 |
| --- | --- | ---: | ---: | ---: |
| Flux2-klein-4B, 4 GPU | 1024x1024 | 177.4 -> 71.8 | 2.47x | ~32.2 dB |
| Flux2-klein-4B, 8 GPU | 1024x1024 | 176.6 -> 60.3 | 2.93x | ~30.3 dB |
| Flux.1-dev, 4 GPU | 1024x1024 | 177 -> 72 | 2.5x | 40-44 dB |
| Qwen-Image (mc17), 8 GPU | 1328x1328 | 520 -> 225 | 2.3x | 49.9 dB |

### Readout

- Decode compute drops 2.3-2.9x. Speedup is sub-linear because the halo adds
  redundant work, mid-block attention is O(tokens^2) per tile, and there is fixed
  launch/communication overhead at this latent size.
- VAE decode is the single largest stage of a 4-step run, which is exactly why
  parallelizing it matters most for distilled models.
- Quality: Qwen-Image (49.9 dB) and Flux.1 (40-44 dB) are visually identical to
  rank-0 decode. Flux2-klein's `AutoencoderKLFlux2` is more tiling-sensitive
  (GroupNorm gives each tile slightly different statistics): ~30-32 dB with faint
  per-tile brightness bands only visible under ~3x diff amplification, not at
  normal viewing. Fewer tiles help (4 GPU 32.2 dB > 8 GPU 30.3 dB);
  `CHITU_VAE_PARALLEL_DECODE=0` restores exact rank-0 decode.

## parallel_vae_offload_policy

Adaptive model residency for the per-stage `device_scope` (VAE / text encoder).
`device_scope` previously always moved the model back to its origin device and
called `torch.cuda.empty_cache()` on exit. At `low_mem_level=0` the model is
already GPU-resident, so the move-back is a no-op but the `empty_cache()` still
runs every stage and churns the allocator. The new policy keeps ample-VRAM models
resident (skipping `empty_cache`), while preserving CPU offload under memory
pressure.

Implementation: `chitu_diffusion/runtime/adapter/base.py::device_scope` +
`DiffusionBackend.should_keep_resident` / `gpu_mem_info` (true free VRAM via
`torch.cuda.mem_get_info`). Config `infer.diffusion.offload_policy`
(env `CHITU_OFFLOAD_POLICY`): `auto` (default) / `always_offload` (legacy) /
`never_offload`.

### Summary

Flux2-klein-4B, 4 steps, 1024x1024, Ulysses CP8, parallel VAE decode on:

| offload_policy | warm VaeDecode region (ms) |
| --- | ---: |
| always_offload (legacy) | 57.9 |
| auto (resident, no empty_cache) | 43.4 |

### Readout

- Skipping the per-stage `empty_cache()` cuts the warm decode region ~25%
  (57.9 -> 43.4 ms) with bit-identical output (verified maxdiff=0).
- The larger win is on memory-constrained configs (`low_mem_level>=2`) where the
  text encoder / VAE are genuinely on CPU: `auto` promotes them to resident when
  free VRAM allows, removing real per-stage CPU<->GPU transfers, and falls back
  to offload when memory is tight.

## serving_vs_benchmark_latency

The ChituBench `overall` wall clock ran the whole generation loop, but the loop
also did benchmark-harness work per task: writing the PNG artifact + sidecar
metadata JSON, dumping per-task timing/memory JSON, and printing the timing
table. None of that is inference latency a real server would pay (a server
returns encoded bytes, it does not persist files to a results dir). This was
inflating both the `VAEDecode` stage time and the end-to-end number.

Change: the VAEDecode stage timer now closes right after decode compute. Artifact
saving and metrics dumps run after, wrapped in a `benchmark_overhead` timer; the
redundant per-task `Timer.print_statistics()` was removed. `dump_timing_summary`
now reports `serving_elapsed_s = overall_elapsed_s - benchmark_overhead_s`.
Implementation: `chitu_diffusion/runtime/generator.py`,
`test/run_context.py::dump_timing_summary`, `ChituBench/scripts/collect_floor.py`.

### Summary

Flux2-klein-4B, 4 steps, 1024x1024, Ulysses CP8, parallel VAE decode on, `auto`:

| metric | before split | after split |
| --- | ---: | ---: |
| VAEDecode stage warm (ms) | ~460 | 43 |
| benchmark_overhead /task (ms) | (inside stage) | 434 |
| metrics-only dumps /task (ms) | (inside stage) | 0.65 |
| overall_elapsed_s | 9.94 | 9.94 |
| serving_elapsed_s | (= overall) | 8.21 |

### Readout

- The VAEDecode stage now matches the decode compute timer (~43 ms warm); the
  ~400 ms it used to show was entirely `save_output` artifact I/O.
- Benchmark metrics dumps are negligible (~0.65 ms/task); the harness cost is the
  PNG encode + disk write (~434 ms/task).
- For this 4-step model the harness was adding ~1.74 s (~18%) to the reported
  wall clock. `serving_elapsed_s` (8.21 s) is the number to track for inference
  latency; outputs are unchanged (artifacts still written, after the stage timer).

## End-to-end VAE-decode contribution by GPU count

Flux2-klein-4B warm per-image stage breakdown (parallel VAE on, `auto`, benchmark
I/O excluded). Shows how VAE decode shrinks and stays the largest non-DiT stage:

| GPUs | TextEncode (ms) | DiT/Denoise (ms) | VAE decode (ms) | serving e2e (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 49 | 1361 | 171 | 1582 |
| 4 | 49 | 409 | 63 | 522 |
| 8 | 50 | 275 | 44 | 368 |

- 1 GPU has no parallel decode (falls back to rank-0): 171 ms.
- At 8 GPU the DiT is so small that VAE decode (44 ms) + TextEncode (50 ms)
  together are ~26% of serving latency, so both non-DiT stages now matter.
