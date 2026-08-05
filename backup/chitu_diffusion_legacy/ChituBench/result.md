# ChituBench Results

Results are split by optimization module so each one shows, on its own, the
improvement ChituDiffusion brings. The README remains the worklog; these pages
are the compact result view. (`plots/` paths are relative to this folder.)

## Modules

| module | what it does | headline result |
| --- | --- | --- |
| [Attention Backend](result_attention.md) | pluggable attention kernels (Flash / Sage / Sparge / FlashInfer) vs origin | ~1.16x quality-preserving (Flux), up to 2.23x on Wan video |
| [Parallel DiT](result_parallel_dit.md) | CFG + context/sequence parallel on the denoise stage | up to 12.81x (Wan2.1-T2V-1.3B, 16 GPU CFP2+Ring2+UP4); Qwen 5.40x |
| [FlexCache](result_flexcache.md) | step/block-level caching (MeanCache, TeaCache, TaylorSeer, Cubic, PAB, BlockDance) | MeanCache 2.94x@26 dB (Flux.1) / 3.62x@24.5 dB (Qwen); Wan MeanCache30 1.66x@35.6 dB, Cubic reaches 2.20x@26.2 dB |
| [Parallel VAE](result_parallel_vae.md) | tile-parallel VAE decode + adaptive offload + serving/benchmark split | decode 2.3-2.9x; serving latency cleaned of harness I/O |

## How to read each module

- **Attention Backend** and **FlexCache** report DiT-forward speedup vs a
  per-model baseline together with quality drift (PSNR / SSIM / 1-LPIPS / HPSv3).
  They are single-GPU studies that trade quality for speed.
- **Parallel DiT** and **Parallel VAE** report multi-GPU scaling and stage
  latency (speed only). They preserve output and trade GPUs for latency.

## Models covered

Flux.1-dev, Flux2-klein-4B (4-step distilled), Qwen-Image, and Wan2.1-T2V-1.3B
(video attention, Parallel DiT, and FlexCache).
