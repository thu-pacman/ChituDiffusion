# Qwen-Image EPAC Integration

Date: 2026-07-26

This package adapts the Diffusers `QwenImagePipeline` to the shared
`chitu_diffusers` EPAC runtime. It keeps model-specific prompt, scheduler,
packed-latent, CFG, transformer, and VAE behavior here; admission, measured
scheduling, lane leases, state transfer, calibration, and worker lifecycle stay
in `chitu_diffusers.epac`.

## Supported Scope

- Qwen-Image text-to-image, batch size one;
- request-local positive/negative Qwen2.5-VL embeddings and scheduler cursor;
- dynamic AGKV context parallelism with replicated text and sharded image tokens;
- true CFG with CFP2 preferred on even lanes;
- lane widths 1, 2, and 4, including live state migration;
- startup DiT/transfer/VAE warmup with at least three samples;
- lane-parallel Qwen 5D VAE decode;
- synchronous Diffusers-style and embedded runtime entry points.

The initial integration does not expose image editing, ControlNet, LoRA,
batching, USP, or FlexCache.

## Parallel Layout

Qwen-Image uses a dual-stream block throughout the transformer. Text tokens are
replicated on every CP rank; packed image tokens and their RoPE frequencies are
split by the active lane. AGKV gathers image K/V while preserving local image
queries and full text queries. The projected image sequence is gathered once
before the scheduler step, so the migratable request state remains canonical.

For true CFG, eligible lanes use CFP2 before image CP:

| Lane width | Layout |
| ---: | --- |
| 1 | serial cond/uncond |
| 2 | CFP2 x CP1 |
| 4 | CFP2 x CP2 |

## Validation Snapshot

Local checkpoint: `/home/chenyy/WORK/models/Qwen-Image`, Diffusers 0.37.0,
BF16 on one H20 node.

- Native Diffusers 512 square, 4-step smoke succeeded.
- EPAC 1/2/4-GPU 4-step outputs differ from native by at most 2/255 per pixel,
  with mean absolute error below 0.134/255.
- Native and four-GPU elastic 50-step outputs differ by at most 3/255,
  mean absolute error 0.137/255, and PSNR 56.73 dB.
- Three-sample 512 warmup measured median denoise costs of 319.5 ms at width 1,
  160.1 ms at width 2, and 103.1 ms at width 4. Warmup first-call samples are
  retained in the report but do not determine these medians.
- A concurrent 50/25-step elastic run assigned two width-2 lanes, then migrated
  the long request from `[2,3]` to `[0,1,2,3]`. Its final PNG hash exactly
  matches the non-migrating four-GPU 50-step run.
- Targeted Qwen/Flux regression tests pass in the designated environment.

Generated images, JSON reports, and per-rank timelines are under
`outputs/chitu-diffusers/qwen_image/` and remain untracked experiment evidence.
