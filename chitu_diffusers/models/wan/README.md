# Wan2.1 T2V EPAC Integration

Date: 2026-07-27

This package adapts Wan2.1 T2V to the shared `chitu_diffusers` EPAC runtime
while retaining the official Diffusers execution surface. The runtime modules
are `WanPipeline`, `WanTransformer3DModel`, `AutoencoderKLWan`,
`FlowMatchEulerDiscreteScheduler`, `VideoProcessor`, and Transformers
`UMT5EncoderModel`. The local Wan original-format files are mapped into those
official modules during load; the Chitu Wan model implementation is not used.

## Supported Scope

- Wan2.1 T2V 1.3B, batch size one, with the same original loader accepting
  the 14B architecture from its `config.json`;
- original Wan checkpoint directories and converted Diffusers directories;
- request-local 5D latent state and scheduler cursor;
- dynamic AGKV context parallelism over flattened video tokens;
- CFG with CFP2 preferred on even lanes;
- lane widths 1, 2, 4, and 8, including live latent migration;
- startup DiT/transfer/VAE warmup with at least three samples;
- synchronous Diffusers-style and embedded runtime entry points;
- lane-wise tiled official Wan VAE decode and full-video completion output.

The upstream 1.3B model card officially recommends 832x480. It can generate
1280x720, but upstream notes that limited 720p training makes those results less
stable than 480p. The adapter accepts positive multiples of 16 for experiments;
that shape compatibility is not a model-quality guarantee for arbitrary sizes.

The 14B path is not GPU-validated in this change. The initial integration does
not expose I2V, Wan2.2, USP, FlexCache, LoRA, or an HTTP MP4 endpoint.

## Parallel Layout

The 3D patch embedding and full Wan RoPE are computed before token sharding.
Each CP rank keeps a contiguous video-token shard and matching RoPE slice.
Self-attention gathers K/V within the active CP lane while retaining local
queries. Cross-attention uses local video queries against replicated text.
Projected video tokens are gathered before unpatchify and the scheduler step,
so every rank retains a canonical migratable 5D latent.

| Lane width | Layout |
| ---: | --- |
| 1 | serial cond/uncond |
| 2 | CFP2 x CP1 |
| 4 | CFP2 x CP2 |
| 8 | CFP2 x CP4 |

VAE decode splits the latent width across the active lane, includes an
eight-latent-column halo by default, and gathers cropped pixel shards on the
lane leader. `--no-parallel-vae` restores leader-only decode and
`--vae-parallel-halo` changes the overlap.

`FlowMatchEulerDiscreteScheduler` is used because its request state is the
canonical latent plus a scalar step cursor. A multistep UniPC scheduler would
also require transferring its prediction history on every lane migration.

## Validation Snapshot

Local checkpoint: `/home/chenyy/WORK/models/Wan2.1-T2V-1.3B`, Diffusers
0.37.0, BF16 transformer/text encoder and FP32 VAE on one H20 node.

- The upstream Wan repository generated an 832x480, 17-frame, one-step MP4.
- The official Diffusers loop and one-GPU EPAC produced identical MP4 hashes.
- Two-GPU CFP2 and four-GPU CFP2 x CP2 one-step outputs were also byte-identical.
- Four-GPU CFP2 x CP2 remained byte-identical to Diffusers native at four steps;
  decoded frame max and mean absolute error were both zero.
- Three-sample 832x480, 17-frame startup warmup measured median denoise costs
  of 611.1 ms at width 1, 308.2 ms at width 2, and 160.6 ms at width 4.
- A same-latent 832x480, 17-frame comparison measured 2.09x VAE speedup on four
  GPUs, 65.37 dB PSNR, and 0.000485 mean absolute error in the normalized
  `[-1, 1]` output tensor.
- An elastic 8/2-step concurrent trace moved the primary request from `[2, 3]`
  to `[0, 1, 2, 3]` at step 7. It transferred the canonical 1,996,800-byte
  latent to ranks 0 and 1, and its final MP4 was byte-identical to the native
  single-GPU 8-step result.

Generated videos, warmup reports, and timelines are under
`outputs/chitu-diffusers/wan/` and remain untracked experiment evidence.
