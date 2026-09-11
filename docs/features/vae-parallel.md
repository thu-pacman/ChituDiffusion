# Static VAE Parallelism

All eight supported model integrations use `parallel_vae_decode` for image/video
decoding. They share the spatial partition and communication in `exact.py`;
`families.py` describes release-specific layouts. Dynamic EPE, training, and
distributed VAE encoding are outside this contract.

## Three Layers

1. Model pipelines restore latent scaling/normalization, invoke the decoder,
   and convert outputs on the leader. FLUX.2-klein uses the upstream pipeline's
   latent output, preserving conditioning, denoising, and unpatchification.
2. Decoder family adapters preserve positions, special tokens, masks, and
   temporal caches/windows. `DecoderSpec` records only the forwards that need
   spatial context; original modules and weights stay intact.
3. Shared operators execute disjoint row bands. The context owns row lengths,
   rank mapping, halo exchange, K/V collection, and leader-only output gathering.

| Operator dependency | Rule |
| --- | --- |
| Pointwise, channel-only norm, residual, nearest upsampling | Execute locally with the original module |
| Finite spatial neighborhood | Exchange the required halo at that layer, with padding only at global boundaries |
| Spatial reduction (GroupNorm) | Merge shard-size-weighted central moments in FP32, or FP64 for FP64 input |
| Global attention | Local Q, one combined K/V all-gather, SDPA, local output |
| Coordinates and masks | Derive from full latent geometry and global token positions |
| Replicated special tokens | Queries may be replicated; give K/V exactly one owner |
| Final image/video | Gather only to the leader; postprocess after assembly |

Transport padding is removed before attention. There is no latent overlap
blending. Results preserve full-image operator semantics, subject to floating
point reduction and kernel differences rather than bitwise identity.

## Families

| Models | Decoder | Adaptation |
| --- | --- | --- |
| Z-Image, FLUX.1 | AutoencoderKL 2D CNN | Convolution halos, global GroupNorm, AGKV |
| LLaDA-Image, FLUX.2-klein | AutoencoderKLFlux2 2D CNN | Same operators; retain latent BatchNorm and patch conversion |
| Wan | Wan causal 3D CNN | Spatial halos, per-frame AGKV, original temporal cache |
| Qwen-Image | Qwen causal 3D CNN | Same rules with its cache and output-frame layout |
| Hunyuan Image 3 | Released AutoencoderKLConv3D | 3D halos, global GroupNorm, spatiotemporal AGKV, DCAE upsampling |
| MiniMax-H3 video | Released ViT3DDecoder | Global RoPE coordinates, AGKV, single-owner register/CLS K/V, original temporal windows |

Hunyuan's input-size-dependent temporal convolution chunker is bypassed during
parallel decode: uneven shards must not choose different numbers of collective
calls. Convolutions operate on the smaller local row bands instead.

MiniMax register/CLS queries are replicated, but only rank zero contributes their
K/V. Each shard keeps all time positions. The adapter preserves ordinary and
block-causal attention, suffix-token visibility, and partial-head RoPE. Output
unpatchification stays in the release implementation. Both serial and parallel
video decode disable spatial tiling, changing the old tile-based reference.
The release's temporal window/trim semantics and audio leader decoding remain.

## Adding a Family

Classify the decoder's local, finite-neighborhood, and global operations first.
Reuse shared collectives; add a `DecoderSpec` only for release-specific forwards.
Preserve global coordinates, mask meaning, and unique ownership of replicated
tokens. Avoid rank-local control flow that changes communication count/order.

Compare full output tensors with dense decoding on identical latents, including
uneven shards, batches, multiple frames, repeat calls, subgroup rank mapping,
and inference precision. Test nonzero attention residuals and special tokens.
A seam metric or equality with a tiled reference alone is insufficient.

Unsupported structures fall back to full-image leader decoding with a warning.
Standard Diffusers tiling must be disabled. Current convolutions require odd
centered kernels and unit height stride; other geometries need an explicit
adapter. Small inputs may also use leader decoding. Legacy tile primitives
remain exported for compatibility, but supported model decode paths do not use them.

See [validation and reproduction](../validation/vae-parallel.md) for
source revisions, numerical tests, pretrained measurements, and coverage limits.
