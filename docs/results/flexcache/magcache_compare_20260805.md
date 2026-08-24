# FlexCache MagCache official-profile validation

Run ID: `magcache_compare_20260805`

This is a compatibility/performance validation of the single official operating
point for each supported checkpoint, not a speed-quality sweep or Pareto claim.
All runs below were newly launched. No old timing was reused.

## Official profiles

| Model | Profile | Reference schedule | FlexCache reuse |
| --- | --- | ---: | ---: |
| Flux.1-dev | E024K5R01 | 19 / 28 | 19 / 28 |
| Qwen-Image | E006K2R02 | 13 / 50 per CFG branch | 13 / 50 per branch |
| Wan2.1 T2V 1.3B | E012K4R02 | 29 / 50 per CFG branch | 29 / 50 per branch |

Wan's official conditional and unconditional ratio arrays differ slightly, but
produce the same default reuse schedule. FlexCache computes every published
branch schedule and uses their intersection, so serial CFG, CFP, and CP ranks
take identical fresh/reuse control flow.

## Official-size single-GPU results

Hardware: one H20. Backend: ChituDiffusion EPE transformer with AGKV attention,
BF16. Timings include the full `generate()` call, including fixed text/VAE work.

| Model and workload | Baseline | MagCache | Speedup | Official report | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flux.1, 1024x1024, 28 steps | 24.83s | 8.31s | 2.99x | about 2.8x | 17.06 | 0.6529 | 0.2963 |
| Qwen-Image, 1664x928, 50 steps | 100.45s | 73.27s | 1.37x | about 1.5x | 35.90 | 0.9869 | 0.0172 |
| Wan 1.3B, 480x832x81, 50 steps | 263.69s | 116.49s | 2.26x | 189s to 68s (2.78x) | 16.00 | 0.7184 | 0.2043 |

Quality metrics compare each MagCache output with its same-backend, same-seed
baseline. Image metrics are computed once over RGB pixels; Wan metrics are the
mean across all 81 decoded frames. The official repository publishes visual
examples and approximate latency, not paired PSNR/SSIM/LPIPS values.

The schedule-only upper bounds, assuming skipped DiT steps cost zero and all
other work is also zero, are 3.11x for Flux (9 fresh steps), 1.35x for Qwen
(37 fresh steps), and 2.38x for Wan (21 fresh steps). The paired FlexCache
measurements reach about 96%, 101% (timing noise), and 95% of those bounds.
The published Qwen 1.5x and Wan 2.78x numbers exceed their own schedule-only
bounds, so they cannot be reproduced by any same-backend paired timing using
only the published skip schedules; those README values are approximate and/or
include differences beyond MagCache skipping.

At the smaller smoke workloads, Flux 512x512 measured 7.45s to 3.75s (1.99x),
Qwen 512x512 measured 16.64s to 12.86s (1.29x), and Wan 17 frames measured
48.15s to 29.02s (1.66x). The lower end-to-end speedups are expected because
fixed text encoding and VAE work dominate more of these small requests.

## Parallel smoke

- Flux.1 static NCCL CP2 (Ulysses degree 2): completed, 19/28 reuse steps.
- Qwen-Image CFP2: completed, 13/50 reuse steps on each rank-local branch.
- Wan 1.3B static CP2 with serial CFG: completed, 29/50 reuse steps per branch.

The schedule is constructed from static published ratios and the replicated
step index. No rank-local hidden-state magnitude participates in the decision.

## Commands

Representative MagCache commands:

```bash
# Flux.1
bash tools/cluster/srun_direct.sh 1 1 -m chitu_diffusion.cli generate --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --output outputs/flexcache/magcache_compare_20260805/flux1024_mag.png \
  --prompt "A photo of a black bicycle." --height 1024 --width 1024 \
  --steps 28 --guidance-scale 3.5 --seed 42 --cache-strategy magcache

# Qwen-Image
bash tools/cluster/srun_direct.sh 1 1 -m chitu_diffusion.cli generate --model qwen-image \
  --model-path /path/to/Qwen-Image \
  --output outputs/flexcache/magcache_compare_20260805/qwen1664_mag.png \
  --height 928 --width 1664 --steps 50 --true-cfg-scale 4 --seed 42 \
  --cache-strategy magcache

# Wan2.1 T2V 1.3B
bash tools/cluster/srun_direct.sh 1 1 -m chitu_diffusion.cli generate --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --output outputs/flexcache/magcache_compare_20260805/wan81_mag.mp4 \
  --height 480 --width 832 --frames 81 --steps 50 --seed 42 \
  --cache-strategy magcache
```

The baseline commands are identical except for
`--cache-strategy none`. The CP2 runs add
`--attention-mode ulysses --ulysses-degree 2`; Qwen CFP2 uses two ranks with
`--cfg-parallel`.

## Reference compatibility note

The checked-out Flux reference directly replaces
`FluxTransformer2DModel.forward` and targets an older Diffusers
`FluxSingleTransformerBlock` signature. It fails against the installed
Diffusers version because the current single block requires separate
`encoder_hidden_states`. Therefore an unmodified reference-vs-FlexCache
pixel-exact run on one installed backend is not possible. FlexCache implements
the same image-token residual and reuse schedule through temporary block hooks,
and its tests cover both the double/single atomic backbone and the published
step schedules without changing model or pipeline source.
