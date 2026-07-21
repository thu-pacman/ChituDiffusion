# Model Kernel Recipes

ChituDiffusion defaults to the best end-to-end recipe verified on NVIDIA H20.
Users normally only need `infer.attn_type=auto`. Overrides are intended for
reproducibility, debugging, or hardware-specific benchmarking.

All gains below are measurements, not sums of isolated kernel speedups. A
mathematically equivalent kernel may produce a different diffusion trajectory
because floating-point reductions are not associative; this alone is not
treated as a quality regression.

## Shared attention

- Default: `infer.attn_type=auto`, currently selecting PyTorch SDPA on H20.
- Override: `torch_sdpa`, `torch_sdpa_math`, `flash_attn3`, `flash_attn`,
  `flashinfer`, `sage`, or `sparge`.
- H20 result: SDPA was `0-3%` faster than FA3 on tested dense diffusion shapes.
  FA3 was about `1.5x` faster than FA2 and remains useful as an explicit Hopper
  backend. Missing explicit backends fail loudly; `auto` always falls back.

## Flux1

Default recipe:

```bash
infer.attn_type=auto
```

There is no additional model-specific fusion. On Flux1-dev 1024px, SDPA reached
`711.99 ms/step`; FA3 reached `762.95 ms/step`, while FA2 reached
`770.11 ms/step`. Keep `auto` unless comparing attention implementations.

## Flux2

Default recipe:

```bash
infer.attn_type=auto
CHITU_FLUX2_FUSED_QKNORM_ROPE=auto
CHITU_FLUX2_SILU_MUL_FUSION=auto
```

- QK RMSNorm+RoPE: automatically attempted on eager SM90, FP16/BF16,
  contiguous BSHD with head dimension 128. Unsupported hardware, compilation,
  or layout falls back to PyTorch. Set `off` to disable. H20 measured about
  `15x` kernel speedup and about `1.5%` end-to-end improvement.
- SiLU×mul: automatically attempted for contiguous CUDA BF16; all unsupported
  cases fall back. Set `off` to disable or `triton` to request the same adapter.
  H20 measured `4.72x` kernel speedup and about `0.4%` steady-state end-to-end
  improvement.

The two gains were measured independently and must not be added mechanically.

## Qwen-Image

Default recipe:

```bash
infer.attn_type=auto
CHITU_QWEN_IMAGE_MODULATION_FUSION=off
```

- Ordinary single requests have no padding mask, so SDPA is the measured best
  path and padded-varlen does not run.
- For padded request batches on Hopper, `infer.attn_type=flash_attn3` enables
  Triton pack/scatter plus FA3 varlen when a boolean valid-token mask is
  present. The measured `[1,5120,24,128]` shape was `1.213x` faster than masked
  SDPA. All-valid, unmasked, and context-parallel inputs keep their normal path.
- Select01 modulation is benchmark-only. It has `5.98-7.02x` isolated speedup,
  but lacks a representative end-to-end gate and is not part of the recipe.

## Z-Image

Default recipe:

```bash
infer.attn_type=auto
CHITU_Z_IMAGE_NORM_FUSION=auto
```

`auto` selects `diffusers_fp32`: fused RMSNorm+scale and
RMSNorm+tanh(gate)+residual while preserving the diffusers FP32 reduction
policy. H20 measured `11.86-15.17x` isolated speedup and `2.67%` denoise
improvement.

Overrides:

- `diffusers_fp32`: the default and recommended policy.
- `native_bf16`: reproduce SGLang's BF16 variance policy; it gave no additional
  end-to-end speedup in the measured recipe.
- `off`: disable the fusion.

## Wan 2.1 / Wan 2.2

Default recipe:

```bash
infer.attn_type=auto
CHITU_WAN_VAE_CAT_PAD=off
CHITU_CP_ATTN_STATE_MERGE=off
```

- The VAE cat+pad mover was bitwise exact and `1.84-2.03x` faster in isolation,
  but made real VAE decode `1.95x` slower because many small launches dominated.
- The CP state merge had no measurable steady-state gain and added `18.3%`
  first-use JIT overhead in a real 2×H20 Ring run.

Both remain disabled for performance reasons, not because bitwise identity is
required. `CHITU_WAN_VAE_CAT_PAD=triton` is retained only for experiments;
production CP merge activation is intentionally unavailable.

## Reproducibility

To disable all newly adopted model-specific defaults:

```bash
CHITU_Z_IMAGE_NORM_FUSION=off \
CHITU_FLUX2_FUSED_QKNORM_ROPE=off \
CHITU_FLUX2_SILU_MUL_FUSION=off \
chitu ...
```

Detailed shapes, quality comparisons, commands, and rejected candidates are in
[`worklog.md`](worklog.md).
