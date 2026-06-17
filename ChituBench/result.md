# ChituBench Results

This file collects the numeric tables and key figures for each completed
experiment. The README remains the worklog; this page is the compact result
view.

## flux1_dev_attention

Model: `Flux1-dev`

Family: attention backend, no parallelism, no FlexCache

Run: `flux1_attn_50step_20260613_121311`

Command:

```bash
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=3 \
CHITUBENCH_WARMUP_RUNS=1 \
CHITUBENCH_RUN_ID=flux1_attn_50step_20260613_121311 \
CHITUBENCH_HPSV3_CONFIG=ChituBench/results/hpsv3_assets/HPSv3_7B.local.yaml \
CHITUBENCH_HPSV3_CHECKPOINT=ChituBench/results/hpsv3_assets/HPSv3.chitu_compat.safetensors \
ChituBench/scripts/run_flux1_attention.sh
```

Notes:

- Flux1-dev uses 50 denoising steps.
- Each case uses 3 prompts x 3 seeds = 9 measured images, plus 1 warmup image.
- Quality is measured against `origin_flash` for the same prompt and seed.
- HPSv3 was computed on a Slurm compute node because it requires CUDA.

### Summary

| case | tasks | DiT forward mean (s) | speedup vs origin | PSNR | SSIM | 1-LPIPS | HPSv3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| origin_flash | 9 | 37.960 | 1.000 | inf | 1.0000 | 1.0000 | 13.461 |
| Torch | 9 | 79.111 | 0.480 | 39.859 | 0.9876 | 0.9961 | 13.422 |
| sage | 9 | 32.711 | 1.160 | 32.918 | 0.9595 | 0.9824 | 13.466 |
| sparge | 9 | 32.502 | 1.168 | 15.048 | 0.6442 | 0.6474 | 12.548 |

### Readout

- Torch is the native math SDPA control: it preserves image quality
  closely but is about 0.48x the speed of `origin_flash`.
- `sage` is the best point in this run: about 1.16x speedup with a small HPSv3
  gain and moderate pixel/perceptual drift.
- `sparge` is slightly faster than `sage`, but the quality drop is visible in
  PSNR, SSIM, 1-LPIPS, and HPSv3. It should not be treated as an accepted
  method point before improving the backend or its policy.

### Speed-Quality Trade-off

![flux1_dev_attention speed-quality trade-off](plots/flux1_dev_attention/speed_quality_flux1_attn_50step_20260613_121311.png)

### Visual Contact Sheet

![flux1_dev_attention contact sheet](plots/flux1_dev_attention/contact_sheet_flux1_attn_50step_20260613_121311.png)

## flux1_dev_flexcache

Model: `Flux1-dev`

Family: FlexCache strategies, Flash Attention backend, `cp=1`

Run: consolidated `flux1_flexcache_with_meancache_50step_20260616`, reusing
`flux1_teacache_fix_50step_20260614_1520`,
`flux1_cubic_4x4_w8c2_50step_20260614_1343`,
`flux1_taylorseer_mid_50step_20260615_1112`, and the original
`flux1_flexcache_50step_20260614_1200`, plus new MeanCache runs
`flux1_meancache25_e2e`, `flux1_meancache17_smoke_rerun`, and
`flux1_meancache10_e2e`

Command:

```bash
# Reuse completed runs by symlinking them into one consolidated result dir,
# then evaluate and collect once.
./.venv/bin/python ChituBench/scripts/evaluate_quality.py \
  ChituBench/results/flux1_dev_flexcache/flux1_flexcache_with_meancache_50step_20260616 \
  --origin-dir ChituBench/results/flux1_dev_flexcache/flux1_teacache_fix_50step_20260614_1520/chitubench-flux1-flexcache-origin_flash-20260614_152038-origin_flash

./.venv/bin/python ChituBench/scripts/collect.py \
  ChituBench/results/flux1_dev_flexcache/flux1_flexcache_with_meancache_50step_20260616 \
  --experiment-id flux1_dev_flexcache \
  --allow-partial \
  --title 'Flux1-dev FlexCache Trade-off' \
  --no-point-labels
```

Additional MeanCache end-to-end runs:

```bash
MASTER_PORT=63431 \
CHITUBENCH_RUN_ID=flux1_meancache25_e2e \
CHITUBENCH_CASE_ID=flux1_meancache25_e2e \
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=1 \
CHITUBENCH_WARMUP_RUNS=0 \
CHITUBENCH_IMAGE_SIZE=1024,1024 \
CHITUBENCH_PROMPT_FILE=ChituBench/prompts/flux1_attention.json \
CHITUBENCH_FLEXCACHE_PARAMS='{"strategy":"meancache","fresh_steps":25,"warmup":0,"cooldown":0,"use_jvp":true}' \
./.venv/bin/chitu run \
  ChituBench/results/flux1_dev_flexcache/configs/flux1_meancache17_smoke.yaml \
  --gpus-per-node 1 \
  --cfp 1
```

Notes:

- Flux1-dev uses 50 denoising steps.
- The consolidated chart reuses the existing 9-task strategy runs and adds
  three MeanCache runs with 3 prompts x 1 seed each.
- Quality is measured against `origin_flash` for the same prompt and seed.
- DiTango is excluded from this run because it is not fully usable yet.
- TeaCache rows use the Flux reference coefficients and now keep all four
  thresholds 0.25/0.40/0.60/0.80 so the family curve is visible.
- Cubic rows use the 4x4 spatial retest: `block_size=16`,
  `uniform_square_min_splits=4`, `warmup=8`, `cooldown=2`, `tau=8`, and target
  speedups 2/3/4/5.
- TaylorSeer f2o1/f4o1 rows come from the mid-speed retest. Together with the
  original f3o1/f5o2 rows, they form one TaylorSeer speed-quality curve.
- MeanCache is now adapted through the new FlexCache MeanCache interface for
  Flux1-dev, and the B=25/17/10 points are validated end to end on the real
  launch path.
- HPSv3 was recomputed for the original all-strategy run on a Slurm compute
  node. The consolidated MeanCache update is summarized with PSNR/SSIM/1-LPIPS.

### Summary

| case | tasks | DiT forward mean (s) | speedup vs origin | PSNR | SSIM | 1-LPIPS | HPSv3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| origin_flash | 9 | 38.183 | 1.000 | inf | 1.0000 | 1.0000 | - |
| teacache_t025 | 9 | 20.711 | 1.844 | 22.118 | 0.8402 | 0.8898 | - |
| teacache_t040 | 9 | 14.911 | 2.561 | 18.037 | 0.7517 | 0.8026 | - |
| teacache_t060 | 9 | 11.459 | 3.332 | 16.257 | 0.7078 | 0.7403 | - |
| teacache_t080 | 9 | 9.441 | 4.045 | 15.690 | 0.6890 | 0.7015 | - |
| blockdance_b18g2 | 9 | 33.990 | 1.120 | 33.129 | 0.9588 | 0.9833 | 13.435 |
| blockdance_b24g3 | 9 | 31.322 | 1.215 | 31.389 | 0.9482 | 0.9776 | 13.582 |
| pab_s2c3 | 9 | 29.064 | 1.310 | 25.952 | 0.8890 | 0.9330 | 13.475 |
| pab_s3c4 | 9 | 26.443 | 1.439 | 22.805 | 0.8303 | 0.8858 | 13.423 |
| cubic_s20_tau8_4x4_w8c2 | 9 | 23.930 | 1.595 | 29.211 | 0.9275 | 0.9621 | - |
| cubic_s30_tau8_4x4_w8c2 | 9 | 20.678 | 1.846 | 27.156 | 0.9028 | 0.9467 | - |
| cubic_s40_tau8_4x4_w8c2 | 9 | 17.696 | 2.157 | 25.616 | 0.8739 | 0.9241 | - |
| cubic_s50_tau8_4x4_w8c2 | 9 | 16.093 | 2.371 | 24.792 | 0.8572 | 0.9102 | - |
| taylorseer_f2o1 | 9 | 22.089 | 1.716 | 24.713 | 0.8992 | 0.9406 | - |
| taylorseer_f3o1 | 9 | 16.323 | 2.332 | 20.722 | 0.8158 | 0.8716 | 13.639 |
| taylorseer_f4o1 | 9 | 14.050 | 2.698 | 18.590 | 0.7596 | 0.8181 | - |
| taylorseer_f5o2 | 9 | 12.093 | 3.147 | 15.715 | 0.6979 | 0.7585 | 13.482 |
| flux1_meancache25_e2e | 3 | 19.009 | 2.009 | 28.684 | 0.9164 | 0.9488 | - |
| flux1_meancache17_smoke_rerun | 3 | 12.991 | 2.939 | 26.424 | 0.8750 | 0.9135 | - |
| flux1_meancache10_e2e | 3 | 7.654 | 4.989 | 19.888 | 0.7734 | 0.8438 | - |

### Readout

- TeaCache and MeanCache are now both visible as step-reduction curves. TeaCache
  reaches the highest raw speed among the older Flux1 strategies, but MeanCache
  is noticeably better in pixel and perceptual quality at similar or higher
  speedups.
- BlockDance is the most conservative useful acceleration family: b18g2 gives
  1.12x speedup with the best pixel/perceptual metrics among accelerated cases,
  while b24g3 reaches 1.22x with slightly more drift.
- The Cubic 4x4 retest forms the middle-to-high speed Pareto segment. The
  conservative target-2 point reaches 1.59x with PSNR 29.21 and 1-LPIPS
  0.9621, while target-4 reaches 2.16x with PSNR 25.62 and 1-LPIPS 0.9241.
- TaylorSeer is the fastest family. The new f2o1 point is a conservative 1.72x
  setting with better PSNR/1-LPIPS than TeaCache at similar speed; f3o1 reaches
  2.33x and has the highest HPSv3 in this prompt set. f4o1 lands near the
  requested 2.5x region at 2.70x, while f5o2 is the most aggressive 3.15x
  point.
- Flux1 MeanCache is now adapted through the new FlexCache interface and forms
  a clean three-point curve: `B=25` is the conservative quality-first point,
  `B=17` is a strong middle setting around 2.94x, and `B=10` pushes to 4.99x
  with a visible quality trade-off.

### Speed-Quality Trade-off

![flux1_dev_flexcache speed-quality trade-off](plots/flux1_dev_flexcache/speed_quality_flux1_flexcache_with_meancache_50step_20260616.png)

### Visual Contact Sheet

![flux1_dev_flexcache contact sheet](plots/flux1_dev_flexcache/contact_sheet_flux1_flexcache_with_meancache_50step_20260616.png)

## flux2_klein_attention

Model: `Flux2-klein-4B`

Family: attention backend, no parallelism, no FlexCache

Run: `flux2_klein_attn_50step_20260613_130859`

Command:

```bash
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=3 \
CHITUBENCH_WARMUP_RUNS=1 \
CHITUBENCH_RUN_ID=flux2_klein_attn_50step_20260613_130859 \
CHITUBENCH_HPSV3_CONFIG=ChituBench/results/hpsv3_assets/HPSv3_7B.local.yaml \
CHITUBENCH_HPSV3_CHECKPOINT=ChituBench/results/hpsv3_assets/HPSv3.chitu_compat.safetensors \
ChituBench/scripts/run_flux2_klein_attention.sh
```

Notes:

- Flux2-klein-4B uses 50 denoising steps.
- Each case uses 3 prompts x 3 seeds = 9 measured images, plus 1 warmup image.
- Quality is measured against `origin_flash` for the same prompt and seed.
- HPSv3 was computed on a Slurm compute node because it requires CUDA.

### Summary

| case | tasks | DiT forward mean (s) | speedup vs origin | PSNR | SSIM | 1-LPIPS | HPSv3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| origin_flash | 9 | 16.972 | 1.000 | inf | 1.0000 | 1.0000 | 12.264 |
| Torch | 9 | 35.056 | 0.484 | 36.146 | 0.9903 | 0.9929 | 12.209 |
| sage | 9 | 14.591 | 1.163 | 29.587 | 0.9677 | 0.9750 | 12.258 |
| sparge | 9 | 14.576 | 1.164 | 15.938 | 0.6544 | 0.6930 | 11.742 |

### Readout

- Torch remains the slow native math SDPA control: about 0.48x the
  speed of `origin_flash`, with quality close to the origin output.
- `sage` gives about 1.16x speedup and keeps HPSv3 nearly identical to
  `origin_flash`, with moderate pixel/perceptual drift.
- `sparge` is only marginally faster than `sage`, while quality drops heavily
  across PSNR, SSIM, 1-LPIPS, and HPSv3. It needs method-side improvement before
  becoming an acceptable open-source performance point for Flux2-klein.

### Speed-Quality Trade-off

![flux2_klein_attention speed-quality trade-off](plots/flux2_klein_attention/speed_quality_flux2_klein_attn_50step_20260613_130859.png)

### Visual Contact Sheet

![flux2_klein_attention contact sheet](plots/flux2_klein_attention/contact_sheet_flux2_klein_attn_50step_20260613_130859.png)

## qwen_image_attention

Model: `Qwen-Image`

Family: attention backend, no parallelism, no FlexCache

Run: `qwen_image_attn_50step_20260615_1550`

Additional FlashInfer probe: `chitubench-qwen-image-attn-flashinfer-20260617_121208-flashinfer`

Command:

```bash
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=3 \
CHITUBENCH_WARMUP_RUNS=1 \
CHITUBENCH_RUN_ID=qwen_image_attn_50step_20260615_1550 \
CHITUBENCH_HPSV3_CONFIG=ChituBench/results/hpsv3_assets/HPSv3_7B.local.yaml \
CHITUBENCH_HPSV3_CHECKPOINT=ChituBench/results/hpsv3_assets/HPSv3.chitu_compat.safetensors \
ChituBench/scripts/run_qwen_image_attention.sh
```

Notes:

- Qwen-Image uses 50 denoising steps at 1328x1328.
- Each case uses 1 prompt x 3 seeds = 3 measured images, plus 1 warmup image.
- Quality is measured against Flash Attention for the same prompt and seed.
- HPSv3 was computed on a Slurm GPU node and is shown in the visual contact
  sheet labels.
- Qwen-Image currently supports this benchmark on single-GPU attention only;
  sequence/context parallel attention is not included yet.
- FlashInfer was added as a follow-up single-prompt probe after the original
  four-backend sweep. Its first run includes JIT compilation; the table below
  reports a warmed 50-step run with the JIT cache already populated.

### Summary

| case | tasks | DiT forward mean (s) | speedup vs Flash Attention | PSNR | SSIM | 1-LPIPS | HPSv3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash Attention | 3 | 113.564 | 1.000 | inf | 1.0000 | 1.0000 | 12.761 |
| Torch | 3 | 335.033 | 0.339 | 34.913 | 0.9785 | 0.9942 | 12.677 |
| flashinfer | 1 | 125.264 | 0.907 | - | - | - | - |
| sage | 3 | 105.093 | 1.081 | 19.742 | 0.8222 | 0.9032 | 12.494 |
| sparge | 3 | 101.954 | 1.114 | 15.742 | 0.6549 | 0.7377 | 12.929 |

### Readout

- Flash Attention is the baseline for the current Qwen-Image Chitu attention
  adapter, because this run does not include an origin-flash path.
- Torch is the slow native math SDPA control. It preserves outputs
  relatively closely but runs at about 0.34x the speed of Flash Attention.
- `flashinfer` is functional through the Chitu attention backend and ran the
  Qwen-Image 50-step coffee prompt end to end, but this dense full-attention
  workload is slower than Flash Attention in the measured single-image run
  (125.264s vs 113.564s DiT-forward latency). It is not a better default for
  Qwen-Image yet.
- `sage` gives a modest 1.08x DiT-forward speedup, but it introduces visible
  output drift in this Qwen-Image processor path. Treat it as an experimental
  performance point rather than an accepted quality-preserving backend.
- `sparge` is the fastest backend in this run at 101.954s mean DiT latency, but
  it has the largest output drift. It is useful as a performance bound, not as
  a quality-preserving point yet.
- HPSv3 scores are close across the four cases and rank `sparge` highest on
  this single coffee prompt. That reward score should be read alongside
  PSNR/SSIM/LPIPS, which measure consistency against the Flash Attention baseline.

### Latency-Quality Trade-off

![qwen_image_attention latency-quality trade-off](plots/qwen_image_attention/speed_quality_qwen_image_attn_50step_20260615_1550.png)

### Visual Contact Sheet

![qwen_image_attention contact sheet](plots/qwen_image_attention/contact_sheet_qwen_image_attn_50step_20260615_1550.png)

## qwen_image_parallel

Model: `Qwen-Image`

Family: CFG/context parallel scaling, Flash Attention backend, no FlexCache

Run: `qwen_parallel_50step_20260616_stable`, plus the 2-GPU CFP1+UP2 retest
`qwen_image_cfp1up2_2gpu_20260616_155741` and 8-GPU CFP2+UP4 retest
`qwen_image_cfp2up4_8gpu_20260616_154411`

Command:

```bash
MASTER_PORT=62531 \
SRUN_EXTRA_ARGS='--exclusive --exclude=bjdb-h20-node-021' \
CHITUBENCH_RUN_ID=qwen_parallel_50step_20260616_stable \
CHITUBENCH_CASES=baseline_1gpu,cfp2_2gpu,cfp2up2_4gpu,cfp2ring2_4gpu \
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=1 \
CHITUBENCH_WARMUP_RUNS=0 \
CHITUBENCH_IMAGE_SIZE=1328,1328 \
CHITUBENCH_ATTN_TYPE=flash_attn \
ChituBench/scripts/run_qwen_image_parallel.sh
```

2-GPU CFP1+UP2 retest:

```bash
MASTER_PORT=62871 \
SRUN_EXTRA_ARGS='--exclusive --exclude=bjdb-h20-node-021' \
CHITUBENCH_RUN_ID=qwen_image_cfp1up2_2gpu_20260616_155741 \
CHITUBENCH_CASES=cfp1up2_2gpu \
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=1 \
CHITUBENCH_WARMUP_RUNS=0 \
CHITUBENCH_IMAGE_SIZE=1328,1328 \
ChituBench/scripts/run_qwen_image_parallel.sh
```

8-GPU CFP2+UP4 retest:

```bash
MASTER_PORT=62841 \
SRUN_EXTRA_ARGS='--exclusive --exclude=bjdb-h20-node-021' \
CHITUBENCH_RUN_ID=qwen_image_cfp2up4_8gpu_20260616_154411 \
CHITUBENCH_CASES=cfp2up4_8gpu \
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=1 \
CHITUBENCH_WARMUP_RUNS=0 \
CHITUBENCH_IMAGE_SIZE=1328,1328 \
ChituBench/scripts/run_qwen_image_parallel.sh
```

Notes:

- Qwen-Image uses 50 denoising steps at 1328x1328.
- Each case uses the single coffee-sign prompt with seed 42.
- Speed is measured from rank-0 `dit_forward` timer totals across the denoising
  loop. Quality metrics are intentionally omitted for this scaling run.
- The first formal launch on `bjdb-h20-node-021` failed during model load
  because GPU0 already had about 94GB occupied; the successful runs excluded
  that node.
- Qwen-Image CP uses the dedicated joint-attention path where text states stay
  replicated and image states are sharded. Ring/USP variants are not reported
  here because supporting them cleanly would require a separate Qwen joint
  attention sequence implementation.

### Summary

| case | GPUs | parallel mode | tasks | DiT forward mean (s) | speedup vs 1 GPU | efficiency |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| baseline_1gpu | 1 | none | 1 | 138.819 | 1.000 | 1.000 |
| cfp1up2_2gpu | 2 | Qwen image CP2 | 1 | 80.935 | 1.715 | 0.858 |
| cfp2_2gpu | 2 | CFG parallel | 1 | 69.569 | 1.995 | 0.998 |
| cfp2up2_4gpu | 4 | CFG parallel + Ulysses CP2 | 1 | 41.325 | 3.359 | 0.840 |
| cfp2ring2_4gpu | 4 | CFG parallel + ring CP2 | 1 | 41.154 | 3.373 | 0.843 |
| cfp2up4_8gpu | 8 | CFG parallel + Qwen image CP4 | 1 | 25.688 | 5.404 | 0.676 |

### Readout

- CFG parallel is almost ideal for this single-image Qwen-Image workload:
  `cfp2_2gpu` reaches 1.995x speedup.
- Pure Qwen image CP2 (`cfp1up2_2gpu`) reaches 1.715x speedup. It improves
  latency over the 1-GPU baseline, but is slower than CFG parallel alone on 2
  GPUs because this Qwen CP path gathers image K/V inside the joint-attention
  processor.
- Adding CP2 on top of CFG parallel improves total DiT latency to about 41.2s,
  or 3.36-3.37x over the 1-GPU baseline.
- On 4 GPUs, ring CP2 is slightly faster than Ulysses CP2 in this run
  (41.154s vs 41.325s), but the difference is small enough to treat as a tie
  until repeated runs are collected.
- The 8-GPU `cfp2up4_8gpu` retest is now the best supported Qwen-Image
  parallel point: 25.688s DiT-forward latency, 5.404x speedup over the 1-GPU
  baseline, and 67.6% parallel efficiency.
- The 8-GPU output was visually checked on the coffee-sign prompt and did not
  show black-image, noise-image, or obvious CP semantic collapse.

### Parallel Scaling

![qwen_image_parallel scaling](plots/qwen_image_parallel/parallel_scaling_qwen_parallel_50step_20260616_cfp1up2_cfp2up4.png)

## qwen_image_flexcache

Model: `Qwen-Image`

Family: FlexCache strategies, Flash Attention backend, CFP2

Run: `qwen_image_flexcache_50step_20260616`

Command:

```bash
# Reuse the completed 50-step coffee prompt runs by symlinking them into the
# qwen_image_flexcache result directory, then evaluate and collect once.
./.venv/bin/python ChituBench/scripts/evaluate_quality.py \
  ChituBench/results/qwen_image_flexcache/qwen_image_flexcache_50step_20260616 \
  --origin-dir ChituBench/results/qwen_image_attention/qwen_image_attn_50step_20260615_1550/chitubench-qwen-image-attn-torch-sdpa-20260615_154538-torch_sdpa \
  --skip-hpsv3

./.venv/bin/python ChituBench/scripts/collect.py \
  ChituBench/results/qwen_image_flexcache/qwen_image_flexcache_50step_20260616 \
  --experiment-id qwen_image_flexcache \
  --allow-partial \
  --title 'Qwen-Image FlexCache 50-step Trade-off'
```

Additional sweeps, sharing model loads:

```bash
MASTER_PORT=63121 \
CHITUBENCH_RUN_ID=qwen_flexcache_extra_meancache \
CHITUBENCH_CASES=flexcache_sweep \
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=1 \
CHITUBENCH_WARMUP_RUNS=0 \
CHITUBENCH_ATTN_TYPE=torch_sdpa \
CHITUBENCH_IMAGE_SIZE=1328,1328 \
CHITUBENCH_FLEXCACHE_SWEEP='[
  {"case_id":"qwen_meancache17_50_cfp2","flexcache_params":{"strategy":"meancache","fresh_steps":17,"warmup":0,"cooldown":0,"use_jvp":true}},
  {"case_id":"qwen_meancache10_50_cfp2","flexcache_params":{"strategy":"meancache","fresh_steps":10,"warmup":0,"cooldown":0,"use_jvp":true}}
]' \
./.venv/bin/python ChituBench/scripts/qwen_image_benchmark.py \
  --gpus-per-node 2 \
  --cfp 2
```

```bash
MASTER_PORT=63231 \
SRUN_EXTRA_ARGS='--exclusive --exclude=bjdb-h20-node-021' \
CHITU_RUN_TASK_ID=qwen_flexcache_extra_sweep \
CHITUBENCH_RUN_ID=qwen_flexcache_extra_sweep \
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=1 \
CHITUBENCH_WARMUP_RUNS=0 \
CHITUBENCH_ATTN_TYPE=torch_sdpa \
CHITUBENCH_IMAGE_SIZE=1328,1328 \
CHITUBENCH_FLEXCACHE_SWEEP='[
  {"case_id":"qwen_pab_s3c4_50_cfp2","flexcache_params":{"strategy":"pab","warmup":5,"cooldown":5,"skip_self_range":3,"skip_cross_range":4}},
  {"case_id":"qwen_pab_s4c5_50_cfp2","flexcache_params":{"strategy":"pab","warmup":5,"cooldown":5,"skip_self_range":4,"skip_cross_range":5}},
  {"case_id":"qwen_blockdance_g3_50_cfp2","flexcache_params":{"strategy":"blockdance","warmup":5,"cooldown":5,"boundary_block":20,"group_size":3,"start_fraction":0.40,"end_fraction":0.95}},
  {"case_id":"qwen_blockdance_g4_50_cfp2","flexcache_params":{"strategy":"blockdance","warmup":5,"cooldown":5,"boundary_block":20,"group_size":4,"start_fraction":0.40,"end_fraction":0.95}},
  {"case_id":"qwen_cubic20_50_cfp2","flexcache_params":{"strategy":"cubic","target_speedup":2.0,"warmup":7,"cooldown":3,"tau_max":8,"block_size":8,"uniform_square_min_splits":4}}
]' \
./.venv/bin/chitu run \
  ChituBench/results/qwen_image_flexcache/qwen_image_flexcache_50step_20260616/configs/qwen_flexcache_extra_sweep_cfp2.yaml \
  --gpus-per-node 2 \
  --cfp 2
```

Notes:

- Qwen-Image uses 50 denoising steps at 1328x1328.
- Each FlexCache point uses the same coffee-sign prompt with seed 42; the
  Flash Attention baseline reuses the existing three-seed attention-backend run.
- The result directory reuses completed PAB, BlockDance, MeanCache25, Cubic1.5,
  and Cubic3.0 runs via symlinks, then adds two batched sweeps for MeanCache
  and the missing PAB/BlockDance/Cubic points.
- PAB, BlockDance, Cubic, and MeanCache each have three points, so the
  speed-quality plot shows method curves rather than isolated single markers.
- Quality is measured against the Flash Attention coffee image with PSNR, SSIM,
  and 1-LPIPS. HPSv3 is skipped for this FlexCache trade-off pass.
- The speedup column uses the rank-0 `dit_forward` timer total, matching the
  other ChituBench FlexCache summaries.

### Summary

| case | DiT forward mean (s) | speedup vs Flash Attention | PSNR | SSIM | 1-LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash Attention | 113.564 | 1.000 | inf | 1.0000 | 1.0000 |
| qwen_pab50_cfp2 | 49.907 | 2.276 | 20.817 | 0.9083 | 0.9514 |
| qwen_blockdance50_cfp2 | 57.362 | 1.980 | 23.826 | 0.9500 | 0.9596 |
| qwen_cubic15_50_cfp2 | 51.216 | 2.217 | 19.108 | 0.8657 | 0.9162 |
| qwen_pab_s3c4_50_cfp2 | 46.011 | 2.468 | 17.227 | 0.8321 | 0.8779 |
| qwen_pab_s4c5_50_cfp2 | 43.362 | 2.619 | 17.683 | 0.8345 | 0.8916 |
| qwen_blockdance_g3_50_cfp2 | 55.468 | 2.047 | 23.841 | 0.9403 | 0.9593 |
| qwen_blockdance_g4_50_cfp2 | 54.554 | 2.082 | 23.710 | 0.9193 | 0.9578 |
| qwen_cubic20_50_cfp2 | 43.104 | 2.635 | 22.912 | 0.9351 | 0.9549 |
| qwen_cubic30_w9c1_tau10_50_cfp2 | 37.396 | 3.037 | 21.794 | 0.9118 | 0.9434 |
| qwen_meancache25_50_cfp2 | 31.410 | 3.616 | 24.507 | 0.9299 | 0.9533 |
| qwen_meancache17_50_cfp2 | 21.302 | 5.331 | 22.451 | 0.9161 | 0.9468 |
| qwen_meancache10_50_cfp2 | 12.490 | 9.092 | 10.375 | 0.3403 | 0.6008 |

### Readout

- MeanCache spans the widest range. `mc25` is the best quality-speed point in
  that family, `mc17` is the aggressive usable point, and `mc10` is a fast
  lower-quality bound.
- Cubic now forms a clearer middle frontier. The `cubic2` point is the best
  Cubic balance in this sweep: 2.63x speedup with 0.9549 1-LPIPS, while
  `cubic3.0` trades text quality for more speed.
- PAB speeds up as skip ranges increase, but the two more aggressive settings
  drop PSNR/LPIPS sharply on the coffee prompt.
- BlockDance preserves high fidelity across all three points, but its current
  Qwen-Image settings only move from 1.98x to 2.08x speedup, so it is quality
  preserving rather than latency optimal here.

### Speed-Quality Trade-off

![qwen_image_flexcache speed-quality trade-off](plots/qwen_image_flexcache/speed_quality_qwen_image_flexcache_50step_20260616.png)

### Visual Contact Sheet

![qwen_image_flexcache coffee contact sheet](plots/qwen_image_flexcache/contact_sheet_qwen_image_flexcache_coffee_50step_20260616.png)

## flux1_dev_sequence_parallel

Model: `Flux1-dev`

Family: sequence parallel scaling, Flash Attention backend, no FlexCache

Run: `flux1_sp_50step_20260613_144519`

Command:

```bash
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=3 \
CHITUBENCH_WARMUP_RUNS=1 \
CHITUBENCH_RUN_ID=flux1_sp_50step_20260613_144519 \
ChituBench/scripts/run_flux1_sequence_parallel.sh
```

Notes:

- Flux1-dev uses 50 denoising steps.
- Each case uses 3 prompts x 3 seeds = 9 measured images, plus 1 warmup image.
- All cases use `infer.attn_type=flash`.
- This experiment records speed only; quality metrics are intentionally omitted.

### Summary

| case | GPUs | parallel mode | tasks | DiT forward mean (s) | speedup vs 1 GPU | efficiency |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| baseline_1gpu | 1 | none | 9 | 38.027 | 1.000 | 1.000 |
| ring_2gpu | 2 | ring | 9 | 20.860 | 1.823 | 0.911 |
| ulysses_2gpu | 2 | Ulysses | 9 | 21.018 | 1.809 | 0.905 |
| ring_4gpu | 4 | ring | 9 | 12.238 | 3.107 | 0.777 |
| usp_r2u2_4gpu | 4 | USP r2u2 | 9 | 12.395 | 3.068 | 0.767 |
| ulysses_4gpu | 4 | Ulysses | 9 | 11.580 | 3.284 | 0.821 |
| ring_8gpu | 8 | ring | 9 | 10.354 | 3.673 | 0.459 |
| usp_r4u2_8gpu | 8 | USP r4u2 | 9 | 9.811 | 3.876 | 0.484 |
| ulysses_8gpu | 8 | Ulysses | 9 | 7.852 | 4.843 | 0.605 |

### Readout

- 2 GPU ring and Ulysses are close, with ring slightly faster in this run.
- 4 GPU Ulysses is the best CP4 point: 3.284x speedup with 82.1% parallel
  efficiency.
- 8 GPU Ulysses is the best overall point: 4.843x speedup with 60.5% parallel
  efficiency.
- CP8 still improves absolute latency, but the efficiency drop is clear. USP
  r4u2 improves over 8 GPU ring, while full Ulysses remains strongest.

### Parallel Scaling

![flux1_dev_sequence_parallel scaling](plots/flux1_dev_sequence_parallel/parallel_scaling_flux1_sp_50step_20260613_144519.png)

## flux2_klein_sequence_parallel

Model: `Flux2-klein-4B`

Family: sequence parallel scaling, Flash Attention backend, no FlexCache

Run: `flux2_klein_sp_4step_20260613_1545`

Command:

```bash
CHITUBENCH_STEPS=4 \
CHITUBENCH_NUM_SEEDS=3 \
CHITUBENCH_WARMUP_RUNS=1 \
CHITUBENCH_RUN_ID=flux2_klein_sp_4step_20260613_1545 \
ChituBench/scripts/run_flux2_klein_sequence_parallel.sh
```

Notes:

- Flux2-klein-4B uses 4 denoising steps.
- Each case uses 3 prompts x 3 seeds = 9 measured images, plus 1 warmup image.
- All cases use `infer.attn_type=flash`.
- This experiment records speed only; quality metrics are intentionally omitted.

### Summary

| case | GPUs | parallel mode | tasks | DiT forward mean (s) | speedup vs 1 GPU | efficiency |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| baseline_1gpu | 1 | none | 9 | 1.358 | 1.000 | 1.000 |
| ring_2gpu | 2 | ring | 9 | 0.732 | 1.857 | 0.928 |
| ulysses_2gpu | 2 | Ulysses | 9 | 0.740 | 1.836 | 0.918 |
| ring_4gpu | 4 | ring | 9 | 0.435 | 3.119 | 0.780 |
| usp_r2u2_4gpu | 4 | USP r2u2 | 9 | 0.439 | 3.095 | 0.774 |
| ulysses_4gpu | 4 | Ulysses | 9 | 0.409 | 3.320 | 0.830 |
| ring_8gpu | 8 | ring | 9 | 0.344 | 3.951 | 0.494 |
| usp_r4u2_8gpu | 8 | USP r4u2 | 9 | 0.335 | 4.057 | 0.507 |
| ulysses_8gpu | 8 | Ulysses | 9 | 0.278 | 4.880 | 0.610 |

### Readout

- 2 GPU ring and Ulysses are close, with ring slightly faster in this run.
- 4 GPU Ulysses is the best CP4 point: 3.320x speedup with 83.0% parallel
  efficiency.
- 8 GPU Ulysses is the best overall point: 4.880x speedup with 61.0% parallel
  efficiency.
- Flux2-klein still benefits from CP8, but because the default workload is only
  4 denoising steps, communication and setup overhead are visible in the
  efficiency curve.

### Parallel Scaling

![flux2_klein_sequence_parallel scaling](plots/flux2_klein_sequence_parallel/parallel_scaling_flux2_klein_sp_4step_20260613_1545.png)
