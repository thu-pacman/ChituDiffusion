# ChituBench

ChituBench is the reproducible benchmark workspace for ChituDiffusion. It keeps
each experiment focused on one model and one method family, records both numeric
metrics and human-readable visual samples, and uses this README as the worklog.

## Protocol

- Run one model and one method family at a time.
- Keep each case pure: one case should represent one method choice.
- Use the same prompts, seeds, steps, size, and model checkpoint across cases in
  the same experiment.
- Record speed, PSNR, SSIM, 1-LPIPS, and HPSv3.
- Add a visual contact sheet for every experiment. Image models use generated
  images directly; video models use a fixed representative frame.
- If a method underperforms, record the issue here, fix the method, and create a
  new rerun instead of overwriting the previous result.

## Directory Layout

```text
ChituBench/
  configs/        # Renderable chitu run configs grouped by model and method family
  prompts/        # Prompt sets used by experiments
  scripts/        # Run, collect, quality, and visualization scripts
  results/        # Raw experiment outputs
  plots/          # Selected final figures
```

## Experiment Index

| id | model | family | status | result |
| --- | --- | --- | --- | --- |
| flux1_dev_attention | Flux1-dev | attention backend | completed | `results/flux1_dev_attention/flux1_attn_50step_20260613_121311` |
| flux1_dev_sequence_parallel | Flux1-dev | sequence parallel scaling | completed | `results/flux1_dev_sequence_parallel/flux1_sp_50step_20260613_144519` |
| flux1_dev_flexcache | Flux1-dev | FlexCache strategies | completed | `results/flux1_dev_flexcache/flux1_flexcache_50step_20260614_1200` |
| flux2_klein_attention | Flux2-klein-4B | attention backend | completed | `results/flux2_klein_attention/flux2_klein_attn_50step_20260613_130859` |
| flux2_klein_sequence_parallel | Flux2-klein-4B | sequence parallel scaling | completed | `results/flux2_klein_sequence_parallel/flux2_klein_sp_4step_20260613_1545` |
| qwen_image_attention | Qwen-Image | attention backend | completed | `results/qwen_image_attention/qwen_image_attn_50step_20260615_1550` |

## Running Flux1-dev Attention

Quick smoke:

```bash
CHITUBENCH_STEPS=4 \
CHITUBENCH_NUM_SEEDS=1 \
CHITUBENCH_WARMUP_RUNS=0 \
ChituBench/scripts/run_flux1_attention.sh
```

Full run:

```bash
ChituBench/scripts/run_flux1_attention.sh
```

HPSv3 uses a local Qwen2-VL snapshot and a Chitu-compatible HPSv3 checkpoint.
Prepare them once, then pass the generated paths to the run script:

```bash
python ChituBench/scripts/prepare_hpsv3.py \
  --source-config .venv/lib/python3.12/site-packages/hpsv3/config/HPSv3_7B.yaml \
  --model-path ~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/<snapshot-id> \
  --source-checkpoint ~/.cache/huggingface/hub/models--MizzenAI--HPSv3/blobs/<hpsv3-blob> \
  --output-dir ChituBench/results/hpsv3_assets

CHITUBENCH_HPSV3_CONFIG=ChituBench/results/hpsv3_assets/HPSv3_7B.local.yaml \
CHITUBENCH_HPSV3_CHECKPOINT=ChituBench/results/hpsv3_assets/HPSv3.chitu_compat.safetensors \
ChituBench/scripts/run_flux1_attention.sh
```

## Worklog

### 2026-06-13 - ChituBench reset

- Created ChituBench to replace the previous mixed `key_exps` benchmark layout.
- First target: `Flux1-dev` attention backends without parallelism or FlexCache.
- Metrics: mean DiT forward time, speedup vs `origin_flash`, PSNR, SSIM,
  1-LPIPS, HPSv3.
- Visual check: image contact sheet with prompts as rows and methods as columns.

### 2026-06-13 - flux1_dev_attention smoke

- Command:
  `CHITUBENCH_STEPS=4 CHITUBENCH_NUM_SEEDS=1 CHITUBENCH_WARMUP_RUNS=0 CHITUBENCH_RUN_ID=smoke_20260613_115806 ChituBench/scripts/run_flux1_attention.sh`
- Result: `ChituBench/results/flux1_dev_attention/smoke_20260613_115806`.
- Status: passed end to end through `chitu run`; HPSv3 was skipped because
  `CHITUBENCH_HPSV3_CONFIG` and `CHITUBENCH_HPSV3_CHECKPOINT` were not set.
- Smoke speed summary: `origin_flash` 3.077s, `torch_sdpa_math` 6.375s,
  `sage` 2.991s, `sparge` 2.776s mean DiT forward time.
- Smoke quality summary: `torch_sdpa_math` PSNR 50.563 / SSIM 0.9968 /
  1-LPIPS 0.9992; `sage` PSNR 48.512 / SSIM 0.9962 / 1-LPIPS 0.9986;
  `sparge` PSNR 20.847 / SSIM 0.8138 / 1-LPIPS 0.8166.
- Visuals: `plots/speed_quality.png` and `visuals/contact_sheet.png` under the
  result directory. This is a smoke check only, not the accepted final benchmark.

### 2026-06-13 - flux1_dev_attention 50-step run

- Command:
  `CHITUBENCH_STEPS=50 CHITUBENCH_NUM_SEEDS=3 CHITUBENCH_WARMUP_RUNS=1 CHITUBENCH_RUN_ID=flux1_attn_50step_20260613_121311 CHITUBENCH_HPSV3_CONFIG=ChituBench/results/hpsv3_assets/HPSv3_7B.local.yaml CHITUBENCH_HPSV3_CHECKPOINT=ChituBench/results/hpsv3_assets/HPSv3.chitu_compat.safetensors ChituBench/scripts/run_flux1_attention.sh`
- Result: `ChituBench/results/flux1_dev_attention/flux1_attn_50step_20260613_121311`.
- Status: four backend cases completed with 50 steps, 3 prompts, and 3 seeds.
  HPSv3 was recomputed on a Slurm compute node because it requires CUDA.
- Speed summary: `origin_flash` 37.960s, `torch_sdpa_math` 79.111s,
  `sage` 32.711s, `sparge` 32.502s mean DiT forward time.
- Quality summary: `torch_sdpa_math` PSNR 39.859 / SSIM 0.9876 / 1-LPIPS
  0.9961 / HPSv3 13.422; `sage` PSNR 32.918 / SSIM 0.9595 / 1-LPIPS
  0.9824 / HPSv3 13.466; `sparge` PSNR 15.048 / SSIM 0.6442 / 1-LPIPS
  0.6474 / HPSv3 12.548.
- Readout: `sage` is the current best trade-off point. `sparge` is slightly
  faster but quality is not acceptable yet and should be improved before using
  it as an open-source performance point.
- Visuals: selected figures are copied to `ChituBench/plots/flux1_dev_attention`.

### 2026-06-13 - flux2_klein_attention 50-step run

- Command:
  `CHITUBENCH_STEPS=50 CHITUBENCH_NUM_SEEDS=3 CHITUBENCH_WARMUP_RUNS=1 CHITUBENCH_RUN_ID=flux2_klein_attn_50step_20260613_130859 CHITUBENCH_HPSV3_CONFIG=ChituBench/results/hpsv3_assets/HPSv3_7B.local.yaml CHITUBENCH_HPSV3_CHECKPOINT=ChituBench/results/hpsv3_assets/HPSv3.chitu_compat.safetensors ChituBench/scripts/run_flux2_klein_attention.sh`
- Result: `ChituBench/results/flux2_klein_attention/flux2_klein_attn_50step_20260613_130859`.
- Status: four backend cases completed with 50 steps, 3 prompts, and 3 seeds.
  HPSv3 was recomputed on a Slurm compute node because it requires CUDA.
- Speed summary: `origin_flash` 16.972s, `torch_sdpa_math` 35.056s,
  `sage` 14.591s, `sparge` 14.576s mean DiT forward time.
- Quality summary: `torch_sdpa_math` PSNR 36.146 / SSIM 0.9903 / 1-LPIPS
  0.9929 / HPSv3 12.209; `sage` PSNR 29.587 / SSIM 0.9677 / 1-LPIPS
  0.9750 / HPSv3 12.258; `sparge` PSNR 15.938 / SSIM 0.6544 / 1-LPIPS
  0.6930 / HPSv3 11.742.
- Readout: `sage` is again the best usable speed-quality point. `sparge` is
  marginally faster than `sage`, but the quality loss is large and should be
  fixed before presenting it as a strong result for Flux2-klein.
- Visuals: selected figures are copied to `ChituBench/plots/flux2_klein_attention`.

### 2026-06-13 - flux1_dev_sequence_parallel 50-step run

- Command:
  `CHITUBENCH_STEPS=50 CHITUBENCH_NUM_SEEDS=3 CHITUBENCH_WARMUP_RUNS=1 CHITUBENCH_RUN_ID=flux1_sp_50step_20260613_144519 ChituBench/scripts/run_flux1_sequence_parallel.sh`
- Result: `ChituBench/results/flux1_dev_sequence_parallel/flux1_sp_50step_20260613_144519`.
- Status: six sequence-parallel cases completed with Flash Attention, 50 steps,
  3 prompts, and 3 seeds. Quality metrics were intentionally not computed for
  this scaling experiment.
- Cases: `baseline_1gpu`, `ring_2gpu`, `ulysses_2gpu`, `ring_4gpu`,
  `usp_r2u2_4gpu`, `ulysses_4gpu`, `ring_8gpu`, `usp_r4u2_8gpu`, and
  `ulysses_8gpu`.
- Speed summary: 1 GPU baseline 38.027s; 2 GPU ring 20.860s / 1.823x; 2 GPU
  Ulysses 21.018s / 1.809x; 4 GPU ring 12.238s / 3.107x; 4 GPU USP r2u2
  12.395s / 3.068x; 4 GPU Ulysses 11.580s / 3.284x; 8 GPU ring 10.354s /
  3.673x; 8 GPU USP r4u2 9.811s / 3.876x; 8 GPU Ulysses 7.852s / 4.843x
  mean DiT forward time.
- Readout: Flux1-dev scales cleanly on sequence parallelism with Flash
  Attention. In this run, Ulysses is slightly slower than ring at 2 GPUs, best
  at 4 GPUs, and remains the best 8 GPU point. CP8 continues to improve
  absolute latency, but efficiency drops to 60.5%, showing clear communication
  overhead at larger CP size.
- Visuals: selected scaling figure is copied to
  `ChituBench/plots/flux1_dev_sequence_parallel`.

### 2026-06-14 - flux1_dev_flexcache 50-step run

- Command:
  `CHITUBENCH_RUN_ID=flux1_flexcache_50step_20260614_1200 CHITUBENCH_STEPS=50 CHITUBENCH_NUM_SEEDS=3 CHITUBENCH_WARMUP_RUNS=1 CHITUBENCH_HPSV3_CONFIG=ChituBench/results/hpsv3_assets/HPSv3_7B.local.yaml CHITUBENCH_HPSV3_CHECKPOINT=ChituBench/results/hpsv3_assets/HPSv3.chitu_compat.safetensors ChituBench/scripts/run_flux1_flexcache.sh`
- Result: `ChituBench/results/flux1_dev_flexcache/flux1_flexcache_50step_20260614_1200`.
- Status: completed on Flux1-dev with Flash Attention, `cp=1`, 50 steps, 3
  prompts, and 3 seeds. DiTango was intentionally excluded because it is not
  fully usable yet. HPSv3 was recomputed on a Slurm compute node after the first
  login-node quality pass reported `cuda is required by HPSv3`.
- Cases: `origin_flash`, TeaCache thresholds 0.30/0.50/0.70, PAB s2c3/s3c4,
  BlockDance b18g2/b24g3, TaylorSeer f3o1/f5o2, and Cubic 1.5x/2.0x targets.
- Speed summary: origin 38.061s; TeaCache 38.179-38.195s; PAB 29.064s and
  26.443s; BlockDance 33.990s and 31.322s; TaylorSeer 16.323s and 12.093s;
  Cubic 29.318s and 22.922s mean DiT forward time.
- Quality summary: TeaCache is identical to origin in this run. Among effective
  acceleration points, BlockDance keeps the highest PSNR/SSIM/1-LPIPS, Cubic
  provides a middle-speed point, and TaylorSeer provides the largest speedup
  while showing the largest pixel/perceptual drift. HPSv3 stays close across the
  set and ranks `taylorseer_f3o1` highest in this prompt mix.
- Readout: TeaCache is currently a no-op speed point for Flux1-dev under these
  parameters and should not be presented as acceleration. TaylorSeer f3o1 is the
  strongest headline speed point if visual quality is acceptable; BlockDance and
  Cubic are better conservative trade-off candidates.
- Visuals: selected speed-quality and contact-sheet figures are copied to
  `ChituBench/plots/flux1_dev_flexcache`.

### 2026-06-13 - flux2_klein_sequence_parallel 4-step run

- Command:
  `CHITUBENCH_STEPS=4 CHITUBENCH_NUM_SEEDS=3 CHITUBENCH_WARMUP_RUNS=1 CHITUBENCH_RUN_ID=flux2_klein_sp_4step_20260613_1545 ChituBench/scripts/run_flux2_klein_sequence_parallel.sh`
- Result: `ChituBench/results/flux2_klein_sequence_parallel/flux2_klein_sp_4step_20260613_1545`.
- Status: nine sequence-parallel cases completed with Flash Attention, 4 steps,
  3 prompts, and 3 seeds. Quality metrics were intentionally not computed for
  this scaling experiment.
- Implementation note: Flux2-klein builds rotary embeddings from text/image id
  tensors inside the model, so CP dispatch now splits `txt_ids` and `img_ids`
  together with hidden states. Ring P2P sends contiguous tensors to avoid
  non-dense distributed send failures.
- Cases: `baseline_1gpu`, `ring_2gpu`, `ulysses_2gpu`, `ring_4gpu`,
  `usp_r2u2_4gpu`, `ulysses_4gpu`, `ring_8gpu`, `usp_r4u2_8gpu`, and
  `ulysses_8gpu`.
- Speed summary: 1 GPU baseline 1.358s; 2 GPU ring 0.732s / 1.857x; 2 GPU
  Ulysses 0.740s / 1.836x; 4 GPU ring 0.435s / 3.119x; 4 GPU USP r2u2
  0.439s / 3.095x; 4 GPU Ulysses 0.409s / 3.320x; 8 GPU ring 0.344s /
  3.951x; 8 GPU USP r4u2 0.335s / 4.057x; 8 GPU Ulysses 0.278s / 4.880x
  mean DiT forward time.
- Readout: Flux2-klein keeps scaling through CP8, but efficiency drops faster
  than Flux1 because the 4-step workload is small and communication/setup
  overhead is proportionally larger. Ulysses is the best CP4 and CP8 point in
  this run.
- Visuals: selected scaling figure is copied to
  `ChituBench/plots/flux2_klein_sequence_parallel`.

### 2026-06-15 - qwen_image_attention 50-step run

- Command:
  `CHITUBENCH_STEPS=50 CHITUBENCH_NUM_SEEDS=3 CHITUBENCH_WARMUP_RUNS=1 CHITUBENCH_RUN_ID=qwen_image_attn_50step_20260615_1550 CHITUBENCH_HPSV3_CONFIG=ChituBench/results/hpsv3_assets/HPSv3_7B.local.yaml CHITUBENCH_HPSV3_CHECKPOINT=ChituBench/results/hpsv3_assets/HPSv3.chitu_compat.safetensors ChituBench/scripts/run_qwen_image_attention.sh`
- Result:
  `ChituBench/results/qwen_image_attention/qwen_image_attn_50step_20260615_1550`.
- Status: completed on Qwen-Image with single-GPU staged Chitu runtime,
  `cp=1`, 50 steps, 1 prompt, 3 measured seeds, and 1 warmup image per backend.
- Cases: `torch_sdpa`, `torch_sdpa_math`, `sage`, and `sparge`.
- Speed summary: `torch_sdpa` 113.564s; `torch_sdpa_math` 335.033s / 0.339x;
  `sage` 105.093s / 1.081x; `sparge` 101.954s / 1.114x mean DiT forward time.
- Quality summary uses `torch_sdpa` as the reference because Qwen-Image does not
  have an origin-flash case in this run. `torch_sdpa_math` is close to
  reference at PSNR 34.913 / SSIM 0.9785 / 1-LPIPS 0.9942 / HPSv3 12.677,
  while `sage` has visible drift at PSNR 19.742 / SSIM 0.8222 / 1-LPIPS
  0.9032 / HPSv3 12.494, and `sparge` drifts more at PSNR 15.742 / SSIM
  0.6549 / 1-LPIPS 0.7377 / HPSv3 12.929. The `torch_sdpa` reference HPSv3 is
  12.761.
- Readout: `sage` gives a modest speedup on Qwen-Image, but the current
  processor path changes results enough that it should be treated as an
  experimental performance point. `sparge` is the fastest point but has the
  largest quality drop in baseline-consistency metrics despite the highest
  HPSv3 on this single prompt. `torch_sdpa_math` is the slow quality-control
  point.
- Visuals: selected latency-quality and contact-sheet figures are copied to
  `ChituBench/plots/qwen_image_attention`.
