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
| flux2_klein_attention | Flux2-klein-4B | attention backend | completed | `results/flux2_klein_attention/flux2_klein_attn_50step_20260613_130859` |

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
