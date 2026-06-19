# Wan2.1-T2V-1.3B Attention Benchmark Worklog

Date: 2026-06-17

Repo: `/home/chenyy/WORK/cyy/ChituDiffusion`

## Current Status

Wan2.1-T2V-1.3B attention benchmark has completed one accepted ChituBench run.

Completed run:

```text
ChituBench/results/wan2_1_t2v_1_3b_attention/wan21_13b_attn_2video_50step_20260617
```

Published docs updated:

- `README.md`
- `ChituBench/result.md`

Plots copied to:

```text
ChituBench/plots/wan2_1_t2v_1_3b_attention/
```

## Benchmark Protocol Used

- Model: `Wan2.1-T2V-1.3B`
- Family: attention backend comparison
- Cases: `origin_flash`, `torch_sdpa`, `sage`, `sparge`
- Steps: 50
- Frame count: 81
- Video size: 832x480
- Prompts: 2
- Seeds: 1, seed 42
- Warmup videos: 0
- Videos per case: exactly 2
- Quality reference: matching `origin_flash` output for the same prompt and seed
- Required quality metrics: PSNR, 1-LPIPS, HPSv3
- Optional metric also computed: SSIM
- HPSv3 frame rule: middle frame of each video

Prompts:

1. Text/sign prompt: `A steady video of a small storefront sign that clearly reads CHITU BENCH, warm indoor light, gentle camera movement.`
2. No-text prompt: `A cat walking on grass, natural daylight, smooth realistic motion, no text or subtitles.`

Command:

```bash
CHITUBENCH_RUN_ID=wan21_13b_attn_2video_50step_20260617 \
CHITUBENCH_STEPS=50 \
CHITUBENCH_NUM_SEEDS=1 \
CHITUBENCH_WARMUP_RUNS=0 \
CHITUBENCH_FRAME_NUM=81 \
bash ChituBench/scripts/run_wan2_1_t2v_1_3b_attention.sh
```

## Result Snapshot

| case | videos | DiT forward mean (s) | speedup vs origin_flash | PSNR | SSIM | 1-LPIPS | HPSv3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| origin_flash | 2 | 311.820 | 1.000 | inf | 1.0000 | 1.0000 | 10.226 |
| torch_sdpa | 2 | 234.372 | 1.330 | 30.594 | 0.9542 | 0.9884 | 10.226 |
| sage | 2 | 166.289 | 1.875 | 19.116 | 0.7946 | 0.8888 | 10.226 |
| sparge | 2 | 139.942 | 2.228 | 12.742 | 0.4511 | 0.6105 | 10.226 |

Interpretation as of this checkpoint:

- `origin_flash` means FlashAttention v2 via `infer.attn_type: flash`, normalized to `flash_attn`.
- `torch_sdpa` means PyTorch `scaled_dot_product_attention` with automatic kernel selection. It is not forced math attention.
- In this environment, PyTorch SDPA auto has flash, efficient, math, and cuDNN SDPA backends enabled, so its speedup over external FlashAttention v2 is plausible and needs a backend-specific probe before making a stronger claim.
- Wan cross-attention still calls `chitu_diffusion.modules.attention.wan_attention.attention()`, which uses FlashAttention when available. The attention benchmark mainly swaps Wan self-attention backend.
- `torch_sdpa` is the closest quality-preserving point in this two-video run.
- `sparge` is the fastest point but has large PSNR/SSIM/1-LPIPS drift.

## Code And Artifact Changes

Added benchmark assets:

- `ChituBench/prompts/wan2_1_t2v_1_3b_attention.json`
- `ChituBench/configs/wan2_1_t2v_1_3b/attention/origin_flash.yaml`
- `ChituBench/configs/wan2_1_t2v_1_3b/attention/torch_sdpa.yaml`
- `ChituBench/configs/wan2_1_t2v_1_3b/attention/sage.yaml`
- `ChituBench/configs/wan2_1_t2v_1_3b/attention/sparge.yaml`
- `ChituBench/scripts/wan_video_benchmark.py`
- `ChituBench/scripts/run_wan2_1_t2v_1_3b_attention.sh`

Updated shared ChituBench utilities:

- `ChituBench/scripts/evaluate_quality.py`
  - Added MP4 output discovery.
  - Added video frame loading/alignment for PSNR and SSIM.
  - Uses representative video frames for LPIPS and HPSv3.
- `ChituBench/scripts/make_contact_sheet.py`
  - Added MP4 middle-frame contact sheet support.

Updated docs:

- `README.md`
  - Added Wan result to News and ChituBench Highlights.
- `ChituBench/result.md`
  - Added `wan2_1_t2v_1_3b_attention` result section.

Updated local Codex skill:

- `/home/chenyy/.codex/skills/chitubench-model-evaluator/SKILL.md`
- `/home/chenyy/.codex/skills/chitubench-model-evaluator/references/chitubench-flow.md`

Important skill rules recorded there:

- Video benchmark default is exactly 2 prompts, 1 seed, and 0 warmup videos.
- Do not run 3-seed video sweeps unless explicitly requested.
- Each method/case should produce at most 2 videos.
- Accepted benchmarks require PSNR, 1-LPIPS, and HPSv3.
- Reuse matching single-GPU FlashAttention baseline/reference when possible.

## Validation Already Done

Commands that passed:

```bash
bash -n ChituBench/scripts/run_wan2_1_t2v_1_3b_attention.sh
.venv/bin/python -m py_compile \
  ChituBench/scripts/wan_video_benchmark.py \
  ChituBench/scripts/evaluate_quality.py \
  ChituBench/scripts/make_contact_sheet.py
```

Additional checks:

- `summary.csv` contains all four cases.
- Each case has exactly 2 tasks/videos.
- `psnr_mean`, `one_minus_lpips_mean`, and `hpsv3_score_mean` are present for every case.
- Early smoke and accidental extra-run artifacts were removed; only the formal Wan result remains under `ChituBench/results/wan2_1_t2v_1_3b_attention/`.

## Known Issues / Follow-up

1. Clarify naming in docs and plots. (CODE FIX DONE 2026-06-19; PLOT REGEN PENDING)
   - `origin_flash` is FlashAttention v2 baseline.
   - `torch_sdpa` is PyTorch SDPA auto backend, not "Flash Attention" and not forced math.
   - Root cause: `collect.py` (`display_label`, legend `family_labels`) and
     `make_contact_sheet.py` (`FAMILY_STYLES`, `display_case`) hard-mapped the
     `torch_sdpa` case to "Flash Attention" for every experiment. That label is
     a Qwen-specific convention (Qwen's `torch_sdpa` case is its flash baseline),
     so it incorrectly relabeled Wan's genuine PyTorch SDPA point.
   - Fix applied 2026-06-19:
     - The `torch_sdpa` -> "Flash Attention" relabel is now gated to qwen
       experiments only (`experiment_id.startswith("qwen")`); all other models,
       including Wan, display `torch_sdpa` as "Torch SDPA".
     - `torch_sdpa_math` -> "Torch" is unchanged everywhere (still accurate;
       Flux uses this case so Flux plots are untouched).
     - `make_contact_sheet.py` gained `--experiment-id`; all five runners
       (wan, qwen, flux1/flux2 attention, flux1 flexcache) now pass it.
   - PENDING: the committed Wan PNGs under
     `ChituBench/plots/wan2_1_t2v_1_3b_attention/` were generated before the fix
     and still show "Flash Attention". They must be regenerated once a working
     shell is available (the agent shell was non-functional on 2026-06-19, so
     `py_compile` and plot regen could not be run). Regen commands:
     ```bash
     cd /home/chenyy/WORK/cyy/ChituDiffusion
     .venv/bin/python -m py_compile ChituBench/scripts/collect.py ChituBench/scripts/make_contact_sheet.py
     RESULT_ROOT=ChituBench/results/wan2_1_t2v_1_3b_attention/wan21_13b_attn_2video_50step_20260617
     .venv/bin/python ChituBench/scripts/collect.py "$RESULT_ROOT" \
       --experiment-id wan2_1_t2v_1_3b_attention \
       --title "Wan2.1 T2V 1.3B Attention Backend Performance"
     .venv/bin/python ChituBench/scripts/make_contact_sheet.py "$RESULT_ROOT" \
       --seed 42 \
       --title "Wan2.1 T2V 1.3B Attention Backend Visual Check" \
       --experiment-id wan2_1_t2v_1_3b_attention \
       --cases origin_flash torch_sdpa sage sparge --frame-index -1
     cp "$RESULT_ROOT/plots/speed_quality.png" \
       ChituBench/plots/wan2_1_t2v_1_3b_attention/speed_quality_wan21_13b_attn_2video_50step_20260617.png
     cp "$RESULT_ROOT/visuals/contact_sheet.png" \
       ChituBench/plots/wan2_1_t2v_1_3b_attention/contact_sheet_wan21_13b_attn_2video_50step_20260617.png
     ```
   - Re-verify the qwen plots are unchanged after regen (qwen runner now passes
     `--experiment-id qwen_image_attention`, preserving "Flash Attention").

2. Add an SDPA backend probe.
   - Run small controlled probes for:
     - PyTorch SDPA auto
     - PyTorch SDPA forced FLASH_ATTENTION
     - PyTorch SDPA forced CUDNN_ATTENTION
     - PyTorch SDPA forced EFFICIENT_ATTENTION
     - PyTorch SDPA forced MATH
     - external `flash_attn.flash_attn_func`
   - Goal: explain why PyTorch SDPA auto is faster than external FlashAttention v2 for this Wan self-attention shape.

3. Consider adding explicit ChituBench cases after the probe.
   - `torch_sdpa_math` for true torch/math reference.
   - Maybe `torch_sdpa_flash` / `torch_sdpa_cudnn` if the backend wrapper is extended to force those kernels.

4. Revisit README wording after the probe.
   - Current README says Torch SDPA is closest quality-preserving point, which is true for this run.
   - Avoid implying this is a pure torch/math implementation.

## Resume Pointers

Useful files to open first:

```text
ChituBench/result.md
ChituBench/scripts/run_wan2_1_t2v_1_3b_attention.sh
ChituBench/scripts/wan_video_benchmark.py
ChituBench/scripts/evaluate_quality.py
ChituBench/scripts/make_contact_sheet.py
chitu_diffusion/modules/attention/diffusion_attn_backend.py
chitu_diffusion/core/models/model_wan.py
```

Useful result files:

```text
ChituBench/results/wan2_1_t2v_1_3b_attention/wan21_13b_attn_2video_50step_20260617/summary.csv
ChituBench/results/wan2_1_t2v_1_3b_attention/wan21_13b_attn_2video_50step_20260617/quality/quality_summary.csv
ChituBench/plots/wan2_1_t2v_1_3b_attention/speed_quality_wan21_13b_attn_2video_50step_20260617.png
ChituBench/plots/wan2_1_t2v_1_3b_attention/contact_sheet_wan21_13b_attn_2video_50step_20260617.png
```

