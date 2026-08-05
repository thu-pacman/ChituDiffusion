# MeanCache reference parity

Run ID: `meancache_compare_20260804`

## Setup

- H20, one GPU, BF16, 512×512, 50 steps, seed 42
- Prompt: `A red fox sitting in a snowy forest, cinematic photography`
- Qwen-Image true CFG 4; Z-Image guidance 4
- Baseline, B25, and all sweep points were newly generated. No prior timing was
  reused. CP2/CFP2 validation images were generated separately.
- Quality is measured against each backend's own no-cache baseline.

## Results

| Model | Backend | Fresh | Time | Speedup | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Z-Image | official | 25 | 6.20s | 2.24x | 29.25 | 0.9461 | 0.0366 |
| Z-Image | FlexCache | 25 | 6.17s | 2.26x | 29.25 | 0.9461 | 0.0366 |
| Z-Image | official | 17 | 5.09s | 2.73x | 24.90 | 0.8902 | 0.0734 |
| Z-Image | FlexCache | 17 | 4.94s | 2.83x | 24.90 | 0.8902 | 0.0734 |
| Z-Image | official | 13 | 3.33s | 4.17x | 19.76 | 0.7920 | 0.1678 |
| Z-Image | FlexCache | 13 | 3.30s | 4.23x | 19.76 | 0.7920 | 0.1678 |
| Qwen-Image | official | 25 | 8.09s | 2.37x | 31.15 | 0.9745 | 0.0340 |
| Qwen-Image | FlexCache | 25 | 7.94s | 2.07x | 31.15 | 0.9745 | 0.0340 |
| Qwen-Image | official | 17 | 6.39s | 3.01x | 30.10 | 0.9690 | 0.0331 |
| Qwen-Image | FlexCache | 17 | 6.22s | 2.65x | 30.11 | 0.9690 | 0.0330 |
| Qwen-Image | official | 10 | 3.34s | 5.75x | 27.43 | 0.9529 | 0.0473 |
| Qwen-Image | FlexCache | 10 | 3.48s | 4.73x | 27.44 | 0.9531 | 0.0471 |

Z-Image official and FlexCache images are pixel-identical at baseline and every
sweep point. Qwen-Image's EPAC and vanilla backends have a small baseline
numeric difference, but their cache-relative quality curves overlap.

The generated speed-quality plot, contact sheet, and raw CSV/JSON rows remain
local reproducible artifacts and are intentionally excluded from version
control. This table is the canonical committed result.

## Commands

Representative FlexCache commands:

```bash
# Z-Image; repeat with fresh steps 25, 17, and 13.
bash script/srun_direct.sh 1 1 chitu_diffusion/examples/zimage_epe.py \
  --model-path /path/to/Z-Image --output outputs/zimage-b25.png \
  --prompt "A red fox sitting in a snowy forest, cinematic photography" \
  --negative-prompt "" --height 512 --width 512 --steps 50 \
  --guidance-scale 4 --seed 42 --cache-strategy meancache \
  --meancache-fresh-steps 25

# Qwen-Image; repeat with fresh steps 25, 17, and 10.
bash script/srun_direct.sh 1 1 chitu_diffusion/examples/qwen_image_epac.py \
  --model-path /path/to/Qwen-Image --output outputs/qwen-b25.png \
  --prompt "A red fox sitting in a snowy forest, cinematic photography" \
  --negative-prompt " " --height 512 --width 512 --steps 50 \
  --true-cfg-scale 4 --seed 42 --cache-strategy meancache \
  --meancache-fresh-steps 25
```

Static CP2 used
`script/srun_direct.sh 1 2 ... --no-cfg-parallel`; Qwen CFP2 used the same
launcher with `--cfg-parallel`.

Reference runs loaded the local checkpoint once, replaced only the pipeline
`__call__` with the corresponding official `meancache_inference`, populated the
official `should_calc_list` and `edge_order` tables, and reset the CUDA generator
to seed 42 for each point. The Qwen reference needed a Diffusers 0.37
compatibility guard for a `None` all-valid attention mask; the Z-Image reference
needed missing `typing` names injected before import. Neither compatibility
guard changes denoising or MeanCache math.

