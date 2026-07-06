# Stage Profile Report — Non-Denoise Pipeline Stages (Text-Encode / VAE-Decode / Save)

**Goal.** Test the hypothesis that the *non-denoise* Z-Image pipeline stages are
**memory-bound** (so accumulating several requests and batching these stages before
dispatching to denoise would amortize weight reads + kernel launches and pay off).
This extends the M1 worker-roofline work (see
[`cost_model.py`](cost_model.py), [`profile_worker.py`](profile_worker.py),
[`m1_cost_model_report.md`](m1_cost_model_report.md)) from the denoise DiT step to
`encode_text`, `decode_latents`, and `save_output`.

**Status:** Complete. All numbers measured on a single **NVIDIA H20, bf16, Z-Image**.
**Artifacts (new files only):**
- Profiler: [`profile_stages.py`](profile_stages.py)
- Raw data: `stage_profile.json`（可再生成，已由 `.gitignore` 覆盖）

## TL;DR — hypothesis REJECTED (on H20)

The non-denoise GPU stages are **not memory-bound; they are compute-bound**, just like
denoise (M1's core H20 finding). Per-item latency barely drops with batch size:
**text-encode 1.20× (B1→B8), VAE-decode@1024² 1.11×, VAE-decode@512² 1.25×** — no
memory-bound plateau, no bandwidth saturation (text-encode moves weights at only
~160 GB/s, ~4% of H20's ~4 TB/s peak, while sustaining ~55–66% of BF16 compute peak).
Because the gain is small *and* the stages are a small slice of end-to-end time
(text+VAE = 9% at steps=4, ~2% at steps=20, <1% at steps=50), **accumulate-then-batch
text/VAE is a NO-GO throughput lever on H20**: expected end-to-end speedup is ~1.2% at
steps=4 and <0.3% at steps≥20. The only non-denoise stage worth optimizing is
**image save (~204 ms/image, CPU/disk)** — via **async overlap** with the next
request's denoise, not GPU batching.

## Method

Standalone profiler (no scheduler/generator, no `chitu_init`/distributed), mirroring
the M1 `profile_worker.py --mode compute` precedent so it runs on one debug GPU. It
loads the real Z-Image components once and times each stage directly with **CUDA
events, 3 warmup + 10 timed iters, median** (image-save uses `perf_counter`, being a
CPU/disk op). Each stage is a faithful copy of the adapter code path:

- **Text-encode** = `ZImageRuntimeAdapter.encode_text` → `ZImagePipeline._encode_prompt`
  (`z_image.py:435`): `apply_chat_template` → tokenize with
  `padding="max_length", max_length=512` → `Qwen3Model(...).hidden_states[-2]`.
  **Key:** every prompt is padded to **512 tokens**, so the encode forward is a
  fixed `[B, 512]` shape regardless of prompt length — a perfectly batchable op.
  Model is the **Qwen3 text encoder (~4.0B params, 8.04 GB bf16; 3.64B / 7.27 GB
  are transformer weights read every forward, the rest is the gather-only embedding).**
- **VAE-decode** = `ZImageRuntimeAdapter.decode_latents` (`z_image.py:1265`):
  `z = latents/scaling_factor + shift_factor`, then `AutoencoderKL.decode`. On a
  single GPU the runtime's `parallel_tiled_vae_decode` falls back to a plain
  `vae.decode` (world=1), which is what we time. VAE = Flux AutoencoderKL (0.17 GB;
  decoder 0.10 GB), 8× spatial, 16 latent channels.
- **Image-save** = `ZImageRuntimeAdapter.save_output` (`z_image.py:1315`):
  `image_processor.postprocess(..., "pil")` + `image.save(png, quality=95,
  subsampling=0)` to disk.

**VAE tiling control.** Serving does not call `enable_tiling`, so tiling is OFF. We
verified this doesn't matter at these sizes: an `--also-tiling` sweep produced
byte-identical timings (tiling only triggers above the VAE tile threshold, i.e. not
for ≤1024²). All VAE numbers below are single-pass decodes.

**Denoise reference** for the end-to-end share is taken from the M1 calibrated grid
([`cost_model.json`](cost_model.json)): **1024² B=1 = 596.86 ms / denoise step**
(seq_len 4608).

## Raw results — latency & per-item vs B

### 1) Text-encode (Qwen3, fixed seq=512, bf16)

| B | latency (ms) | per-item (ms) | per-item speedup vs B=1 |
|---:|---:|---:|---:|
| 1 | 45.97 | 45.97 | 1.00× |
| 2 | 82.42 | 41.21 | 1.12× |
| 4 | 155.56 | 38.89 | 1.18× |
| 8 | 305.40 | 38.17 | **1.20×** |

### 2) VAE-decode @ 1024² (latent 128×128, tiling off)

| B | latency (ms) | per-item (ms) | per-item speedup vs B=1 |
|---:|---:|---:|---:|
| 1 | 169.51 | 169.51 | 1.00× |
| 2 | 342.17 | 171.09 | 0.99× |
| 4 | 635.60 | 158.90 | 1.07× |
| 8 | 1225.23 | 153.15 | **1.11×** |

### 2b) VAE-decode @ 512² (latent 64×64, tiling off)

| B | latency (ms) | per-item (ms) | per-item speedup vs B=1 |
|---:|---:|---:|---:|
| 1 | 42.02 | 42.02 | 1.00× |
| 2 | 78.62 | 39.31 | 1.07× |
| 4 | 142.69 | 35.67 | 1.18× |
| 8 | 269.79 | 33.72 | **1.25×** |

### 3) Image save/output @ 1024² (per image, CPU/disk)

| stage | ms/image |
|---|---:|
| tensor→PIL postprocess (CPU) | 9.29 |
| PNG encode + disk write | 194.50 |
| **total** | **203.79** |

Not a GPU op and not shape-batchable — a fixed per-image CPU/disk cost. It is a
candidate for **async overlap** with denoise, not GPU batching.

## Memory-bound vs compute-bound verdict (per stage)

Memory-/launch-bound would show per-item latency **dropping sharply** with B until
bandwidth saturates; compute-bound shows **flat per-item / linear total** (like
denoise at 1024² in M1, which scaled 1.97×/3.9× at B2/B4).

| stage | per-item B1→B8 | total scaling | achieved BW / MFU | **verdict** |
|---|---:|---|---|---|
| text-encode (seq=512) | 1.20× | ~linear (B8 = 6.6× of B1) | ~160 GB/s (4% of ~4 TB/s peak); 81→98 TFLOP/s (**55→66% of ~148 TF peak**) | **compute-bound** (mild launch amortization only) |
| VAE-decode @1024² | 1.11× | ~linear (B8 = 7.2× of B1) | flat per-item | **compute-bound** |
| VAE-decode @512² | 1.25× | slightly sub-linear | mild launch amortization | **mostly compute-bound**, weak launch-bound tail |
| image-save | n/a (CPU/disk) | linear per image | — | **CPU/disk-bound** (overlap, don't batch) |

**Why compute-bound (not memory-bound) here:** H20 is a low-FLOP / very-high-BW part
(~148 TF BF16, ~4 TB/s HBM3). Reading the 7.27 GB text-encoder weights at peak BW
would take <2 ms, but the `[B,512]` forward takes 46 ms at B=1 — the matmuls, not the
weight reads, are the wall. Batching therefore only recovers the small fixed
launch/overhead tail (~1.1–1.25×), never the memory-amortization win the hypothesis
assumed. This is the exact same story M1 found for denoise on H20.

## End-to-end share across step counts (1024², B=1, CFG pos+neg)

Per-request non-denoise costs: text-encode pos+neg = 2×45.97 = **91.9 ms**, VAE
decode = **169.5 ms** (GPU non-denoise = 261.4 ms), image save = **203.8 ms** (CPU).
`prepare_denoise` is CPU-side latent RNG + scheduler setup (a few ms, scales with
n_sample) — negligible and not a GPU-batch target. Denoise = steps × 596.86 ms.

| steps | denoise (ms) | text+VAE share | save share | **non-denoise share (incl. save)** |
|---:|---:|---:|---:|---:|
| 4  | 2 387  | 9.2% | 7.1% | **16.3%** |
| 20 | 11 937 | 2.1% | 1.6% | **3.8%** |
| 50 | 29 843 | 0.9% | 0.7% | **1.5%** |

Denoise dominates (≥84% at steps=4, ≥98% at steps=50), so any non-denoise batching
benefit is bounded by these small shares.

## Expected benefit of accumulate-then-batch text+VAE (B=8)

Batching cuts per-request GPU non-denoise from 261.4 ms → (2×38.17 + 153.15) =
**229.5 ms**, i.e. a **12.2% cut of non-denoise GPU time** — but only:

| steps | end-to-end speedup from B=8 text+VAE batching |
|---:|---:|
| 4  | **~1.2%** (0.26% if save is the serial tail; 1.21% if save already overlapped) |
| 20 | **~0.26%** |
| 50 | **~0.11%** |
| 1 (distilled, best case) | ~3.0% (up to ~3.7% if save overlapped) |

Even the most favorable regime (a distilled **1-step** model, where text+VAE is ~25–30%
of the request) caps at **<4% end-to-end**, because the per-item batching gain itself
is only ~1.1–1.25× (compute-bound).

## Recommendation — GO / NO-GO

**NO-GO** for building an "accumulate-then-batch text-encode/VAE" path *as a throughput
lever on H20*. It fails on both axes the hypothesis required:
1. The stages are **compute-bound, not memory-bound** — batching yields only
   1.1–1.25× per item, not the near-free amortization batching gives to a
   memory-bound op.
2. Their **end-to-end share is small** at realistic step counts (≤9% text+VAE at
   steps=4, ~2% at steps=20), so even the 12% non-denoise cut is ~1% end-to-end at
   steps=4 and <0.3% at steps≥20 — not worth the accumulation latency, ragged-batch
   plumbing, and scheduler complexity.

**Instead, do this for non-denoise cost:**
- **Async image save (highest value).** Save is ~204 ms/image of CPU/disk that today
  sits on the critical path; overlapping it with the next request's denoise reclaims
  up to ~7% at steps=4 with zero batching and no numerical risk. This is the single
  best non-denoise win.
- **Keep VAE tile-parallel decode** (already in `parallel/vae.py`) for *latency* when
  spare GPUs are idle — it is orthogonal to batching.
- Text-encode/VAE batching is only worth revisiting under a **narrow regime**:
  distilled few-step (≤2) models **and** high arrival rate of **same-size** requests
  **and** a FLOP-rich GPU. Which leads to the caveat below.

## Caveats

- **Hardware-specific (H20).** H20's low FLOP : high BW ratio is what makes these
  stages compute-bound. On a FLOP-rich, lower-relative-BW GPU (e.g. H100 ~990 TF /
  3.35 TB/s, B200), the `[B,512]` text-encode forward would finish its matmuls far
  faster and could flip **memory-bound**, at which point batching would amortize the
  7.27 GB weight read and pay off much more. Re-profile with `profile_stages.py`
  before assuming this NO-GO transfers to other GPUs.
- **CFG assumption.** Share numbers assume text-encode runs pos+neg (guidance>1). With
  `cfg_size=2` the two encodes run on separate ranks in parallel, halving text's
  contribution and making the case for batching even weaker.
- **Fixed 512-token text seq.** Because the pipeline pads to `max_sequence_length=512`,
  text-encode cost is independent of prompt length; "small/complex prompts" do **not**
  change the batching calculus here.
- **Denoise reference reused from M1** (596.86 ms/step @1024²) rather than re-measured
  in this run; it is the same H20/bf16/flash_attn config.

## Reproduce

```bash
srun -p debug --gres=gpu:1 \
  /home/chenyy/WORK/cyy/ChituDiffusion/.venv/bin/python \
  experiments/cp_dp_hot_switch/profile_stages.py \
  --ckpt /home/chenyy/WORK/models/Z-Image \
  --out experiments/cp_dp_hot_switch/stage_profile.json --also-tiling
```
