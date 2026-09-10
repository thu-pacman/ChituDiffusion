# Static Layer-Wise VAE Decode

Validated 2026-09-10 against Diffusers 0.38.0 and Torch 2.10.0+cu130.
This replaces independent latent tiles in the static decode paths for Z-Image,
FLUX.1, LLaDA-Image, Wan, and Qwen-Image. Dynamic EPE is outside this work's
implementation and validation scope.

## Semantics and Communication

`chitu_diffusion/parallel/vae/exact.py` partitions latent height into disjoint
row bands. Uneven divisions keep their actual lengths throughout decoding.
The original module classes, parameters, and video cache accounting remain
intact. Dense decoding still uses the original forwards outside a parallel call.

| Operator | Distributed behavior |
| --- | --- |
| Spatial convolution | Exchange kernel-radius boundary rows with adjacent ranks using batched P2P. Zero padding appears only at the global image edges. Halo size follows each layer's kernel and dilation. |
| GroupNorm | Compute local central moments in FP32 (FP64 for FP64 input), gather only two scalars per batch/group/rank, and merge with actual shard-size weights. CUDA applies these statistics and channel affine with native inference BatchNorm, folding batch into channels; it neither computes batch statistics nor changes running state. |
| Attention | Keep Q local. Stack K/V into one all-gather, remove transport padding for uneven shards, and run SDPA against global K/V. No full attention-output replication. |
| Video temporal operations | Keep the entire temporal sequence on every spatial shard and preserve the original causal cache protocol. Channel RMSNorm stays local. |
| Output | Gather decoded rows only to the VAE leader. Workers do not retain the full output. |

GroupNorm statistics use the central-moment merge
`mean = sum(w_i * mean_i)` and
`var = sum(w_i * (var_i + (mean_i - mean)^2))`.
This avoids cancellation from subtracting two large squared quantities.
Only the small statistics tensors are gathered, not the feature maps.

Wan and Qwen also have spatial attention in their mid blocks. An empty
`attn_scales` list does not remove that attention. The need for global
communication is therefore not specific to Z-Image. Additional latent overlap
or boundary blending cannot reproduce these global operators.

Supported decoders are the Diffusers `Decoder`, `WanDecoder3d`, and
`QwenImageDecoder3d` structures used by the four tested VAE families below.
They require eval mode, centered odd-kernel convolutions with unit height
stride, and standard SDPA self-attention. Unsupported decoders/processors
warn and use dense leader decoding. Active Diffusers tiling is rejected because
it changes whole-image semantics. A single rank or shards smaller than the
required halo also use the dense leader path.

Z-Image and LLaDA retain their conservative serial defaults. Use
`--parallel-vae` for the static distributed path. The legacy
`--vae-parallel-halo` option is accepted but does not control per-layer halos.
Sidecars record `vae_decode_mode: layerwise`, the actual degree, and
`vae_parallel_halo: null`; requested options remain in `requested_*` fields.

## Correctness Tests

Model coverage at the time of these fixes:

| Model | Integration | Validation |
| --- | --- | --- |
| Z-Image, Wan | Layer-wise static decoding | Full pretrained VAE on two GPUs; Z-Image also has an end-to-end generation check |
| FLUX.1, LLaDA-Image, Qwen-Image | Layer-wise static decoding | Small models from the corresponding VAE families on CPU |
| FLUX.2-klein | Independent pipeline still uses original Diffusers decoding | AutoencoderKLFlux2 family support does not imply pipeline integration |
| Hunyuan Image 3, MiniMax-H3 video | Existing release tile distribution | Not adapted here; equivalence to non-tiled full-image decoding is not established |

MiniMax-H3 audio decoding remains on the leader. Only Z-Image and Wan weights
were available locally; no full-checkpoint GPU results are claimed for other models.

```bash
.venv/bin/python -m pytest -q -m 'not gpu and not distributed and not benchmark'
.venv/bin/python -m pytest -q tests/distributed/test_exact_vae.py -m gpu
```

The CPU regression suite passed **690 tests**, with 21 skips and 35 deselections
before the additional four-rank and GPU-only tests were added. CPU distributed tests compare
`AutoencoderKL`, `AutoencoderKLFlux2`, `AutoencoderKLWan`, and
`AutoencoderKLQwenImage` with dense decoding. Coverage includes even/uneven
height, batched images, multiple video frames with temporal upsampling, repeat
decodes, static noncontiguous subgroups, exception cleanup, changed attention
processors, disabled parallelism, and undersized inputs. Full decoded tensors
are compared, including boundary rows. Parameter keys and dense outputs are
also checked after adaptation.

The CUDA GroupNorm test passed on two PRO6000 GPUs. It covers FP64 with a large
mean offset, FP32, FP16, BF16, affine on/off, and batched image/video inputs.
Full pretrained GPU verification covers Z-Image and Wan; Flux2/Qwen family
coverage is through small CPU models, not their full GPU checkpoints.

The final focused run passed **10 tests**, with one GPU test deselected:

```bash
.venv/bin/python -m pytest -q tests/distributed/test_exact_vae.py \
  tests/distributed/test_zimage_vae_decode.py \
  tests/integration/test_chitu_diffusion_cli.py \
  tests/integration/test_public_imports.py -m 'not gpu'
```

This includes full two/four-rank groups and a two-rank subgroup on global ranks
1 and 3. The CUDA test was run separately and passed. Ruff and `git diff --check`
also passed.

## Pretrained GPU Measurements

Node `fuse3`, two RTX PRO 6000 Blackwell Server Edition GPUs. Identical seeded
random latents and weights are used for dense and parallel decode. Each path
has one warmup and five timed samples. Timings include VAE communication and
leader output gathering, exclude model loading and CPU transfers, and report
the median of the slowest rank per sample. Dense references run independently
on both GPUs. Memory is maximum additional allocated memory over all ranks,
including operator workspace, excluding already-resident weights/latents.

| VAE / output size | Precision | Dense ms | Native VAEP2 ms | Speedup | RMSE | Maximum error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Z-Image / 1024 x 1024 | BF16 | 105.97 | 57.11 | 1.86x | 0.002387 | 0.072266 |
| Z-Image / 1024 x 1024 | FP32, TF32 off | 425.28 | 227.69 | 1.87x | 6.95e-7 | 2.02e-5 |
| Wan / 17 x 480 x 832 | FP32, TF32 off | 2800.22 | 1484.23 | 1.89x | 5.43e-7 | 1.72e-5 |

Z-Image BF16 additional peak allocation falls from 2434 to 1475 MiB; Wan FP32
falls from 12723 to 8708 MiB. Z-Image IEEE FP32 instead uses 9764 / 9995 MiB
because convolution workspace depends on shape. Memory reduction is not
guaranteed for every precision and cuDNN algorithm.

The mean absolute jump across the split boundary is 0.0991406 / 0.0991638 for
dense / parallel Z-Image BF16, and 0.04014081 / 0.04014080 for Wan FP32.
These are raw decoder-value units, not 8-bit RGB units. Whole-tensor comparisons
are the correctness check; a small boundary metric alone is insufficient.

An initial FP32 run with TF32 convolution enabled had Z-Image RMSE 9.87e-5;
disabling TF32 reduced it below 7e-7. Production code does not change global
precision settings. The algorithm preserves full-image operator semantics,
but changed reduction order and convolution/SDPA kernels prevent a general
bitwise-equivalence guarantee, especially in BF16.

## DistVAE Comparison and Reproduction

Reference: [DistVAE](https://github.com/xdit-project/DistVAE), commit
`dce484185dc11d7406076117771e36acd796aa6a`, read under ignored `refs/DistVAE`.
The native implementation adds no DistVAE or custom-kernel build dependency.
In this revision DistVAE's 2D adapter runs the input convolution and mid block
replicated before sharding the upsampling stack. Its video attention adapter
gathers features and evaluates full attention on each rank. Our attention uses
local Q with gathered K/V in both families.

In the same Z-Image BF16 run, DistVAE measured **61.49 ms**, RMSE **0.002433**,
maximum error **0.0625**, and additional peak allocation **1475 MiB**. Native
VAEP measured **57.11 ms**. This is one configuration, not a general performance
ranking; no pretrained Wan DistVAE comparison was run.

Example launch from the repository root:

```bash
srun --immediate=30 -p Star -w fuse3 --nodes=1 --ntasks=1 \
  --gres=gpu:PRO6000:2 --cpus-per-task=16 --mem=100G --time=00:15:00 \
  --chdir=/home/chenyy20/ChituDiffusion \
  /usr/bin/env PYTHONPATH=/home/chenyy20/ChituDiffusion/refs/DistVAE:/home/chenyy20/ChituDiffusion \
  /home/chenyy20/ChituDiffusion/.venv/bin/torchrun --standalone --nproc-per-node=2 \
  tools/benchmarks/vae_parallel.py --family kl \
  --model-path /home/chenyy20/models/Z-Image --distvae \
  --output outputs/issue-validation/vae-kl-bf16-distvae.json
```

Omit `--distvae` and its `PYTHONPATH` entry for native-only verification.
Use `--dtype float32 --max-rmse 0.00002` for the FP32 check. TF32 is disabled
by default in the benchmark; `--allow-tf32` explicitly enables it. For Wan use
`--family wan --model-path /home/chenyy20/models/Wan2.1-T2V-1.3B-Diffusers
--height 480 --width 832 --frames 17 --dtype float32 --max-rmse 0.00002`.
Raw JSON reports are retained in ignored `outputs/issue-validation/`.

## Original Seam Reproduction

The original Z-Image CLI case also completed with `--parallel-vae`: two GPUs,
1024 x 1024, BF16, 50 steps, seed 7, prompt `a red cube on a white table`.
The output is `outputs/issue-validation/zimage-exact.png`; its sidecar confirms
`vae_decode_mode: layerwise`, `parallel_vae: true`, and `vae_parallel_degree: 2`.
Generation took 11.553 s, excluding model loading and image saving.

Using the original report's seam metric (mean absolute adjacent-row difference
of RGB channel means, in 0-255 units), the row 511/512 jump is **0.628255**.
The saved whole-image baseline is **0.628906** and the original broken tile
output was **4.705729**. The new image's strongest row transition is at row 399,
away from the split. Whole-image RGB RMSE against the saved dense baseline is
**0.360128**, with MAE **0.129658**. Visual inspection found no horizontal seam.
