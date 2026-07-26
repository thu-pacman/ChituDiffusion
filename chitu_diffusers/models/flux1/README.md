# Flux.1 EPAC Adaptation Report

Date: 2026-07-23

Status: first AGKV implementation complete and validated on 1/2/4 H20 GPUs.

## First Integration Scope

The first integration targets `FLUX.1-dev` text-to-image generation through the
Diffusers-native `chitu_diffusers` stack. It supports:

- `FluxPipeline` component loading from a local Diffusers checkpoint;
- request-local CLIP + T5 encoding, packed latents, scheduler, and denoise state;
- one resumable denoise step through `DiffusersModelAdapter`;
- offline generation and the generic embedded EPAC runtime;
- dynamic lane widths with AGKV;
- leader-side image postprocessing and lane-parallel tiled VAE decode.

The first correctness milestone is deliberately narrower than the full upstream
pipeline. It excludes true CFG, ControlNet, IP-Adapter, LoRA, batching, Schnell,
USP, and FlexCache. These remain follow-up work.

## Frozen References

The local reference sources are ignored by Git:

| Reference | Location | Revision |
| --- | --- | --- |
| BFL Flux | `refs/flux1-upstream` | `802fb4713906133fcbd0d8dc5351620ca4773036` |
| Hugging Face Diffusers | `refs/diffusers-v0.37.0` | tag `v0.37.0`, commit `d791c5c024014784df4a8dac2601e19fb4d300fc` |
| Local weights | `/home/chenyy/WORK/models/Flux-1` | Diffusers `FluxPipeline` layout |

The checkpoint is FLUX.1-dev and is covered by the FLUX.1-dev non-commercial
license in its local `LICENSE.md`. The VAE has separate Apache-2.0 terms.

This integration uses the user-designated sibling checkout environment at
`/home/chenyy/WORK/cyy/ChituDiffusion/.venv`, which resolves
Diffusers `0.37.0`, Torch `2.9.1+cu130`, and Transformers `4.56.2`. The runtime
contract and dependency declarations target `0.37.0`.

## Existing Code To Reuse

The model-independent EPAC implementation already covers request admission,
pulse scheduling, dynamic lane leases, state migration, cancellation, measured
costs, startup warmup, terminal scheduling, and embedded runtime lifecycle.
Flux.1 should plug into these boundaries rather than add another runtime.

| Flux.1 concern | Existing source | Intended use |
| --- | --- | --- |
| Request-local Flux tensors and scheduler | `chitu_diffusion/runtime/adapter/flux1.py` | Behavioral reference, not a dependency |
| Packed latent helpers | `chitu_diffusion/modules/utils/flux` | Reuse or replace with upstream Diffusers helpers |
| Flux block and weight layout | `chitu_diffusion/models/model_flux1.py` | Parallel-forward reference only |
| Joint attention projections | `chitu_diffusion/modules/attention/flux_attention.py` | Projection/RoPE reference |
| Dynamic topology and groups | `chitu_diffusers/parallel/groups.py` | Reuse directly |
| Replicated-text, sharded-image attention | `chitu_diffusers/parallel/image_attention.py` | Reuse directly for AGKV/USP |
| Runtime adapter contract | `chitu_diffusers/epac/adapters/base.py` | Implement `Flux1ModelAdapter` |
| Embedded executor lifecycle | `chitu_diffusers/epac/model_executor.py` | Subclass and implement Flux hooks |
| Measured scheduler facade | `chitu_diffusers/epac/model_scheduling.py` | Reuse directly |
| Synchronous pipeline facade | `chitu_diffusers/epac/api.py` | Declare Flux pipeline and adapter |
| End-to-end model pattern | `chitu_diffusers/models/zimage` | Mirror ownership and lifecycle, not tensor semantics |

No change is required in the EPAC planner or lane broker for the first Flux.1
integration. The generic cost model already keys on image tokens, lane width,
batch size, condition count, terminal cost, and state bytes.

## Model-Specific State

One `Flux1DenoiseState` should be authoritative for all request-local mutable
state:

- packed `latents` in `[batch, image_tokens, 64]` layout;
- `prompt_embeds` from T5 and `pooled_prompt_embeds` from CLIP;
- replicated `text_ids` and full `latent_image_ids`;
- a cloned `FlowMatchEulerDiscreteScheduler` and its `timesteps`;
- the distilled-guidance vector when `guidance_embeds` is enabled;
- `step_index`, requested width/height, and image token count.

For FLUX.1-dev, `guidance_scale` is a model input for guidance distillation. It
does not imply two-branch CFG. The initial request profile therefore uses one
condition. True CFG must be represented by a separate request option later and
must not be inferred from `guidance_scale > 1`.

The canonical migratable state is the full packed latent tensor plus
`step_index`. Text embeddings and position IDs are immutable replicated request
data. State byte accounting must use the actual latent dtype; the current
runtime reference keeps scheduler latents in float32.

## Pipeline And Adapter Mapping

The implementation should add the following model-owned files:

```text
chitu_diffusers/models/flux1/
  __init__.py
  api.py          Flux1EPACPipeline and Flux1Request
  adapter.py      split Diffusers denoise-step adapter
  attention.py    topology-aware Flux attention processor
  executor.py     embedded runtime factory/executor
  pipeline.py     request preparation, one step, decode, warmup
  transformer.py  local-image/replicated-text dynamic CP forward
```

The upstream `FluxPipeline.__call__` maps to EPAC as follows:

| Upstream operation | EPAC owner |
| --- | --- |
| validate inputs, encode CLIP/T5 prompts | `prepare_request` |
| pack noise latents and build image/text IDs | `prepare_request` |
| clone scheduler and retrieve shifted timesteps | `prepare_request` |
| build timestep and distilled-guidance inputs | `prepare_step` |
| call `FluxTransformer2DModel` | `model_forward` |
| optional true-CFG combination | `process_model_output`, deferred |
| `scheduler.step` and advance cursor | `scheduler_step` |
| unpack packed latents and apply VAE scale/shift | `finalize_gpu` |
| image processor conversion | asynchronous `postprocess` |

The public facade should follow the Z-Image split: a lightweight pipeline API
for offline use and a factory usable by `EmbeddedDiffusionRuntime.create()`.
The HTTP server is currently Z-Image-specific and is not part of the first
Flux.1 milestone.

## Dynamic Context Parallel Design

Flux has two attention phases:

1. Double-stream blocks update replicated text tokens and sharded image tokens.
2. Single-stream blocks concatenate text and image tokens internally.

At every forward, the transformer must read `parallel_context.active`; it must
not cache a lane width or process group at construction. Image tokens and image
RoPE IDs are sliced by `rank_in_lane`. Text tokens and text IDs remain
replicated. `ImageContextParallelAttention` already implements the required
attention contract for both AGKV and USP: local image queries attend to gathered
image K/V plus replicated text K/V, while replicated text queries receive a
consistent full-context result.

The custom transformer boundary is still required because an attention
processor alone cannot safely establish and preserve the local-image versus
replicated-text layout across all double-stream and single-stream blocks. The
final image projection is local; gather the image sequence once at transformer
output so the scheduler retains one canonical packed latent tensor on every
lane rank.

Initial constraints:

- image token count must divide the active lane width;
- batch size is one;
- no attention masks, ControlNet residuals, IP-Adapter inputs, or true CFG;
- AGKV is the bring-up backend; USP follows after 1/2/4-rank AGKV parity;
- cp1 must call the native Diffusers forward to provide a clean parity oracle.

## VAE And Terminal Work

Before VAE decode, packed latents must be unpacked using the Diffusers Flux
layout and transformed with the checkpoint's `scaling_factor` and
`shift_factor`. The existing lane-parallel VAE helper can split the spatial
latent height and gather decoded pixels on the lane leader. Flux VAE halo size
and pixel parity must be measured independently; Z-Image's halo value is not an
assumed default for Flux.

The embedded executor enables parallel VAE by default and accepts
`parallel_vae=False` plus a configurable `vae_parallel_halo`. Packed Flux
latents are unpacked before tiling; the tile gather uses the AutoencoderKL's
actual 8x spatial scale rather than Flux's 16x packed-latent scale. Startup
terminal warmup measures the slowest VAE rank in each lane plus leader D2H.

## Implemented Code Surface

The non-wrapper surface is the Flux dynamic-CP transformer, attention processor,
request-state preparation, and Flux-specific warmup/decode behavior. Measured
scheduling, executor lifecycle, stage-world setup, request validation/profile,
state transfer, and synchronous API execution are shared under
`chitu_diffusers.epac`. The scheduler and worker pool remain model-independent.

## Bring-Up And Acceptance Order

1. Run the native Diffusers `0.37.0` `FluxPipeline` baseline with local weights
   on one GPU through `script/srun_direct.sh`.
2. Add request-state and split-step unit tests using small fake components.
3. Run the EPAC adapter on one GPU with native transformer forward and compare
   final packed latents and decoded image against the native pipeline for the
   same prompt, seed, dimensions, steps, and guidance.
4. Add AGKV transformer execution and validate fixed cp2 and cp4 against cp1.
5. Validate dynamic phase changes such as cp1 -> cp4 -> cp2 without stale
   collectives or scheduler cursor drift.
6. Add startup DiT/VAE warmup and run one embedded-runtime request through the
   real launcher.
7. Add USP and only then evaluate FlexCache strategies already supported by the
   older Flux.1 runtime.

The minimum GPU acceptance matrix is 512 and 1024 square images, 1/2/4 ranks,
one-step smoke plus one complete 50-step image, fixed seed, and a native
Diffusers reference. The final handoff must include the exact commands, output
paths, latency rows, and image/latent comparison results.

## Validation Snapshot

- Native Diffusers job `197890`: 512 square, 2 steps, seed 7.
- EPAC cp1/cp2/cp4 jobs `197908`, `197919`, and `197924` produced byte-identical
  PNG files to the native baseline.
- Dynamic job `197925` ran lane widths `1 -> 4 -> 2`; the maximum latent
  difference across ranks after every step was `0.0`.
- Embedded job `197933` completed startup warmup, request scheduling, denoise,
  VAE, D2H, asynchronous postprocess, and shutdown. Its output was pixel-identical
  to the native two-step baseline.
- Native/cp4 jobs `197938` and `197939` validated 1024 square at one step. The
  cp4 output had maximum pixel error `2/255`, mean absolute error `0.0587`, and
  PSNR `60.45 dB` relative to native Diffusers.
- Four-GPU strategy jobs `198078`, `198080`, and `198081` replayed the same
  closed batch of eight alternating 512/1024 requests at four steps. Model load
  and startup warmup were excluded from request makespan.

  | Strategy | Makespan | Throughput | Mean latency | P95 latency |
  | --- | ---: | ---: | ---: | ---: |
  | static-DP | 7.525 s | 1.063 req/s | 4.107 s | 6.790 s |
  | static-CP | 8.600 s | 0.930 req/s | 4.879 s | 7.929 s |
  | EPE elastic | 8.248 s | 0.970 req/s | 5.242 s | 8.220 s |

  EPE selected four singleton lanes for all 32 denoise steps, so this saturated
  workload exercised elastic pulse coordination but did not trigger a resize.
  Its outputs were pixel-identical to static-DP. Static-CP was pixel-identical
  at 512; at 1024 its four-step outputs were 51.4-54.1 dB PSNR from static-DP,
  with maximum pixel differences of 16-26/255. The result is a short smoke
  benchmark, not a steady-state or complete-quality acceptance run.
- Four-GPU jobs `198092`, `198094`, and `198096` replayed the default Z-Image
  `mixed_512_1024_2048_r3n12_12step.json` trace on the same H20 node, overriding
  all 12 requests to 50 steps and using a five-step elastic pulse. The Poisson
  trace submits four requests at each of 512, 1024, and 2048 within 3.77 s.

  | Strategy | Makespan | Throughput | Mean latency | P95 latency |
  | --- | ---: | ---: | ---: | ---: |
  | static-DP | 235.823 s | 0.05089 req/s | 128.405 s | **219.964 s** |
  | static-CP | 289.091 s | 0.04151 req/s | 179.716 s | 278.012 s |
  | EPE elastic | **232.271 s** | **0.05166 req/s** | **104.845 s** | 229.204 s |

  EPE improves throughput by 1.53%, makespan by 1.51%, and mean latency by
  18.35% over static-DP, while its P95 latency is 4.20% worse. It schedules 589
  of 600 denoise steps at cp1, four at cp2, and seven at cp4. All 12 EPE pixel
  hashes are identical to static-DP. This saturated result meets the basic
  no-regression requirement for throughput, but the tail-latency regression and
  pulse waiting remain optimization targets. Raw metrics and rank timelines are
  under `outputs/flux1-benchmark/20260723-mixed50/`.
- Four-GPU jobs `198086`, `198088`, and `198090` replayed the canonical Z-Image
  `phased_dp_cp_slo_n15_12step.json` arrival trace with every request overridden
  to 50 denoise steps and a five-step elastic pulse. The 15 requests retain the
  original 0-130 s arrivals and 512/1024/2048 resolutions.

  | Strategy | Makespan | Throughput | Mean latency | P95 latency |
  | --- | ---: | ---: | ---: | ---: |
  | static-DP | 266.325 s | 0.0563 req/s | 81.487 s | 184.446 s |
  | static-CP | 307.962 s | 0.0487 req/s | 115.517 s | 203.672 s |
  | EPE elastic | **247.624 s** | **0.0606 req/s** | **80.328 s** | 188.993 s |

  EPE improves throughput by 7.55% and makespan by 7.02% over static-DP, while
  improving throughput by 24.37% over static-CP. Its P95 latency is still 2.47%
  above static-DP. Across 750 denoise steps, EPE scheduled 602 at cp1, 86 at cp2,
  and 62 at cp4, so this run exercises real resizing rather than only the elastic
  control path. Fourteen of 15 EPE pixel hashes match static-DP; `req0000` starts
  with five cp4 steps before switching to cp1 and consequently differs. The
  trace's single 22.5 s deadline was tuned for Z-Image at 12 steps and is
  infeasible for this 50-step Flux.1 run, so none of the strategies met it.
  Raw metrics and rank timelines are under
  `outputs/flux1-benchmark/20260723-phased50/`.
- Four-GPU jobs `198784`, `198786`, `198789`, and `198790` validated Flux VAE
  tiling against leader-only decode on the same H20 node. At 512 square, cp4
  VAE latency falls from 42.60 ms to 22.10 ms (1.93x); the halo-8 output is
  38.48 dB PSNR / 0.9942 SSIM from leader-only. At 2048 square, cp4 VAE latency
  falls from 770.41 ms to 214.44 ms (3.59x), with 44.92 dB PSNR / 0.9975 SSIM
  and no visible tile seam. One-request end-to-end latency is not reported as a
  speedup because asynchronous CPU postprocess cold-start varied by about 1.9 s.
  Raw warmup rows, timelines, and PNGs are under
  `outputs/flux1-benchmark/20260724-parallel-vae/`.
- Four-GPU job `199784` validated the warmup-adaptive three-arm runner at 50
  steps with three requests in each of the sparse, crossover, and overload
  phases. Parallel-VAE warmup estimated equal-mix capacities of 0.05653 req/s
  for static-DP and 0.04216 req/s for serial static-CP. The generated offered
  rates were 0.02108, 0.04882, and 0.06784 req/s; every request used the same
  203.45 s deadline, equal to 1.2x the longest measured cp1 request latency.

  | Strategy | Makespan | Throughput | Mean latency | P95 latency | Mean queue | SLO |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | static-DP | 437.407 s | 0.02058 req/s | 71.736 s | 171.618 s | 0.008 s | 9/9 |
  | static-CP | 324.559 s | 0.02773 req/s | 35.472 s | 55.920 s | 10.481 s | 9/9 |
  | EPE elastic | 323.384 s | 0.02783 req/s | 28.803 s | 62.720 s | 1.117 s | 9/9 |

  EPE matches static-CP throughput on this arrival-spaced trace while reducing
  mean latency by 18.80% and mean queue delay by 89.34%; its P95 latency is
  12.16% higher because the crossover 2048 request gives up GPUs to later
  arrivals. All SLOs pass. Aggregate throughput includes intentional sparse
  gaps, so sustained capacity is still represented by the warmup estimates and
  the earlier dense trace, not by 9 / makespan. Raw results and the shared
  timeline are under `outputs/flux1-benchmark/20260726-adaptive-3arm-r3-v2/`.
- Targeted CPU tests pass under the designated sibling environment.

Remaining validation and release work:

- validate USP before exposing it as accepted rather than experimental;
- adapt and benchmark FlexCache strategies;
