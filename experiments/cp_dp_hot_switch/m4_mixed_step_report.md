# M4 Mixed-Step Continuous Batching — Root-Cause Report

**Verdict up front:** Z-Image **does** support interleaved / mixed-step batching. The
per-item-timestep math is **bitwise correct** (a row's output is provably
independent of the timesteps of the other rows sharing the forward). The only
divergence between a batched run and an independent (solo) run is **floating-point
batch-size non-determinism in the batched GEMM/attention kernels** — a
**pre-existing property of Z-Image batched inference** (it reproduces with M4
completely OFF, via plain `n_sample`), **not** an M4 bug and **not** an
architectural row-coupling.

Classification: **Category A** (fp reduction-order / kernel batch-nondeterminism).
Not Category B (no shared-scheduler/timestep-broadcast/batch-leak bug — all
disproven). Not Category C (no architectural coupling — disproven by
content-independence and reproduction in the reference path).

Launcher for every GPU run below (worktree code, slurm `debug`):
```
cd /home/chenyy/WORK/cyy/ChituDiffusion-hotswitch && \
CHITU_PYTHON_BIN=/home/chenyy/WORK/cyy/ChituDiffusion/.venv/bin/python \
/home/chenyy/WORK/cyy/ChituDiffusion/.venv/bin/python -m chitu_diffusion.cli run <config> ...
```
(confirmed the printed `script/srun_direct.sh` path is the worktree's.)

---

## 1. Attention-backend isolation (flash_attn vs torch_sdpa)

Single batched forward, **fixed membership {A,B}**, A@step 2, B varied 2→6.
`test/test_z_image_mixed_forward.py` + `test/configs/z_image_mixed_forward.yaml`.

| quantity | flash_attn (bf16) | torch_sdpa (bf16) | torch_sdpa (fp32) |
|---|---|---|---|
| A invariant to B's timestep (fixed membership) | bitwise 0.0 | **bitwise 0.0** | **bitwise 0.0** |
| batch-size probe, UNIFORM (A vs [A,A-copy], initial latents) | 0.50 (rms 0.057) | 0.50 (rms 0.057) | **3.36e-5 (rms 5.9e-6)** |
| batch-size probe, RAGGED (A vs [A,B], diff prompt, initial latents) | 0.50 (rms 0.057) | 0.50 (rms 0.057) | **3.36e-5** |

Reading:
- **A is bitwise-invariant to B's timestep** on both backends → the mixed-step
  unlock is exact; no timestep leakage across rows.
- The batched-vs-solo diff is **identical for flash and torch_sdpa** in bf16 →
  the residual is not a flash-specific tiling artifact; it is generic fp
  reduction-order.
- In **fp32 it collapses ~4 orders of magnitude** (0.057 → 5.9e-6) for initial
  latents → the bf16 component is unambiguously reduction-order (Category A).
- RAGGED == UNIFORM to 4 sig figs → **content-independent** (a different-prompt B
  and an identical copy of A perturb A's row equally) → **not** attention
  padding/mask, **not** joint attention.

torch_sdpa is used for all correctness gates below (deterministic reference);
flash_attn remains the default for throughput runs.

## 2. Bisecting the forward

Signed-mean test (torch_sdpa, single forward): `signed_mean/abs_mean = 0.11`
(near zero-mean) → random-signed perturbation, the signature of reduction-order
noise, not a systematic bias.

Per-op localization (a) AdaLN/timestep: `adaln_input = t_embedder(t)` is built
from the **per-row** timestep vector (`transformer_z_image.py:914`; our mixed
forward passes each row its own `current_step` timestep) — confirmed by the
bitwise B-step-invariance. (b) RoPE: per-row `freqs_cis`. (c) FlowMatch scheduler:
`denoise_step_group_once` sets `pipe.scheduler._step_index = step_index` **per
task per call** → stateless w.r.t. group membership (confirmed: batch-1 steps are
bitwise, see §3). (d) CFG row layout: cond rows then uncond rows, per-task offset
slicing — verified correct. (e) batch-dim reduction/normalization: **none** —
proven by content-independence (a batch-mean/var would make [A,B] differ from
[A,A-copy]; they are identical).

### The one subtlety: evolved (mid-trajectory) latents

Holding A's **input latent byte-identical** (diff = 0.0), at a mid-trajectory step
(step 3 of 8) the batch-size perturbation is **larger and persists in strict fp32**:

| measurement (fp32, torch_sdpa, A@step3) | max | rms |
|---|---|---|
| A: solo vs [A, A-copy]  (identical B content) | 1.83 | 0.159 |
| A: solo vs [A, B@same-step] (batch-size) | 1.83 | 0.159 |
| A: [A,B@same] vs [A,B@0]  (**B-step leak into A**) | **0.0** | **0.0** |
| same, TF32 disabled (strict IEEE fp32) | 1.83 | 0.159 |

So the residual is: **content-independent, B-step-independent (bitwise), and not
TF32** — it is the batched GEMM/attention kernel simply not being bit-exact across
batch sizes, whose *magnitude* scales with the (ill-conditioned) intermediate
latent distribution. It is **not** a coupling bug (a bug would depend on the
co-batched row's content or step; this depends on neither).

### It is pre-existing (reproduces with M4 OFF)

Plain single-task path, **continuous batching OFF**, fp32, `n_sample=2` vs
`n_sample=1` (same prompt, same seed), row 0:

```
max=4.46  mean=0.166  rms=0.287  bitwise=False
```

This is the stock diffusers Z-Image batched forward with no M4 involvement →
**the batch-size numerics are a property of the model, not of continuous
batching.** M4 faithfully reproduces base-model batching behavior.

## 3. Trajectory: where divergence enters and why it grows

Driving A solo vs A batched-with-B after 3 steps, comparing A's latent each step
(bf16 and fp32 give the **same** trajectory):

```
step 0 (A_mix bs=1): 0.0        (bitwise)
step 1 (bs=1):       0.0
step 2 (bs=1):       0.0
step 3 (bs->2):      max 0.129  mean 6.0e-3   <- batch size changes here
step 4 (bs=2):       max 0.274  mean 1.3e-2
step 5 (bs=2):       max 0.724  mean 3.6e-2
step 6 (bs=2):       max 3.64   mean 0.130
```

Divergence is **exactly 0.0 while A stays at batch-1**, enters the moment A's
batch size changes, and then **amplifies** ~2–5×/step. The amplification is
intrinsic to few-step diffusion sampling (small per-step perturbations are
magnified by the sampler), which is the same reason batched `n_sample`
inference yields visibly different-but-valid images than solo.

## 4. Correctness gates

**Gate (a) — baseline integrity (regression).** OFF vs `continuous_batch=true`
with a **single** request, flash_attn bf16, 8 steps:
```
max_abs_diff = 0.000e+00   bitwise = True   PASS
```

**Gate (b) — mixed-step correctness.** A late-joining request (B) shares forwards
with an in-flight one (A) at different steps.
- *Forward-level (the M4 invariant), torch_sdpa:* row A's noise prediction is
  **bitwise invariant** to B's timestep (max_abs_diff = 0.0, bf16 and fp32).
  A row's per-step math depends only on its own latent+timestep+conditioning. **PASS.**
- *End-to-end final latent vs solo* (stagger=3, 8 steps): bf16 seed42 5.68 /
  seed43 0.53; fp32 seed42 5.08 / seed43 3.06 — **not** within 1e-5. This is the
  pre-existing batch-size fp property of §2 amplified over steps (§3), **not**
  an M4 defect: it reproduces identically in the plain `n_sample` reference with
  M4 off. A batched result is an equally-valid sample, exactly as with any
  batched diffusion inference.

**Conclusion on feasibility:** mixed-step batching is mathematically sound and
correctly implemented. Bitwise equivalence to a solo run at the *final latent* is
not attainable on Z-Image because its batched kernels are not bit-exact across
batch sizes — a base-model property shared by all batched inference (M3, plain
`n_sample`), independent of M4.

## 5. Gate (c) — staggered serve throughput / latency (OFF vs ON)

Poisson trace via `experiments/cp_dp_hot_switch/gen_client_trace.py`, server
`test/test_z_image_serve.py`, flash_attn (default), world_size 1, 20 steps.

### 1024×1024, rate 3 req/s, 12 requests (compute-bound)

| metric | OFF | ON | Δ |
|---|---|---|---|
| throughput (req/s) | 0.0479 | 0.0493 | +2.9% |
| makespan (s) | 250.4 | 243.4 | −2.8% |
| mean latency (s) | 134.6 | 186.5 | **+38%** (worse) |
| p50 latency (s) | 134.7 | 161.4 | +20% (worse) |
| p95 latency (s) | 235.6 | 240.2 | +2% (worse) |
| mean queue / admission delay (s) | 9.52 | **3.07** | **−68%** (better) |
| p95 queue delay (s) | 19.62 | 6.38 | −67% (better) |

### 512×512, rate 6 req/s, 16 requests (some headroom)

| metric | OFF | ON | Δ |
|---|---|---|---|
| throughput (req/s) | 0.180 | 0.190 | +5.2% |
| makespan (s) | 88.7 | 84.4 | −4.9% |
| mean latency (s) | 46.4 | 62.2 | **+34%** (worse) |
| p95 latency (s) | 82.6 | 82.4 | ~flat |
| mean queue / admission delay (s) | 2.73 | **2.14** | −22% (better) |

### Where the benefit is (and isn't)

- **Queue / admission delay drops sharply** (1024²: −68%). This is M4's real,
  measured win: the engine is now **step-granular**, so an arrival is admitted at
  the next denoise **step** boundary instead of waiting for the whole in-flight
  request to finish. Requests enter mid-denoise.
- **Throughput is ~flat** (+3–5%): both resolutions are compute-bound on this
  H20, so same-shape batching multiplies FLOPs/step and conserves total work
  (expected, reported honestly).
- **Mean/p95 latency gets worse** under this naive FIFO same-shape coupling:
  requests that would finish quickly solo are coupled to the whole batch and
  finish later. On compute-bound single-GPU this is the honest cost of coupling.

The value of M4 here is **step-granular admission/fairness** and the enabling of
later levers (retire-aware / SLA scheduling, dynamic SP, FlexCache), not raw
throughput on a compute-bound single GPU — consistent with the stated M4 goal.

## 6. Files touched (uncommitted)

- `chitu_diffusion/runtime/adapter/z_image.py` — mixed-step batched forwards
  (`_transformer_forward_*_group_mixed`, `_z_guided_noise_pred_batch_mixed`,
  `denoise_step_group_once`).
- `chitu_diffusion/runtime/generator.py`, `runtime/main.py`, `runtime/scheduler.py`
  — M4 step-at-a-time engine (admit/step/retire), admission planner.
- `test/test_z_image_mixed_forward.py` — root-cause bisection harness (isolation,
  batch-size uniform/ragged, signed-mean, fp32, TF32, trajectory).
- `test/configs/z_image_mixed_forward.yaml` (torch_sdpa),
  `z_image_serve_{1024,512}_{off,on}.yaml`, `z_image_smoke_{sdpa,cbon}.yaml`.
- `experiments/cp_dp_hot_switch/traces/serve/poisson_1024_r3n12.json`,
  `poisson_512_r6n16.json`.
