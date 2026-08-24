---
name: model-adapter-chitudiffusion
description: Adapt a new image diffusion model into the Diffusers-native ChituDiffusion runtime. Use when the user wants to inspect upstream code, run the original demo, add a model executor/pipeline integration, wire attention and CFG/context parallelism, adapt FlexCache, benchmark the result, clean artifacts, and split categorized commits.
---

# ChituDiffusion Model Adapter

## Operating Rules

- Work in the current ChituDiffusion checkout. Use `./.venv/bin/python` and the
  real `chitu generate`/`tools/cluster/srun_direct.sh` path for acceptance.
- Put upstream/reference code under `refs/` and model weights under `~/WORK/models` unless the user gives another location.
- Prefer direct integration with existing runtime, adapter, scheduler, parallel, and FlexCache APIs. Do not add compatibility shims for obsolete paths unless the user asks.
- Treat generated outputs as evidence, not code. Commit source changes,
  reusable benchmark scripts, and final Markdown summaries. Keep plots/media
  ignored unless the repository explicitly curates them; delete smoke/debug
  runs and local-only symlinks before committing.
- Keep user updates short while running long Slurm or benchmark jobs, and reuse completed results where possible.

## Workflow

1. **Reference intake**
   - Identify the official repo, license, model card, demo command, required weights, and expected image size/step defaults.
   - Clone or copy the upstream implementation into `refs/<model-name>/`.
   - Download weights to `~/WORK/models/<model-name>/` or verify existing weights.
   - Run the upstream demo first in the current virtualenv. If GPU resources are needed, use `srun`/`tools/cluster/srun_direct.sh`; do not fall back to CPU for diffusion demo validation.

2. **Gap report before heavy adaptation**
   - Read the upstream pipeline, transformer, scheduler, VAE/image processor, prompt encoder, and attention code.
   - Classify code into:
     - directly covered by ChituDiffusion APIs,
     - needs a runtime adapter or stage-level wrapper,
     - needs custom attention/parallel logic,
     - needs scheduler or latent-shape handling,
     - benchmark-only glue.
   - Give the user a short plan and estimate of the non-wrapper code surface before large edits.

3. **Runtime adapter**
   - Add a model package under `chitu_diffusion/models/<family>/`, following the
     existing API/executor/pipeline/transformer split.
   - Implement the `DiffusionBackend` executor operations for loading,
     request preparation, one denoise step, decoding, and output handling.
   - Normalize model-specific latent/token shapes at executor boundaries.
     Prefer one authoritative shape variable instead of duplicate
     `image_size`/`img_shape` state.
   - Register a `chitu generate --model ...` module in
     `chitu_diffusion/cli.py`. Add `serve` support only when the model satisfies
     the persistent EPE backend contract.

4. **Attention and parallelism**
   - Wire attention through the model package's attention processor and the
     shared `chitu_diffusion.parallel` topology/NCCL CP utilities.
   - For joint text/image attention, verify whether text tokens need sharding. If encoder states are short, prefer keeping text replicated and sharding image tokens only.
   - Bring up CFG parallel first when CFG semantics exist. Then bring up NCCL context parallel. Only implement a Ring composition if it fits the model sequence semantics without a large separate attention implementation.
   - Validate 1/2/4/8 GPU cases with minimal prompts before full benchmarks.

5. **FlexCache and acceleration**
   - Start with strategies whose model contracts can be represented by the
     existing model/block/leaf hooks.
   - Isolate each algorithm in `chitu_diffusion/flexcache/strategies/`; add
     reusable model sites or probes only through `flexcache/spec.py`.
   - Add immutable parameters to `flexcache/config.py`, register construction in
     `flexcache/strategies/factory.py`, and expose example CLI arguments in
     `chitu_diffusion/commands/cache_args.py`.
   - Keep fresh/reuse decisions rank-identical under CP and CFP. Reject a
     parallel mode when the decision depends on unsynchronized local shards.
   - Test at least one end-to-end image before sweeping.

6. **Benchmark and visualization**
   - Use the model's root `examples/` entry point and keep a concise
     public summary in `docs/features/flexcache.md`. Keep generated media,
     per-run notes, and raw measurements under the ignored `outputs/` directory.
   - Reuse previous runs by collecting summaries into a consolidated result directory instead of re-running model loads unnecessarily.
   - For FlexCache trade-off plots and visual sheets, use the `flexcache-bench-visualizer` skill.
   - Record commands, backend, parallel topology, speed, quality, and known
     limitations in the result Markdown.

7. **Cleanup and commits**
   - Run `git diff --check` and a targeted `py_compile`/`compileall` on touched Python files.
   - Delete smoke/debug result directories, local-only symlinks, stale plots, and generated bytecode.
   - Split commits by domain. A typical split is:
     - model/runtime/FlexCache adaptation,
     - benchmark scripts and recorded results,
     - docs or skill updates if they live inside the repo.
   - Report commit hashes and validation commands.

## Acceptance Checklist

- Upstream demo ran successfully on GPU.
- ChituDiffusion generated at least one end-to-end image through the real launcher.
- Attention backend and parallel cases requested by the user were either validated or explicitly scoped out with a concrete reason.
- FlexCache strategies requested by the user have at least one correctness run and benchmark evidence.
- `result.md` and final plots match the measured data.
- Worktree is clean after categorized commits, except intentionally untracked external artifacts outside the repo.
