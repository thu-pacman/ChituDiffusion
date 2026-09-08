# LLaDA-Image PR #2 Review and Acceptance

Date: 2026-09-08.

## Revisions

- PR: <https://github.com/thu-pacman/ChituDiffusion/pull/2>
- Reviewed PR head: `8878340`.
- Integration base: `b2ca508`.
- Integration branch: `codex/review-pr2-llada-image`.
- Official reference: <https://github.com/inclusionAI/LLaDA-Image>,
  commit `e7c861b0aaa00d2f7ed49600a3a6f170e02a9d59`.
- BF16 checkpoint: <https://huggingface.co/inclusionAI/LLaDA-Image>,
  revision `e4e2703f410f7ddb6ee8d6b09dac6a8ec5093039`.
- Local checkpoint: `/home/chenyy/WORK/models/LLaDA-Image`.
  Files downloaded from the official ModelScope mirror are checked against
  the fixed Hugging Face revision's SHA-256 or Git blob hash.

## Review Findings

All findings below are addressed on the integration branch.

1. **P1: the serving factory could not construct the LLaDA scheduler.**
   `scheduling_options_from_pool()` supplies `min_resize_gain_ms`,
   `resize_hysteresis_ms`, and `resize_control_cost_ms`, which the model's
   scheduling constructor rejected. Forward these options to the shared
   planner and test the real pool-to-scheduler contract.
2. **P2: serving enabled approximate VAE decoding without opt-in.**
   A stage that omitted `parallelism.vae` inherited the generic sharded
   default. LLaDA now defaults to full-image decode, including direct executor
   construction. Explicit `enabled: true` or a fixed degree above one opts in.
   The facade also respects the shared `vae_parallel_degree` setting.
3. **P2: constructing the local text encoder left expert tensors uninitialized.**
   The experts allocated `torch.empty` parameters, but their existing
   `reset_parameters()` was never called by the model initializer. This
   produced intermittent NaNs in checkpoint round-trip tests. Initialize the
   expert parameters through the model's initialization hook and assert
   finite parameters before saving the regression fixture.
4. **P2: warmup omitted the fixed QueryFormer condition tokens.**
   The default measured 32 caption tokens even though the checkpoint adds
   256 learned queries. The default now measures `num_queries + 32`; explicit
   profiling lengths remain supported.

The five textual merge conflicts were mechanical: preserve main's unified
VAE placement calls in FLUX.1, Qwen-Image, Wan, and Z-Image; retain Wan/Z-Image
tensor parallel support and add LLaDA's CFG axis to the service capability
table. Main already normalized an unset Ulysses degree, so the PR's duplicate
normalization and formatting changes were unnecessary. No behavior conflict
requires a user decision.

The bundled DiT classes match the official reference at AST level; the
additional RMSNorm subclass and parallel wrappers are reviewed separately.
The official text checkpoint's complete key set matches the bundled text
encoder. Its differences from the reference concern checkpoint loading,
Transformers compatibility, attention selection, and the local MoE operator.

## Environments

- Existing `.venv`: PyTorch `2.9.1+cu130`, Transformers `4.56.2`,
  Diffusers `0.37.0`.
- Isolated `.venv-pr2-review`: PyTorch `2.10.0+cu128`, Transformers `5.12.1`,
  Diffusers `0.38.0`, with the base requirements declared in `pyproject.toml`.
  The default CUDA 12.8 wheel was used for this isolated check; the user's
  existing CUDA 13 environment is preserved.
- Full `uv sync` also resolves optional extras and currently fails on the
  existing `varlen` dependency `nvidia-cutlass-dsl-libs-base==4.6.0.dev0`.
  The base dependency installation avoids that unrelated optional extra:

  ```bash
  uv pip install --no-config --python .venv-pr2-review/bin/python \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    -r pyproject.toml --no-sources pytest ruff httpx
  ```

## Validation

Acceptance passed. Generated images, numerical outputs, and full logs are
kept under the ignored `outputs/pr2-review/` directory. All GPU checks used
NVIDIA H20 GPUs with approximately 96 GiB of memory each.

- Unit and integration suite: **474 passed, 19 skipped in each environment**.
  GPU-specific checks were run separately; the remaining skips are the
  existing suite's optional hardware/dependency tests.
- CPU two-process parallel contract test: passed.
- H20 two-GPU NCCL contract test: passed, including text/VQ/editing
  transformer equivalence and CFG/VQ synchronization.
- H20 targeted transformer/runtime tests: 20 passed, including the Triton
  RMSNorm comparison with the Diffusers implementation.
- Declared-dependency H20 GPU regressions: 5 passed, covering the two-rank
  NCCL contract and transformer/RMSNorm checks.
- Ruff, Python compilation, CLI help, stage configuration parsing, and
  `git diff --check`: passed.

### Real Checkpoint Results

The prompt was `A red ceramic teapot on a white table, studio photograph`,
seed 42, BF16, guidance 4.5, with exact VAE decode. Editing used the official
reference image and `Change the teapot to blue, preserve the composition`.

| Case | Resolution / steps | Result |
| --- | --- | --- |
| Unmodified official demo, existing environment | 1024 / 50 | Successful; coherent red teapot |
| `chitu generate`, existing environment | 1024 / 50 | Successful; 26.21 dB PSNR against official output |
| `chitu generate`, declared dependencies | 1024 / 50 | Successful; 25.49 dB PSNR against official output |
| Single-GPU text and editing | 512 / 8 | Successful; edit changes teapot to blue |
| Single-GPU VQ | 1024 / 50 | Successful; coherent red teapot, 220.0 s generation |
| Two-GPU VQ + CFG smoke | 256 / 8 | Successful; 38.95 dB PSNR against single-GPU smoke |
| Two-GPU HTTP EPE service | Two concurrent 512 / 8 requests | Both completed in approximately 4.16 s; PNG retrieval and health checks passed |

The 256-pixel VQ run checks execution and rank consistency, not composition
quality: it cropped the subject. The standard 1024-pixel VQ run was therefore
also required and visually inspected. The HTTP test rejected editing mode
with status 422 and the service shut down cleanly after testing.

The following real-checkpoint comparisons use the same 512 / 8 single-GPU
baseline. Their image differences are consistent with BF16 changes in batch
shape, reduction order, and fused kernels; outputs are not bit-identical.

| Topology | Text PSNR (dB) | Editing PSNR (dB) |
| --- | ---: | ---: |
| CFG2 | 32.79 | 51.92 |
| AGKV CP2, serial CFG | 44.70 | 51.79 |
| Ulysses CP2, serial CFG | 44.70 | 51.79 |
| CFG2 x AGKV CP2 (4 GPUs) | 44.70 | 51.79 |
| CFG2 x AGKV CP4 (8 GPUs) | 32.67 | 52.42 |

These are correctness checks, not a controlled performance benchmark;
some jobs ran concurrently. The strict small-model tests provide numerical
parallelism checks independent of the end-to-end image metrics.

### Reproduction

The official baseline used the unmodified reference launcher:

```bash
srun -p debug --nodes 1 --ntasks 1 --gres=gpu:1 \
  --cpus-per-task 8 --mem 160G --time=00:30:00 /usr/bin/env \
  PYTHON="$PWD/.venv/bin/python" \
  MODEL_PATH=/home/chenyy/WORK/models/LLaDA-Image \
  OUTPUT_ROOT="$PWD/outputs/pr2-review/upstream" \
  HEIGHT=1024 WIDTH=1024 STEPS=50 GUIDANCE_SCALE=4.5 SEED=42 \
  GENERATION_MODE=text OMP_NUM_THREADS=8 \
  bash refs/llada-image/scripts/run_llada_image.sh \
  "A red ceramic teapot on a white table, studio photograph"
```

The production CLI acceptance command:

```bash
SRUN_CPUS_PER_GPU=8 SRUN_MEM_PER_GPU=160000 OMP_NUM_THREADS=8 \
tools/cluster/srun_direct.sh 1 1 -m chitu_diffusion.cli generate \
  --model llada-image --model-path /home/chenyy/WORK/models/LLaDA-Image \
  --prompt "A red ceramic teapot on a white table, studio photograph" \
  --height 1024 --width 1024 --steps 50 --guidance-scale 4.5 --seed 42 \
  --local-files-only --output outputs/pr2-review/chitu-cli-1024.png
```

For the declared environment, set `CHITU_PYTHON_BIN` to
`$PWD/.venv-pr2-review/bin/python` and `PYTHONPATH` to `$PWD`.

Reusable real-checkpoint validation:

```bash
tools/cluster/srun_direct.sh 1 1 script/validate_llada_image.py \
  --model-path /home/chenyy/WORK/models/LLaDA-Image \
  --output outputs/pr2-review/smoke-1 --modes text
```

Use the same source image for editing comparisons. `--reference` reads saved
single-GPU pixel tensors and reports RMSE/PSNR; these metrics complement the
strict small-model parallel tests and visual inspection. A successful process
exit alone does not establish image quality or parallel equivalence.

The service used `examples/stage-llada-image.yaml` with the local model path,
port 18372, eight request steps, and an elastic two-rank pool. Its temporary
server was stopped after acceptance; this review does not deploy a service.

## Boundaries

- Batch size and output count are one.
- HTTP supports text-to-image; VQ and editing use the offline API/CLI.
- FlexCache remains explicitly unsupported.
- Parallel tiled VAE decode is approximate and opt-in.
- Fast transports require their own hardware/extension acceptance; the
  default Torch/NCCL path is the baseline for this review.
- Request-side migration cost remains unknown (`state_bytes=0`), as disclosed
  by the PR. Exact prepared latent state sizes are available to workers.
