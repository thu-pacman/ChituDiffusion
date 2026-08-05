# ChituDiffusion

[English](README.md) | [简体中文](README.zh-CN.md)

ChituDiffusion is a Diffusers-native runtime for high-performance,
context-parallel diffusion inference. It keeps the tokenizer, text encoder,
scheduler, VAE, and output processing in the upstream Diffusers lifecycle and
parallelizes the DiT denoising workload through a shared execution backend.

> **Project status:** developer preview. The static generation path and core
> EPE runtime are usable and tested, but the service does not yet provide
> cross-rank fault tolerance. See [Limitations](#limitations) before production
> deployment.

## Highlights

- **One backend, two lifecycles:** `chitu generate` runs one full-world static
  context-parallel request; `chitu serve` keeps the same executor alive behind
  the EPE queue and pulse scheduler.
- **Elastic Parallel Execution (EPE):** measured-cost, SLO-aware scheduling can
  repartition workers into dynamic CP lanes at pulse boundaries.
- **Diffusers-native integration:** model adapters preserve the upstream
  pipeline lifecycle instead of replacing it with a second inference stack.
- **Parallel attention and decode:** AGKV is the default context-parallel
  attention backend; USP and tile-parallel VAE are available where supported.
- **Generate-only FlexCache:** MagCache, MeanCache, TeaCache, TaylorSeer, and
  PAB are isolated, request-local strategies with rank-identical cache control.

## Supported models

| Model family | `generate` | EPE `serve` | FlexCache | Output |
| --- | --- | --- | --- | --- |
| Z-Image | Static CP / CFP+CP | Yes | MeanCache, TaylorSeer, PAB | Image |
| FLUX.1-dev | Static CP | Yes | MagCache, TeaCache, TaylorSeer, PAB | Image |
| Qwen-Image | Static CP / CFP+CP | Yes | MagCache, MeanCache, TaylorSeer, PAB | Image |
| Wan 2.1 T2V | Static CP / CFP+CP | Yes | MagCache, TeaCache, TaylorSeer, PAB | MP4 video |
| FLUX.2-klein | Static CP baseline | No elastic switching | Not validated | Image |

Support is explicit and model-specific. Unsupported cache/model combinations
fail early instead of silently falling back to uncached execution.

## Installation

Requirements:

- Linux with NVIDIA GPUs and a working CUDA/PyTorch installation
- Python 3.12 or newer
- Local or accessible Diffusers-format model checkpoints
- `torchrun` or Slurm for multi-GPU execution

The repository uses `uv` to manage the development environment:

```bash
git clone <repository-url>
cd ChituDiffusion
uv sync
source .venv/bin/activate
chitu --help
```

Select the PyTorch CUDA index in `pyproject.toml` for your cluster before
running `uv sync`. Optional attention backends can be installed with extras,
for example:

```bash
uv sync --extra usp
```

## Quick start

Run a single request:

```bash
chitu generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --prompt "a red cube on a white table" \
  --output outputs/zimage.png
```

Run full-world static CP with four local processes:

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --prompt "a lighthouse above a stormy sea" \
  --output outputs/flux1.png
```

On Slurm, the repository wrappers preserve the same module arguments:

```bash
bash script/srun_direct.sh 1 4 chitu_diffusion/examples/wan_epac.py \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --output outputs/wan.mp4
```

All distributed ranks must enter `generate`. Only the full-world leader returns
and writes the final Diffusers output.

Start the persistent service from a native stage configuration:

```bash
chitu serve --stage-config /path/to/stage.yaml
```

All ranks enter `serve`; the stage leader exposes HTTP while follower ranks run
the distributed worker loop. Non-`none` FlexCache strategies are intentionally
rejected on this path.

## Architecture

```text
Diffusers pipeline lifecycle
        |
model-specific DiffusionBackend / executor
        |
shared request, denoise, decode, and postprocess contract
        |
  +-----+--------------------------------+
  |                                      |
generate: full-world static CP     serve: queue + EPE pulses
                                   dynamic lanes + state migration
```

The maintained source is split by ownership:

- [`chitu_diffusion/epac/`](chitu_diffusion/epac/) contains the model-independent
  executor, cost model, scheduling policy, pulse protocol, and worker runtime.
- [`chitu_diffusion/models/`](chitu_diffusion/models/) contains only
  model-family tensor glue and Diffusers adapters.
- [`chitu_diffusion/parallel/`](chitu_diffusion/parallel/) owns process groups,
  context-parallel attention, and parallel VAE communication.
- [`chitu_diffusion/flexcache/`](chitu_diffusion/flexcache/) implements
  request-local cache strategies without strategy branches in model forwards.
- [`chitu_diffusion/serve/`](chitu_diffusion/serve/) owns configuration, HTTP,
  and distributed service lifecycle.

See the [runtime guide](chitu_diffusion/README.md), [EPAC design
guide](chitu_diffusion/epac/README.md), and [FlexCache extension
guide](chitu_diffusion/flexcache/README.md) for implementation details.

## FlexCache

FlexCache is enabled only for single-request generation:

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/wan-magcache.mp4
```

Strategies keep mutable state per request and isolate serial CFG branches.
Fresh/reuse decisions use replicated inputs or shared schedules so CFP and CP
ranks enter the same collectives. Official MagCache and MeanCache profiles are
selected by model and step count and fail fast outside their calibrated domain.
Detailed support constraints and extension instructions are documented in the
[FlexCache README](chitu_diffusion/flexcache/README.md).

Reproducible speed/quality summaries are retained as text:

- [MagCache comparison](outputs/flexcache/magcache_compare_20260805/result.md)
- [MeanCache comparison](outputs/flexcache/meancache_compare_20260804/result.md)

Raw images, videos, and temporary benchmark data are intentionally ignored.

## Development and validation

Run the active CPU test suite and static checks:

```bash
python -m pytest -q
python -m ruff check chitu_diffusion test
python -m build
```

GPU correctness requires model-specific, same-seed comparisons against each
model's native Diffusers baseline. Example launchers and accepted arguments are
listed in [`chitu_diffusion/examples/README.md`](chitu_diffusion/examples/README.md).

When contributing:

1. Keep model-independent behavior in `epac/`, `parallel/`, or `flexcache/`.
2. Keep checkpoint-specific tensor adaptation in `models/<family>/`.
3. Make unsupported capabilities fail explicitly.
4. Add CPU contract tests and document the GPU command and result.
5. Never let rank-local cache signals change distributed control flow.

## Limitations

- The project is not yet production GA and does not guarantee recovery from an
  arbitrary rank failure.
- FlexCache is generate-only; persistent EPE serving rejects cache strategies.
- Cache profiles are calibrated for specific models, schedulers, and step
  counts. Changing these requires a new quality/performance evaluation.
- FLUX.2-klein currently provides a fixed static-CP baseline only.
- Historical ChituBench, DiTango, staged runtime, configurations, tests, and
  results are frozen under
  [`backup/chitu_diffusion_legacy/`](backup/chitu_diffusion_legacy/). That tree
  is excluded from packages and default tests.

## License

ChituDiffusion is released under the [MIT License](LICENSE). Model checkpoints
and upstream dependencies remain subject to their own licenses and acceptable
use policies.
