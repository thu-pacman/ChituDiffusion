# ChituDiffusion

ChituDiffusion is a high-performance diffusion inference framework focused on
video generation workloads. It provides a compact runtime for distributed
inference, multiple attention backends, FlexCache acceleration strategies, and
optional evaluation utilities.

The project is under active development. The current public interface is the
repository launcher in `run.sh` plus the configuration in `system_config.yaml`.

## Features

- Distributed diffusion inference with context parallelism and CFG parallelism.
- Attention backend selection for FlashAttention, SageAttention, and
  SpargeAttention.
- FlexCache strategies including TeaCache, PAB, and DiTango.
- Optional low-memory execution modes.
- Built-in timing, logging, output naming, and evaluation helpers.
- Initial model support for Wan text-to-video models.

## Requirements

- Python 3.12 or newer
- CUDA-capable NVIDIA GPU
- CUDA and PyTorch versions compatible with the selected `pyproject.toml`
  package index
- `uv` is recommended for dependency management

## Installation

Clone the repository and initialize optional submodules as needed:

```bash
git clone <repo-url>
cd ChituDiffusion
git submodule update --init --recursive
```

Install the base environment:

```bash
uv sync
```

Optional extras are available for acceleration and evaluation:

```bash
uv sync --extra sage
uv sync --extra sparge
uv sync --extra eval
uv sync --extra vbench
```

For manual environments:

```bash
pip install -r requirements.txt
pip install -e .
```

## Configuration

Edit `system_config.yaml` before running:

```yaml
model:
  name: Wan2.1-T2V-1.3B
  ckpt_dir: /path/to/Wan2.1-T2V-1.3B

launch:
  num_nodes: 1
  gpus_per_node: 8

parallel:
  cfp: 2

infer:
  attn_type: flash_attn
  enable_flexcache: true
```

The most important field is `model.ckpt_dir`; it must point to a local model
checkpoint directory.

## Usage

Run generation through the single repository entry point:

```bash
bash run.sh system_config.yaml
```

Common overrides:

```bash
bash run.sh system_config.yaml --gpus-per-node 8 --cfp 2
```

`run.sh` reads `system_config.yaml`, builds dotlist overrides, and launches
the configured Python entry through the runtime script.

## Supported Models

The current configuration set includes:

- `Wan2.1-T2V-1.3B`
- `Wan2.1-T2V-14B`
- `Wan2.2-T2V-A14B`
- `FLUX.2-klein-4B`

Model availability depends on the local checkpoint path and the corresponding
configuration under the project config directory.

## Repository Layout

```text
chitu_diffusion/core/            Configuration, schemas, distributed utilities, registry
chitu_diffusion/runtime/         Backend, generator, scheduler, task, main runtime API
chitu_diffusion/modules/         Model-specific and reusable diffusion modules
chitu_diffusion/flex_cache/      FlexCache strategies
chitu_diffusion/evaluation/      Evaluation manager, strategies, metric helpers
chitu_diffusion/observability/   Timing and magnitude logging helpers
script/                         Launch helpers for local and Slurm execution
test/                           Generation and FlexCache test entry points
system_config.yaml              Default runtime configuration
run.sh                          Main launch entry point
```

This layout will continue to be simplified as the project is prepared for
public release.

## Evaluation

Evaluation can be enabled from `system_config.yaml`:

```yaml
eval:
  eval_type: [psnr, lpips]
  reference_path: /path/to/reference/videos
```

Additional metric dependencies are installed with:

```bash
uv sync --extra eval
```

## Outputs

Each run writes to:

```text
outputs/<tag>-<YYYYMMDD_HHMMSS>-<taskid>/
  request_params.json
  system_params.json
  run_config.yaml
  results/
  metrics/
    summary.json
    timing.csv
    timing.json
    quality/
  logs/
    command.log
    run.log
    run.rank<N>.log
```

`results/` contains generated media and sidecar metadata. `metrics/` contains
timing, memory, and quality evaluation files. `logs/` contains process logs and
debug visualizations. `command.log` captures the full launch command output,
including `run.sh`, `srun`, wrapper output, and Python stdout/stderr.

## Development

Run a lightweight import check:

```bash
python - <<'PY'
import chitu_diffusion.core
from chitu_diffusion.runtime.task import DiffusionUserParams
from chitu_diffusion.observability import Timer
print("imports ok")
PY
```

Run tests with:

```bash
pytest test
```

Some tests require CUDA, local checkpoints, and distributed launch settings.

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for
details.
