# ChituDiffusion

ChituDiffusion is a high-performance diffusion inference framework focused on
video generation workloads. It provides a compact runtime for distributed
inference, multiple attention backends, FlexCache and DiTango acceleration, and
optional evaluation utilities.

The project is under active development. The current public interface is the
`chitu` command plus the configuration in `system_config.yaml`.

## Features

- Distributed diffusion inference with context parallelism and CFG parallelism.
- Attention backend selection for FlashAttention, SageAttention, and
  SpargeAttention.
- FlexCache strategies including TeaCache and PAB, plus independent DiTango
  planner/runtime acceleration.
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

`uv sync` installs the CLI into the project virtual environment. Activate it
before using the bare `chitu` command:

```bash
source .venv/bin/activate
chitu --help
```

If you do not want to activate the environment, use `uv run chitu ...`.

Optional extras are available for acceleration and evaluation:

```bash
uv sync --extra sage
uv sync --extra sparge
uv sync --extra flash
uv sync --extra eval
uv sync --extra vbench
```

Build CUDA extension extras on a GPU compute node whose CUDA toolkit matches
the selected PyTorch build.

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
  up: 8

infer:
  attn_type: torch_sdpa
```

The most important field is `model.ckpt_dir`; it must point to a local model
checkpoint directory.

## Usage

Run generation through the single repository entry point:

```bash
chitu run system_config.yaml
```

Common overrides:

```bash
chitu run system_config.yaml --gpus-per-node 8 --cfp 2
```

`chitu run` reads `system_config.yaml`, builds dotlist overrides, and launches
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
chitu_diffusion/flexcache/       [DiT Cache] FlexCache strategies and shared cache utilities
chitu_diffusion/ditango/         [HPDC'26] DiTango planner, runtime attention, visualization
chitu_diffusion/evaluation/      Evaluation manager, strategies, metric helpers
chitu_diffusion/observability/   Timing and magnitude logging helpers
script/                         Launch helpers for local and Slurm execution
test/                           Generation and acceleration test entry points
system_config.yaml              Default runtime configuration
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
    <task_id>/
      *.mp4
      *.json
  metrics/
    timing/
      summary.json
      <task_id>.json
    memory/
      rank<N>.json
    quality/
      summary.json
  logs/
    command.log
    run.log
    run.rank<N>.log
    <task_id>/
      *.ppm
```

`results/<task_id>/` contains generated media and sidecar metadata. `metrics/`
contains JSON-only timing, memory, and quality files in separate subdirectories.
Timing JSON includes aggregate timer stats; `timers.dit_forward.total_ms` is the
overall DiT forward time, and `records.dit_forward_step` stores per-timestep DiT
forward times. Memory JSON is grouped by rank, so `model_loaded`,
`task_complete`, and `final` events for the same rank live in one file.
`output.memory` toggles memory metrics. `output.log_ranks` controls which ranks
write memory metrics and Python logs. Quality JSON includes `by_task_id` groups
for multi-request runs. `logs/` contains process logs and per-task debug
visualizations. `command.log` captures the full launch command output,
including `chitu run`, `srun`, wrapper output, and Python stdout/stderr.

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
