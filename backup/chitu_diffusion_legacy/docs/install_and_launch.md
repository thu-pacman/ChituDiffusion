# Installation and Launch Guide

This guide covers the common ways to prepare ChituDiffusion on development
machines, CUDA machines, Slurm clusters, and direct `torchrun` hosts.

## Environment Layout

Recommended split:

- Keep source code on shared storage, for example `/shared/ChituDiffusion`.
- Keep model checkpoints outside the Git repository, for example `/shared/models`.
- Install Python dependencies in the runtime machine or image.
- Mount checkpoints into containers or CUDA machines and reference them through `model.ckpt_dir`.

ChituDiffusion requires Python 3.12 or newer for the main `pyproject.toml` path.

## Install With uv

Use this path on machines that can access PyPI and PyTorch wheel indexes.

```bash
cd /path/to/ChituDiffusion

# Optional but recommended when submodules are needed.
git submodule update --init --recursive

# Base environment.
uv sync

# Activate the environment.
source .venv/bin/activate
chitu --help
```

Optional acceleration and evaluation stacks are installed only when needed:

```bash
uv sync --extra sage
uv sync --extra sparge
uv sync --extra flash
uv sync --extra flashinfer
uv sync --extra eval
```

The `eval` extra installs the image-quality stack used by ChituBench and the
runtime evaluator: `scikit-image`, `lpips`, `pytorch-fid`, `torch-fidelity`,
`torchmetrics`, and `hpsv3` (including its import-time dependencies such as
`matplotlib` and `tensorboard`).

Build CUDA extension extras on a GPU machine whose CUDA toolkit and PyTorch
version match the selected wheel index.

## Install With pip

Use this path when the CUDA machine already has a system Python or prebuilt
PyTorch environment.

If the machine already has the correct `torch` and `torchvision`, install the
missing Python packages first, then install the project without changing the
existing dependency stack:

```bash
cd /path/to/ChituDiffusion

python3 -m pip install \
  blobfile faker fire ftfy plum-dispatch protobuf netifaces pandas

python3 -m pip install -e . --no-deps
```

Install the quality-evaluation stack with the same optional extra used by `uv`:

```bash
python3 -m pip install -e ".[eval]"
```

If the target already has a pinned CUDA/PyTorch stack and you must avoid changing
it, install the `eval` packages in a wheelhouse or constraints-aware step, then
keep the project editable install as `--no-deps`.

If you want pip to install dependencies from `requirements.txt`, make sure the
PyTorch lines in that file match the CUDA/PyTorch stack on the target machine.
For example, a CUDA 12.4 environment may use:

```bash
python3 -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -e .
```

For an already-provisioned system Python, prefer `--no-deps` for the editable
install to avoid accidentally downgrading PyTorch.

## Docker

Use Docker when you want to freeze the base runtime and keep code/checkpoints on
shared storage.

Build an image:

```bash
cd /path/to/ChituDiffusion
docker build --network host -t chitudiffusion:py312-cu130 .
```

Run on a CUDA machine:

```bash
docker run --gpus all -it --rm \
  -v /path/to/ChituDiffusion:/opt/ChituDiffusion \
  -v /path/to/models:/models:ro \
  chitudiffusion:py312-cu130 \
  bash
```

If you use a private registry, tag and push the image with your own registry
address:

```bash
docker tag chitudiffusion:py312-cu130 <registry>/<namespace>/chitudiffusion:py312-cu130
docker push <registry>/<namespace>/chitudiffusion:py312-cu130
```

Keep large checkpoints outside the image and mount them at runtime.


## Configure Checkpoints

Edit `system_config.yaml` or a copied config file:

```yaml
model:
  name: Z-Image
  ckpt_dir: /models/zimage/Z-Image
```

For direct shared-storage execution without Docker, use the absolute checkpoint
path visible to the CUDA machine.

## Launch With Slurm

Slurm remains the default launch backend.

```yaml
launch:
  backend: srun
  num_nodes: 1
  gpus_per_node: 8
  srun:
    partition: debug
    cpus_per_gpu: 24
    job_name: chitu-diffusion
```

Run:

```bash
chitu run system_config.yaml
```

Useful overrides:

```bash
chitu run system_config.yaml --gpus-per-node 8 --cfp 2
```

## Launch With torchrun

Use this backend on a direct CUDA machine without Slurm.

```yaml
launch:
  backend: torchrun
  num_nodes: 1
  gpus_per_node: 4
```

For a base installation without FlashAttention/Sage/Sparge, use the PyTorch
attention backend:

```yaml
infer:
  attn_type: torch_sdpa
```

Run:

```bash
chitu run system_config.yaml
```

For multi-node `torchrun`, set the standard rendezvous variables before running
the command on each node:

```bash
export MASTER_ADDR=<rank0-host-or-ip>
export MASTER_PORT=29500
export NODE_RANK=<this-node-rank>

chitu run system_config.yaml
```

The launcher passes `num_nodes`, `gpus_per_node`, and `NODE_RANK` to
`torch.distributed.run`.

## Common Checks

Confirm the runtime sees GPUs:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Confirm the CLI is visible:

```bash
chitu --help
```

If `chitu` is not on `PATH`, run through Python:

```bash
PYTHONPATH=/path/to/ChituDiffusion \
python3 -m chitu_diffusion.cli --help
```
