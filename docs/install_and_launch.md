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
uv sync --extra flash3  # Hopper SM90; loads kernels-community/flash-attn3 on first use
uv sync --extra flashinfer
uv sync --extra fast-ulysses
uv sync --extra eval
```

The `eval` extra installs the image-quality stack used by ChituBench and the
runtime evaluator: `scikit-image`, `lpips`, `pytorch-fid`, `torch-fidelity`,
`torchmetrics`, and `hpsv3` (including its import-time dependencies such as
`matplotlib` and `tensorboard`).

Build CUDA extension extras on a GPU machine whose CUDA toolkit and PyTorch
version match the selected wheel index.

### Optional fast-ulysses transport

`fast-ulysses` is a compiled NVSHMEM extension, not a portable Python-only
dependency. ChituDiffusion pins the reviewed upstream revision and exposes it
through the `fast-ulysses` uv extra, but the host must provide the CUDA toolkit
and NVSHMEM installation:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDACXX="$CUDA_HOME/bin/nvcc"
export NVSHMEM_HOME=/opt/nvshmem
export FAST_ULYSSES_CUDA_ARCH=90  # H100/H200/H20

uv sync --extra fast-ulysses
```

Validated stack:

- Linux, Python 3.12, PyTorch 2.9.1+cu130.
- CUDA toolkit 13.0, including a matching compiler, runtime, and CCCL headers.
- NVSHMEM 3.7.2; NVSHMEM 3.7 or newer is required.
- One node with 2-8 mutually P2P-reachable NVIDIA GPUs. Hopper SM90 with NVLink
  is the validated and automatically selected target.

The CUDA toolkit must match the CUDA major/minor used by PyTorch. In particular,
do not mix CUDA 13.2 `nvcc`/PTX tools with the cu130 headers and runtime. Set
`LD_LIBRARY_PATH` so the CUDA runtime and `$NVSHMEM_HOME/lib` are visible when
importing the extension.

The upstream uv package supports a WORLD-spanning Ulysses group. CFP2 requires
the Chitu subgroup extension (`SUPPORTS_WORLD_PARTITION_SUBGROUPS`) in the
reviewed development checkout under `refs/fast-ulysses`. Install that checkout
instead when testing CFP2:

```bash
NVSHMEM_HOME=/opt/nvshmem \
  uv pip install --no-build-isolation -e refs/fast-ulysses
```

Without the subgroup feature marker, transport auto-selection safely retains
NCCL for subgroup layouts. Publishing the subgroup revision upstream (or from a
maintained fork) is required before the one-command uv extra can provide CFP2
support on a fresh clone.

At runtime there are two independent choices:

- `infer.cp_backend=auto` selects AGCP or UCP.
- Within UCP, `CHITU_CP_ULYSSES_BACKEND=auto` (the default) selects
  fast-ulysses only when every rank reports Hopper, a single-node P2P topology,
  a loadable extension, and required subgroup support. Otherwise it retains
  NCCL.

Explicit overrides are available for diagnosis and benchmarking:

```bash
export CHITU_CP_ULYSSES_BACKEND=fast_ulysses  # fail if prerequisites are absent
export CHITU_CP_ULYSSES_BACKEND=nccl          # always use the Torch/NCCL path
export CHITU_FAST_ULYSSES_USE_TMA=auto        # auto, tma, or non_tma
export CHITU_FAST_ULYSSES_POOL_BYTES=$((2 << 30))
```

Fast transport is limited to eager, uniform 4D FP16/BF16 Ulysses swaps. CUDA
Graph capture, 3D varlen attention, FP8 communication, multi-node layouts, and
hybrid Ulysses+Ring groups with a smaller UP subgroup continue to use NCCL.

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
