# Installation

This guide will help you install ChituDiffusion on your system.

## Prerequisites

Before installing ChituDiffusion, ensure you have:

- **Python**: Version 3.12 or higher
- **CUDA**: Version 12.4 or higher (12.8 recommended)
- **GPU**: NVIDIA GPU with compute capability:
    - 8.0+ (Ampere: A100, A10, etc.)
    - 9.0+ (Hopper: H100, H20, etc.)
    - 9.0+ (Blackwell: B100, B200, 5090, etc.)

## Installation Methods

### Method 1: Using uv (Recommended)

`uv` is a fast Python package manager that simplifies the installation process.

#### Step 1: Clone the Repository

```bash
git clone git@github.com:chen-yy20/SmartDiffusion.git
cd SmartDiffusion
git submodule update --init --recursive
```

#### Step 2: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more installation options, see the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

#### Step 3: Configure CUDA Version

Check your CUDA version:

```bash
nvcc --version
```

Edit `pyproject.toml` to match your CUDA version. For example, for CUDA 12.8:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

Available CUDA versions: `cu124`, `cu126`, `cu128`, `cu130`

#### Step 4: Configure GPU Architecture

Set the `TORCH_CUDA_ARCH_LIST` in `pyproject.toml` according to your GPU:

```toml
[tool.uv.extra-build-variables]
sageattention = { 
    EXT_PARALLEL= "4", 
    NVCC_APPEND_FLAGS="--threads 8", 
    MAX_JOBS="32", 
    "TORCH_CUDA_ARCH_LIST" = "8.0;9.0"  # Adjust based on your GPU
}
spas_sage_attn = { 
    EXT_PARALLEL= "4", 
    NVCC_APPEND_FLAGS="--threads 8", 
    MAX_JOBS="32", 
    "TORCH_CUDA_ARCH_LIST" = "8.0;9.0"  # Adjust based on your GPU
}
```

Common architectures:
- Ampere (A100, A10): `8.0`
- Hopper (H100, H20): `9.0`
- Blackwell (B100, B200, 5090): `9.0`

#### Step 5: Install Dependencies

**Basic Installation** (FlashAttention only):

```bash
uv sync -v 2>&1 | tee uv_sync.log
```

**Full Installation** (with SageAttention and SpargeAttention):

```bash
uv sync -v --all-extras 2>&1 | tee build.log
```

!!! note "Build Time"
    Full installation takes approximately 10 minutes on a machine with 32 cores and 256GB RAM.

### Method 2: Using pip

#### Step 1: Clone the Repository

```bash
git clone git@github.com:chen-yy20/SmartDiffusion.git
cd SmartDiffusion
git submodule update --init --recursive
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Install ChituDiffusion

```bash
pip install -e .
```

### Method 3: Docker (Coming Soon)

Docker support is planned for future releases.

## Installing Flash Attention

Flash Attention can be installed via prebuilt wheels:

1. Visit [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.1.post2)
2. Download the wheel matching your Python version and CUDA version
3. Install: `pip install flash_attn-*.whl`

Or build from source (requires ~10-15 minutes):

```bash
pip install flash-attn --no-build-isolation
```

## Verifying Installation

After installation, verify that ChituDiffusion is correctly installed:

```python
import chitu_core
import chitu_diffusion
print("ChituDiffusion installed successfully!")
```

Check available attention backends:

```python
from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttnBackend

# This will list available attention types
backend = DiffusionAttnBackend("auto")
```

## Troubleshooting

### CUDA Version Mismatch

If you encounter CUDA version mismatches:

1. Check your installed CUDA: `nvcc --version`
2. Update `pyproject.toml` to match your CUDA version
3. Reinstall: `uv sync -v --reinstall`

### Out of Memory During Build

If compilation fails due to memory:

1. Reduce `MAX_JOBS` in `pyproject.toml`
2. Retry the build

### Symbol Link Issues with Flash Attention

If you encounter symbol link issues with Flash Attention, uncomment the source build option in `pyproject.toml`:

```toml
[tool.uv.extra-build-variables]
flash_attn = { 
    FLASH_ATTN_CUDA_ARCHS = "80",
    FLASH_ATTENTION_FORCE_BUILD = "TRUE" 
}
```

## Next Steps

Once installation is complete, proceed to:

- [Quick Start Guide](quick-start.md) - Run your first generation
- [Configuration Guide](configuration.md) - Learn about configuration options
- [Basic Usage](../user-guide/basic-usage.md) - Explore the API

## Getting Help

If you encounter issues:

1. Check the [FAQ](../faq.md)
2. Search [existing issues](https://github.com/chen-yy20/SmartDiffusion/issues)
3. Open a [new issue](https://github.com/chen-yy20/SmartDiffusion/issues/new) with:
   - Your system configuration
   - Installation method used
   - Error messages and logs
