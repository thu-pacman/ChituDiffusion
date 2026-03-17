# Smart-Diffusion

[中文版](./README_zh.md) | [Why Smart-Diffusion?](./docs/whySmart.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.4+](https://img.shields.io/badge/cuda-12.4+-green.svg)](https://developer.nvidia.com/cuda-downloads)

Smart-Diffusion is a high-performance diffusion model inference framework built on [Chitu](https://github.com/thu-pacman/chitu). It provides extreme performance and flexible scheduling for AI-generated content (AIGC) workloads.

## Overview

Smart-Diffusion is the pure enjoyment version of Chitu-Diffusion, developed by the PACMAN team from Tsinghua University and QingCheng.ai. We aim to provide support for the rapidly growing Diffusion ecosystem by restructuring DiT models under the API and scheduling philosophy of Chitu, maintaining scheduling flexibility while offering extreme performance.

### Key Features

- **🚀 High Performance**: Optimized diffusion inference with advanced parallelism strategies
- **🔧 Flexible Architecture**: Support for multiple attention backends (FlashAttention, SageAttention, SpargeAttention)
- **💾 Memory Efficient**: Low memory mode with model offloading and VAE tiling
- **📊 Feature Cache**: Support for lossy acceleration algorithms (TeaCache, PAB)
- **🎯 Easy to Use**: Simple API with per-request parameter configuration
- **🌐 Multi-Model**: Currently supports Wan-T2V series (1.3B, 14B, A14B) with more coming soon

### Design Philosophy

Smart-Diffusion follows three core pillars:
1. **Parallelism**: Context parallelism (CP), CFG parallelism, and data parallelism
2. **Kernels**: Optimized attention implementations with quantization support
3. **Algorithms**: Feature reuse and caching strategies for acceleration

See [Why Smart-Diffusion?](./docs/whySmart.md) for detailed design philosophy.

## Table of Contents

- [Installation](#installation)
  - [Quick Start with uv](#quick-start-with-uv)
  - [Manual Installation](#manual-installation)
- [Supported Models](#supported-models)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Advanced Configuration](#advanced-configuration)
- [Key Parameters](#key-parameters)
  - [Attention Backend](#attention-backend)
  - [Low Memory Mode](#low-memory-mode)
  - [FlexCache](#flexcache)
  - [Evaluation](#evaluation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

### Prerequisites

- Python 3.12+
- CUDA 12.4+ (recommended: 12.8)
- NVIDIA GPU with compute capability 8.0+ (Ampere) or 9.0+ (Hopper/Blackwell)

### Quick Start with uv

We recommend using `uv` for a smoother installation experience.

#### 1. Clone the repository

```bash
git clone git@github.com:chen-yy20/SmartDiffusion.git
cd SmartDiffusion
git submodule update --init --recursive
```

#### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for more details.

#### 3. Configure build settings

Check your CUDA version:
```bash
nvcc --version
```

Edit `pyproject.toml` to match your CUDA version. For CUDA 12.8:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

Configure GPU architecture in `pyproject.toml`:

```toml
[tool.uv.extra-build-variables]
# Set TORCH_CUDA_ARCH_LIST according to your GPU
# Ampere: 8.0, Hopper: 9.0, Blackwell: 9.0
sageattention = { 
    EXT_PARALLEL= "4", 
    NVCC_APPEND_FLAGS="--threads 8", 
    MAX_JOBS="32", 
    "TORCH_CUDA_ARCH_LIST" = "8.0;9.0"
}
spas_sage_attn = { 
    EXT_PARALLEL= "4", 
    NVCC_APPEND_FLAGS="--threads 8", 
    MAX_JOBS="32", 
    "TORCH_CUDA_ARCH_LIST" = "8.0;9.0"
}
```

#### 4. Install dependencies

```bash
# Basic installation (FlashAttention only)
uv sync -v 2>&1 | tee uv_sync.log

# Full installation with quantized attention (SageAttention + SpargeAttention)
# Build time: ~10 minutes on 32-core, 256GB memory
uv sync -v --all-extras 2>&1 | tee build.log
```

### Manual Installation

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

**Note**: Flash Attention can be installed via wheel from [GitHub releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.1.post2).

## Supported Models

Smart-Diffusion currently supports the Wan-T2V series:

| Model ID | Parameters | Description |
|----------|------------|-------------|
| `Wan-AI/Wan2.1-T2V-1.3B` | 1.3B | Lightweight text-to-video model |
| `Wan-AI/Wan2.1-T2V-14B` | 14B | High-quality text-to-video model |
| `Wan-AI/Wan2.2-T2V-A14B` | 14B | Advanced two-stage text-to-video model |

More models are being added continuously. Stay tuned!

## Usage

### Basic Example

Create a test script `test_generate.py`:

```python
from chitu_diffusion import chitu_init, chitu_generate, chitu_start
from chitu_diffusion.task import DiffusionUserParams, DiffusionTask, DiffusionTaskPool
from hydra import compose, initialize

# Initialize with configuration
initialize(config_path="config", version_base=None)
args = compose(config_name="wan")

# Set model checkpoint path
args.models.ckpt_dir = "/path/to/your/model/checkpoint"

# Initialize backend
chitu_init(args)
chitu_start()

# Create generation task
user_params = DiffusionUserParams(
    role="user1",
    prompt="A cat walking on grass.",
    num_inference_steps=50,
    height=480,
    width=848,
    num_frames=81,
    guidance_scale=7.0,
)

task = DiffusionTask.from_user_request(user_params)
DiffusionTaskPool.add(task)

# Generate
while not DiffusionTaskPool.all_finished():
    chitu_generate()

print(f"Video saved to: {task.buffer.save_path}")
```

### Launch Scripts

Only `srun` launch is supported.

1. Edit `system_config.yaml` to configure model path, system params, and `cfp`.
2. Run the unified launcher:

```bash
bash run.sh system_config.yaml
```

Optional runtime overrides:

```bash
bash run.sh system_config.yaml --num-nodes 2 --gpus-per-node 8 --cfp 2
```

### Advanced Configuration

Configuration is split into three levels:

1. **Model Parameters** (Static): Defined in `chitu_core/config/models/<model>.yaml`
2. **User Parameters** (Dynamic): Set per-request via `DiffusionUserParams`
3. **System Parameters** (Semi-static): Set in `system_config.yaml`

**Example: Using different attention backend**
```bash
python test_generate.py \
    models.ckpt_dir=/path/to/checkpoint \
    infer.attn_type=sage \
    infer.diffusion.low_mem_level=2
```

## Key Parameters

### Attention Backend

Control your attention implementation with `infer.attn_type`:

| Type | Description | Performance |
|------|-------------|-------------|
| `flash_attn` | Default FlashAttention. High-performance full attention without accuracy loss | Baseline |
| `sage` | [SageAttention](https://github.com/thu-ml/SageAttention) (NIPS25 spotlight). Train-free quantized attention | ~2x speedup |
| `sparge` | [SpargeAttention](https://github.com/thu-ml/SpargeAttn) (ICML25). Train-free sparse attention | ~3x speedup |
| `auto` | Automatically choose best backend | - |

**Example:**
```bash
python test_generate.py infer.attn_type=sage
```

### Low Memory Mode

Control GPU memory usage with `infer.diffusion.low_mem_level`:

| Level | Behavior |
|-------|----------|
| 0 | All models loaded to GPU |
| 1 | VAE enables tiling |
| 2 | T5 encoder offloaded to CPU |
| ≥3 | DiT model offloaded to CPU |

**Example:**
```bash
python test_generate.py infer.diffusion.low_mem_level=2
```

### FlexCache

Enable feature reuse acceleration with `infer.diffusion.enable_flexcache=true`:

| Method | cache_type | Description |
|--------|------------|-------------|
| `teacache` | [TeaCache](https://github.com/ali-vilab/TeaCache) | CVPR24 spotlight. Time embedding tells. |
| `PAB` | [Pyramid Attention Broadcast](https://oahzxl.github.io/PAB/) | ICLR25. Pyramid attention broadcasting |

**Example:**
```python
user_params = DiffusionUserParams(
    prompt="A cat walking on grass.",
    flexcache='teacache',  # or 'PAB'
    # ... other params
)
```

### Evaluation

Enable automatic evaluation with `eval.eval_type` (multi-select):

```bash
python test_generate.py eval.eval_type=[vbench,fid,psnr] eval.reference_path=/path/to/reference_videos
```

Supported evaluation methods:
- `vbench`: VBench custom-mode evaluation
- `fid`: Frechet Inception Distance (requires `reference_path`)
- `fvd`: Frechet Video Distance (requires `reference_path`)
- `psnr`: Peak Signal-to-Noise Ratio (requires `reference_path`)
- `ssim`: Structural Similarity Index (requires `reference_path`)
- `lpips`: Learned Perceptual Image Patch Similarity (requires `reference_path`)

Behavior notes:
- `eval.eval_type=[]` or `null` disables evaluation.
- Metrics requiring references are skipped with warning if `eval.reference_path` is missing or invalid.
- Results are saved under `./vbench_out/` (`vbench`) and `./eval_out/` (other metrics).

## Documentation

- [Why Smart-Diffusion?](./docs/whySmart.md) - Design philosophy and architecture
- [API Reference](./docs/) - Detailed API documentation
- [Configuration Guide](./docs/) - Complete configuration options

## Contributing

We welcome contributions! Smart-Diffusion is in active development.

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper documentation
4. Submit a pull request

Please see our [Developer Guide](./docs/whySmart.md#developer-guide) for parameter taxonomy and best practices.

## Community

- **Issues**: [GitHub Issues](https://github.com/chen-yy20/SmartDiffusion/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chen-yy20/SmartDiffusion/discussions)

## Roadmap

- [ ] More diffusion model support (Flux2, Longcat-Video, FireRed etc.)
- [ ] More acceleration algorithms
- [ ] More parallelism strategies
- [ ] Better operator implementations
- [ ] Production-ready serving framework
- [ ] Comprehensive benchmarks

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Smart-Diffusion in your research, please cite:

```bibtex
@software{smart_diffusion2025,
  title={Smart-Diffusion: High-Performance Diffusion Model Inference Framework},
  author={PACMAN Team, Tsinghua University and QingCheng.ai},
  year={2025},
  url={https://github.com/chen-yy20/SmartDiffusion}
}
```

## Acknowledgments

- [Chitu](https://github.com/thu-pacman/chitu) - Base inference framework
- [xDiT](https://github.com/xdit-project/xDiT) - Scalable Inference Engine for Diffusion Transformers
- [SGLang-Diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen) - Image/Video Generation Framework
- [SageAttention](https://github.com/thu-ml/SageAttention) - Quantized attention implementation
- [SpargeAttention](https://github.com/thu-ml/SpargeAttn) - Sparse+Sage attention implementation
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Efficient attention implementation
- [TeaCache](https://github.com/ali-vilab/TeaCache) - Feature cache strategy
- [PyramidAttentionBroadcast](https://oahzxl.github.io/PAB/) - PAB algorithm


---

**Note**: Smart-Diffusion is currently in testing and development phase. We're working hard to make it better! Join us in building the future of AIGC acceleration. 🚀
