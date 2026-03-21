# Configuration Guide

This guide explains all configuration options available in Smart-Diffusion.

## Configuration Levels

Smart-Diffusion uses a three-tier configuration system:

### 1. Model Parameters (Static)

**Location**: `chitu_core/config/models/<model>.yaml`

**Purpose**: Define model architecture

**Can be changed**: No (tied to checkpoint weights)

**Examples**:
- Number of layers
- Hidden dimensions
- Attention heads
- Model-specific hyperparameters

### 2. User Parameters (Dynamic)

**Location**: `DiffusionUserParams` class

**Purpose**: Control per-request generation

**Can be changed**: Yes (for each generation)

**Key Parameters**:

```python
DiffusionUserParams(
    # Basic
    role="user1",                    # User identifier
    prompt="A cat on grass",         # Text prompt
    
    # Video properties
    height=480,                      # Video height in pixels
    width=848,                       # Video width in pixels
    num_frames=81,                   # Number of frames
    fps=24,                          # Frames per second
    
    # Generation quality
    num_inference_steps=50,          # Denoising steps (30-100)
    guidance_scale=7.0,              # CFG scale (5.0-15.0)
    
    # Advanced
    seed=None,                       # Random seed (None = random)
    save_path=None,                  # Output path (None = auto)
    flexcache=None,                  # Legacy cache strategy field
    flexcache_params=FlexCacheParams(
        strategy="teacache",        # 'teacache' / 'pab' / 'ditango'
        cache_ratio=0.4,             # 0 quality-first, 1 speed-first
        warmup=5,                    # First 5 steps full compute
        cooldown=5,                  # Last 5 steps full compute
    ),
)
```

### 3. System Parameters (Semi-static)

**Location**: Launch arguments (command line or config files)

**Purpose**: Configure system behavior

**Can be changed**: Only at initialization

**Categories**:
- Model selection
- Parallelism configuration
- Memory management
- Attention backends
- Evaluation settings

## System Configuration

### Model Selection

```bash
# Specify model name
models.name="Wan2.1-T2V-14B"

# Specify checkpoint directory
models.ckpt_dir="/path/to/checkpoint"
```

Supported models:
- `Wan2.1-T2V-1.3B`
- `Wan2.1-T2V-14B`
- `Wan2.2-T2V-A14B`

### Attention Backend

```bash
# Select attention implementation
infer.attn_type=<type>
```

Options:
- `flash_attn` - Default FlashAttention (accurate, fast)
- `sage` - SageAttention (quantized, performance testing in progress)
- `sparge` - SpargeAttention (sparse, performance testing in progress)
- `auto` - Automatically select best available

### Memory Management

```bash
# Set low memory level (0-3)
infer.diffusion.low_mem_level=<level>
```

Levels:
- **0**: All models on GPU (default)
- **1**: Enable VAE tiling
- **2**: Offload T5 encoder to CPU
- **3+**: Offload DiT model to CPU

### Parallelism

#### Context Parallelism

```bash
# Split sequence across GPUs
infer.diffusion.cp_size=<num_gpus>
infer.diffusion.up_limit=<seq_length>
```

Example:
```bash
# Use 2 GPUs for context parallelism
infer.diffusion.cp_size=2 infer.diffusion.up_limit=81
```

#### CFG Parallelism

```bash
# Automatically enabled when world_size >= 2 and CFG is active
# Can be explicitly controlled:
infer.diffusion.cfg_size=<num_gpus>
```

Options:
- `1`: No CFG parallelism
- `2`: Split positive/negative prompts

### FlexCache

```bash
# Enable feature cache
infer.diffusion.enable_flexcache=true
```

Then set cache type in user parameters:
```python
from chitu_diffusion.task import DiffusionUserParams, FlexCacheParams

DiffusionUserParams(
    prompt="...",
    flexcache_params=FlexCacheParams(
        strategy='teacache',
        cache_ratio=0.4,
        warmup=5,
        cooldown=5,
    )
)
```

Legacy style is still supported:

```python
DiffusionUserParams(
    prompt="...",
    flexcache='teacache'
)
```

### Evaluation

```bash
# Enable automatic evaluation (multi-select)
eval.eval_type=[vbench,fid,psnr]
eval.reference_path=/path/to/reference_videos
```

Options:
- `[]`/`null` - No evaluation (default)
- `vbench` - VBench custom-mode evaluation
- `fid` - Frechet Inception Distance (needs `eval.reference_path`)
- `fvd` - Frechet Video Distance (needs `eval.reference_path`)
- `psnr` - Peak Signal-to-Noise Ratio (needs `eval.reference_path`)
- `ssim` - Structural Similarity (needs `eval.reference_path`)
- `lpips` - Perceptual similarity LPIPS (needs `eval.reference_path`)

If `eval.reference_path` is missing or invalid, reference-based metrics are skipped with warning while other selected metrics continue.

### Other Settings

```bash
# Random seed
infer.seed=42

# Precision
float_16bit_variant="bfloat16"  # or "float16"

# Output directory
output_dir="./outputs"

# Logging level
logging_level="INFO"  # or "DEBUG"
```

## Configuration Files

### Using Hydra

Smart-Diffusion uses Hydra for configuration management.

**Default config**: `config/wan.yaml`

**Override from command line**:
```bash
python test_generate.py \
    models.name=Wan2.1-T2V-14B \
    models.ckpt_dir=/path/to/checkpoint \
    infer.attn_type=sage \
    infer.diffusion.low_mem_level=2
```

**Create custom config**:

```yaml
# config/my_config.yaml
models:
  name: Wan2.1-T2V-14B
  ckpt_dir: /path/to/checkpoint

infer:
  attn_type: sage
  seed: 42
  diffusion:
    low_mem_level: 2
    cp_size: 1
    enable_flexcache: false

output_dir: ./my_outputs
```

Use with:
```bash
python test_generate.py --config-name my_config
```

## Environment Variables

Smart-Diffusion respects several environment variables:

```bash
# Enable debug mode
export CHITU_DEBUG=1

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0,1

# Distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0
export LOCAL_RANK=0
```

## Common Configuration Patterns

### High Quality, Slow

```bash
python test_generate.py \
    models.name=Wan2.1-T2V-14B \
    infer.attn_type=flash_attn \
    infer.diffusion.low_mem_level=0
```

```python
DiffusionUserParams(
    prompt="...",
    height=720,
    width=1280,
    num_frames=121,
    num_inference_steps=100,
    guidance_scale=9.0,
)
```

### Fast, Lower Quality

```bash
python test_generate.py \
    models.name=Wan2.1-T2V-1.3B \
    infer.attn_type=sparge \
    infer.diffusion.low_mem_level=1
```

```python
DiffusionUserParams(
    prompt="...",
    height=480,
    width=848,
    num_frames=61,
    num_inference_steps=30,
    guidance_scale=7.0,
    flexcache='teacache',
)
```

### Low Memory

```bash
python test_generate.py \
    models.name=Wan2.1-T2V-14B \
    infer.attn_type=sage \
    infer.diffusion.low_mem_level=3
```

```python
DiffusionUserParams(
    prompt="...",
    height=480,
    width=848,
    num_frames=61,
)
```

## Configuration Best Practices

1. **Start with defaults**: Test with default settings first
2. **Adjust incrementally**: Change one parameter at a time
3. **Monitor resources**: Watch GPU memory and utilization
4. **Profile performance**: Measure impact of each change
5. **Document your settings**: Keep track of what works

## Next Steps

- [Basic Usage](../user-guide/basic-usage.md)
- [Performance Tuning](../user-guide/performance-tuning.md)
- [Advanced Features](../user-guide/advanced-features.md)
