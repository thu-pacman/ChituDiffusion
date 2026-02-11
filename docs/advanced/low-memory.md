# Low Memory Mode

Smart-Diffusion provides multiple memory optimization strategies to run diffusion models on GPUs with limited VRAM.

## Overview

Low memory mode uses a multi-level approach to reduce VRAM usage through model offloading, VAE tiling, and mixed precision.

## Memory Levels

Smart-Diffusion implements four memory levels (0-3):

### Level 0: Default (No Optimization)

**VRAM**: Highest (~40-80GB for 14B model)  
**Speed**: Fastest  
**Configuration**: Default

All components stay on GPU:
- Text encoder (T5)
- DiT models
- VAE decoder
- All activations

**Use Case**: When you have abundant VRAM (80GB+ GPUs)

```bash
python test_generate.py infer.diffusion.low_mem_level=0
```

### Level 1: VAE Tiling

**VRAM**: ~30-60GB  
**Speed**: ~5% slower  
**Optimization**: VAE processes video in tiles

**What Changes**:
- VAE decoder operates on tiles instead of full video
- Reduces peak memory during decoding
- Minimal quality impact

**Use Case**: Slightly constrained VRAM (40GB+ GPUs)

```bash
python test_generate.py infer.diffusion.low_mem_level=1
```

### Level 2: CPU Offloading (Text Encoder)

**VRAM**: ~20-40GB  
**Speed**: ~15% slower  
**Optimization**: T5 encoder on CPU

**What Changes**:
- Text encoder moved to CPU
- Embeddings transferred to GPU after encoding
- One-time cost at start

**Use Case**: Limited VRAM (24-40GB)

```bash
python test_generate.py infer.diffusion.low_mem_level=2
```

### Level 3: Aggressive Offloading

**VRAM**: ~15-25GB  
**Speed**: ~40% slower  
**Optimization**: DiT models on CPU, moved to GPU per-layer

**What Changes**:
- DiT model parameters on CPU
- Layers moved to GPU one at a time
- Significant CPU-GPU transfer overhead

**Use Case**: Very limited VRAM (<24GB)

```bash
python test_generate.py infer.diffusion.low_mem_level=3
```

## Memory Comparison

| Level | VRAM (14B) | Speed | Components on GPU |
|-------|------------|-------|-------------------|
| 0 | 40-80GB | 1.0x | All |
| 1 | 30-60GB | 0.95x | All (VAE tiled) |
| 2 | 20-40GB | 0.85x | DiT + VAE |
| 3 | 15-25GB | 0.60x | Active layer only |

## Configuration

### Via Command Line

```bash
# Set memory level
python test_generate.py infer.diffusion.low_mem_level=2
```

### Via Python API

```python
from hydra import compose, initialize

initialize(config_path="config", version_base=None)
args = compose(config_name="wan")

# Set memory level
args.infer.diffusion.low_mem_level = 2

# Initialize with low memory
chitu_init(args)
```

### Via Config File

```yaml
# chitu_core/config/infer.yaml
diffusion:
  low_mem_level: 2
```

## Memory Optimization Strategies

### 1. Model Offloading

Move unused components to CPU:

```python
# Automatic based on low_mem_level
if low_mem_level >= 2:
    text_encoder.to("cpu")
    
if low_mem_level >= 3:
    for model in model_pool:
        model.to("cpu")
```

### 2. VAE Tiling

Process video in spatial tiles:

```python
def decode_vae_tiled(latent, tile_size=8):
    """Decode VAE in tiles to reduce memory"""
    B, C, T, H, W = latent.shape
    
    # Split into tiles
    tiles = []
    for h in range(0, H, tile_size):
        for w in range(0, W, tile_size):
            tile = latent[:, :, :, h:h+tile_size, w:w+tile_size]
            tiles.append(tile)
    
    # Decode each tile
    decoded_tiles = [vae.decode(tile) for tile in tiles]
    
    # Merge tiles
    video = merge_tiles(decoded_tiles, (H, W))
    return video
```

### 3. Gradient Checkpointing

Recompute activations instead of storing:

```python
# Enable for training/fine-tuning
model.enable_gradient_checkpointing()
```

### 4. Mixed Precision

Use FP16/BF16 instead of FP32:

```python
with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    output = model(input)
```

### 5. Activation Offloading

Move activations to CPU between layers:

```python
# Only for level 3
if low_mem_level >= 3:
    for layer in model.layers:
        output = layer(input.to("cuda"))
        output = output.to("cpu")
```

## Advanced Configuration

### Custom Tile Size

```python
# Smaller tiles = less memory, more compute
args.infer.vae_tile_size = 4  # Default: 8
```

### Offloading Strategy

```python
# Fine-tune what to offload
args.infer.offload_text_encoder = True
args.infer.offload_vae = False  # Keep VAE on GPU
args.infer.offload_dit = False
```

## Combining with Other Optimizations

### Low Memory + SageAttention

Combine memory level with quantized attention:

```bash
python test_generate.py \
    infer.diffusion.low_mem_level=2 \
    infer.attn_type=sage
```

**VRAM**: ~15-30GB  
**Speed**: ~0.8x compared to baseline

### Low Memory + FlexCache

Enable caching to offset speed loss:

```bash
python test_generate.py \
    infer.diffusion.low_mem_level=2
```

**VRAM**: ~20-40GB  
**Speed**: ~0.9x compared to baseline

```python
params = DiffusionUserParams(
    prompt="A cat",
    flexcache="teacache"
)
```

### Low Memory + Context Parallelism

Split across GPUs for even lower per-GPU memory:

```bash
torchrun --nproc_per_node=2 test_generate.py \
    infer.diffusion.low_mem_level=2 \
    infer.diffusion.cp_size=2
```

**VRAM per GPU**: ~10-20GB  
**Speed**: ~1.7x (2 GPUs)

## Monitoring Memory Usage

### Real-Time Monitoring

```bash
# In another terminal
watch -n 0.5 nvidia-smi
```

### Python Monitoring

```python
import torch

def print_memory_stats():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
    print(f"Peak: {max_allocated:.2f} GB")

# Call during generation
print_memory_stats()
```

### Memory Profiling

```python
import torch.cuda.memory

# Start profiling
torch.cuda.memory._record_memory_history()

# Run generation
chitu_generate()

# Export profile
torch.cuda.memory._dump_snapshot("memory_profile.pickle")
```

## Troubleshooting

### Still Running Out of Memory

**Solutions**:
1. Increase `low_mem_level`
2. Reduce resolution or num_frames
3. Use SageAttention
4. Enable context parallelism
5. Close other GPU processes

```python
# Emergency memory clearing
torch.cuda.empty_cache()
```

### Slow Generation

**Solutions**:
1. Lower `low_mem_level` if you have VRAM
2. Enable FlexCache
3. Use SageAttention
4. Check CPU-GPU transfer is not bottleneck

```python
# Profile to find bottleneck
with torch.profiler.profile() as prof:
    chitu_generate()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### CPU Bottleneck (Level 3)

**Symptoms**: High CPU usage, GPU underutilized

**Solutions**:
1. Use faster CPU
2. Enable pinned memory
3. Increase batch size (if supported)
4. Use NVMe SSD for faster disk I/O

## Best Practices

### 1. Start Conservative

Begin with level 2 and adjust:
```python
args.infer.diffusion.low_mem_level = 2
```

### 2. Profile First

Measure actual VRAM usage before optimizing:
```python
print_memory_stats()
```

### 3. Balance Memory and Speed

Find optimal level for your use case:
- Interactive: Level 1-2
- Batch processing: Level 2-3
- Production: Level 1 with monitoring

### 4. Combine Techniques

Use multiple optimizations together:
- Low memory + quantized attention
- Low memory + caching
- Low memory + parallelism

### 5. Monitor in Production

Track memory usage over time:
```python
# Log peak memory per request
max_mem = torch.cuda.max_memory_allocated() / 1024**3
logging.info(f"Peak memory: {max_mem:.2f} GB")
torch.cuda.reset_peak_memory_stats()
```

## Hardware Recommendations

### By VRAM Size

| VRAM | Recommended Level | Max Resolution |
|------|------------------|----------------|
| 16GB | Level 3 + Sage | 480x848, 49 frames |
| 24GB | Level 2 | 480x848, 81 frames |
| 40GB | Level 1 | 720x1280, 81 frames |
| 80GB | Level 0 | 720x1280, 121 frames |

### By Use Case

| Use Case | Configuration |
|----------|--------------|
| Development/Testing | Level 2, low resolution |
| Production (latency-sensitive) | Level 1, SageAttention |
| Production (throughput-focused) | Level 2, batch processing |
| Research/Experimentation | Level 0, full precision |

## See Also

- [FlexCache](flexcache.md) - Feature caching
- [Attention Backends](../architecture/attention-backends.md) - Quantized attention
- [Multi-GPU Setup](../user-guide/multi-gpu.md) - Context parallelism
- [Performance Tuning](../user-guide/performance-tuning.md)
