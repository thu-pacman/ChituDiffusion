# Basic Usage

This guide covers the fundamental usage patterns of Smart-Diffusion.

## Prerequisites

- [Installed Smart-Diffusion](../getting-started/installation.md)
- Downloaded model checkpoint
- Basic Python knowledge

## Your First Generation

### Step 1: Import Required Modules

```python
from chitu_diffusion import chitu_init, chitu_generate, chitu_start, chitu_terminate
from chitu_diffusion.task import DiffusionUserParams, DiffusionTask, DiffusionTaskPool
from hydra import compose, initialize
```

### Step 2: Initialize Configuration

```python
# Load configuration
initialize(config_path="config", version_base=None)
args = compose(config_name="wan")

# Set checkpoint path
args.models.ckpt_dir = "/path/to/checkpoint"
```

### Step 3: Initialize Backend

```python
# Initialize the system
chitu_init(args)
chitu_start()
```

### Step 4: Create Generation Task

```python
# Define generation parameters
user_params = DiffusionUserParams(
    role="user1",
    prompt="A cat walking on grass",
    num_inference_steps=50,
    height=480,
    width=848,
    num_frames=81,
)

# Create and add task
task = DiffusionTask.from_user_request(user_params)
DiffusionTaskPool.add(task)
```

### Step 5: Generate

```python
# Run generation loop
while not DiffusionTaskPool.all_finished():
    chitu_generate()

# Cleanup
chitu_terminate()

print(f"Video saved to: {task.buffer.save_path}")
```

## Common Use Cases

### Batch Generation

Generate multiple videos in sequence:

```python
prompts = [
    "A cat walking on grass",
    "A dog playing in the park",
    "A bird flying in the sky"
]

for i, prompt in enumerate(prompts):
    params = DiffusionUserParams(
        role=f"user{i}",
        prompt=prompt,
        save_path=f"./outputs/video_{i}.mp4"
    )
    task = DiffusionTask.from_user_request(params)
    DiffusionTaskPool.add(task)

while not DiffusionTaskPool.all_finished():
    chitu_generate()
```

### Customizing Video Properties

```python
user_params = DiffusionUserParams(
    prompt="Your prompt here",
    # Video dimensions
    height=720,          # Height in pixels
    width=1280,          # Width in pixels
    num_frames=121,      # Number of frames
    fps=24,              # Frames per second
    
    # Quality settings
    num_inference_steps=50,    # More steps = better quality
    guidance_scale=7.0,        # Prompt adherence (5-15)
    
    # Output
    save_path="./my_video.mp4"
)
```

## Understanding the Pipeline

The generation pipeline consists of three main stages:

### 1. Text Encoding

Converts your text prompt into embeddings using T5 encoder:

```
"A cat walking on grass" → [embedding tensor]
```

### 2. Iterative Denoising

Progressively denoises random noise into structured latent:

```
[random noise] → [denoised latent] (50 steps)
```

### 3. VAE Decoding

Converts latent to pixel space:

```
[latent] → [video frames]
```

## Parameter Reference

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | str | Text description of desired video |

### Video Properties

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `height` | int | 480 | Video height in pixels |
| `width` | int | 848 | Video width in pixels |
| `num_frames` | int | 81 | Number of frames to generate |
| `fps` | int | 24 | Frames per second |

### Quality Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_inference_steps` | int | 50 | Number of denoising steps |
| `guidance_scale` | float | 7.0 | CFG scale (higher = more prompt adherence) |
| `seed` | int | None | Random seed (None = random) |

### Advanced Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `role` | str | "user" | User identifier |
| `save_path` | str | None | Output path (None = auto-generate) |
| `flexcache` | str | None | Cache strategy ('teacache', 'PAB', None) |

## Best Practices

### 1. Start Simple

Begin with default parameters:

```python
user_params = DiffusionUserParams(
    prompt="Your prompt"
)
```

Then adjust as needed.

### 2. Iterate on Prompts

Test different prompt phrasings to find what works best:

```python
prompts = [
    "A cat walking",
    "A cat strolling through grass",
    "A fluffy cat walking on green grass"
]
```

### 3. Balance Quality and Speed

- **For testing**: Use lower steps (30) and resolution (480x848)
- **For final output**: Use higher steps (50-100) and resolution (720x1280)

### 4. Monitor Resources

Watch GPU memory usage:

```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

### 5. Use Appropriate Models

- **1.3B model**: Faster, good for testing
- **14B model**: Better quality, requires more VRAM

## Troubleshooting

### Generation is Slow

**Solutions**:
1. Use SageAttention: `infer.attn_type=sage`
2. Reduce steps: `num_inference_steps=30`
3. Lower resolution: `height=480, width=848`

### Out of Memory

**Solutions**:
1. Enable low memory mode: `infer.diffusion.low_mem_level=2`
2. Reduce frames: `num_frames=61`
3. Use smaller model: Wan2.1-T2V-1.3B

### Poor Quality

**Solutions**:
1. Increase steps: `num_inference_steps=100`
2. Adjust guidance scale: `guidance_scale=9.0`
3. Improve prompt phrasing
4. Use 14B model

## Next Steps

- [Advanced Features](advanced-features.md) - Explore FlexCache, CFG parallelism
- [Performance Tuning](performance-tuning.md) - Optimize for speed
- [Multi-GPU Setup](multi-gpu.md) - Scale to multiple GPUs
