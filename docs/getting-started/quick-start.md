# Quick Start

This guide will help you generate your first video with ChituDiffusion in just a few minutes.

## Prerequisites

Before starting, make sure you have:

1. [Installed ChituDiffusion](installation.md)
2. Downloaded a model checkpoint (see [Model Downloads](#model-downloads))

## Model Downloads

ChituDiffusion currently supports the Wan-T2V series models:

| Model | Size | Download |
|-------|------|----------|
| Wan2.1-T2V-1.3B | 1.3B | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.1-T2V-14B | 14B | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.2-T2V-A14B | 14B | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) |

Download the model checkpoint to a local directory, e.g., `/path/to/Wan2.1-T2V-1.3B`.

## Basic Generation

### Step 1: Create a Test Script

Create a file named `test_generate.py`:

```python
from chitu_diffusion import chitu_init, chitu_generate, chitu_start, chitu_terminate
from chitu_diffusion.task import DiffusionUserParams, DiffusionTask, DiffusionTaskPool
from hydra import compose, initialize

# Initialize configuration
initialize(config_path="config", version_base=None)
args = compose(config_name="wan")

# Set model checkpoint path
args.models.ckpt_dir = "/path/to/Wan2.1-T2V-1.3B"

# Initialize the backend
chitu_init(args)
chitu_start()

# Create a generation task
user_params = DiffusionUserParams(
    role="user1",
    prompt="A cat walking on grass.",
    num_inference_steps=50,
    height=480,
    width=848,
    num_frames=81,
    guidance_scale=7.0,
)

# Add task to pool
task = DiffusionTask.from_user_request(user_params)
DiffusionTaskPool.add(task)

# Generate until completion
while not DiffusionTaskPool.all_finished():
    chitu_generate()

# Terminate backend
chitu_terminate()

print(f"✅ Video saved to: {task.buffer.save_path}")
```

### Step 2: Run the Script

**Single GPU:**

```bash
bash run.sh system_config.yaml --num-nodes 1 --gpus-per-node 1 --cfp 1
```

**Multi-GPU (Single Node):**

```bash
bash run.sh system_config.yaml --num-nodes 1 --gpus-per-node 2 --cfp 2
```

**Multi-Node SLURM:**

```bash
bash run.sh system_config.yaml --num-nodes 2 --gpus-per-node 2 --cfp 2  # 4 GPUs
```

### Step 3: View the Output

The generated video will be saved to:
```
./outputs/<timestamp>_<task_id>.mp4
```

## Parameter Customization

### Adjust Video Properties

```python
user_params = DiffusionUserParams(
    prompt="A beautiful sunset over the ocean",
    height=720,          # Video height in pixels
    width=1280,          # Video width in pixels
    num_frames=121,      # Number of frames (higher = longer video)
    fps=24,              # Frames per second
)
```

### Control Generation Quality

```python
user_params = DiffusionUserParams(
    prompt="A dog playing in the park",
    num_inference_steps=50,  # More steps = better quality (slower)
    guidance_scale=7.0,      # Higher = more prompt adherence
)
```

### Set Output Path

```python
user_params = DiffusionUserParams(
    prompt="A spaceship landing on Mars",
    save_path="./my_videos/mars_landing.mp4",
)
```

## Using Different Attention Backends

### SageAttention (Faster, Quantized)

```bash
python test_generate.py \
    models.ckpt_dir=/path/to/checkpoint \
    infer.attn_type=sage
```

### SpargeAttention (Fastest, Sparse)

```bash
python test_generate.py \
    models.ckpt_dir=/path/to/checkpoint \
    infer.attn_type=sparge
```

## Low Memory Mode

If you encounter Out-of-Memory errors:

```bash
python test_generate.py \
    models.ckpt_dir=/path/to/checkpoint \
    infer.diffusion.low_mem_level=2
```

Memory levels:
- **0**: All models on GPU (highest performance)
- **1**: VAE uses tiling
- **2**: T5 encoder on CPU (recommended for 24GB VRAM)
- **3+**: DiT model on CPU (slowest but works on limited VRAM)

## Batch Generation

Generate multiple videos:

```python
prompts = [
    "A cat walking on grass",
    "A dog playing in the park",
    "A bird flying in the sky",
]

for i, prompt in enumerate(prompts):
    user_params = DiffusionUserParams(
        role=f"user{i}",
        prompt=prompt,
        save_path=f"./outputs/video_{i}.mp4",
    )
    task = DiffusionTask.from_user_request(user_params)
    DiffusionTaskPool.add(task)

# Generate all tasks
while not DiffusionTaskPool.all_finished():
    chitu_generate()
```

## Example Outputs

Here are some example generations with different prompts:

### Example 1: Nature Scene
```python
prompt = "A serene mountain lake at sunrise, mist rising from the water"
# Resolution: 1280x720, 121 frames, 24 fps
```

### Example 2: Urban Scene
```python
prompt = "A busy city street at night, neon lights reflecting on wet pavement"
# Resolution: 848x480, 81 frames, 24 fps
```

### Example 3: Abstract
```python
prompt = "Colorful paint swirling and mixing in slow motion"
# Resolution: 720x720, 61 frames, 30 fps
```

## Common Issues

### Issue: Model Not Found

**Error**: `FileNotFoundError: No checkpoint files found`

**Solution**: Verify the checkpoint path is correct:
```bash
ls /path/to/checkpoint/diffusion_pytorch_model.safetensors
```

### Issue: Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Use lower resolution: `height=480, width=848`
2. Enable low memory mode: `infer.diffusion.low_mem_level=2`
3. Reduce batch size or frames: `num_frames=61`

### Issue: Slow Generation

**Solutions**:
1. Use SageAttention: `infer.attn_type=sage`
2. Reduce inference steps: `num_inference_steps=30`
3. Enable FlexCache: `flexcache='teacache'`

## Next Steps

Now that you've generated your first video, explore:

- [Advanced Features](../user-guide/advanced-features.md) - FlexCache, CFG parallelism, etc.
- [Performance Tuning](../user-guide/performance-tuning.md) - Optimize for speed
- [Multi-GPU Setup](../user-guide/multi-gpu.md) - Scale to multiple GPUs
- [API Reference](../api/core.md) - Detailed API documentation

## Getting Help

Need help?

- Check the [FAQ](../faq.md)
- Read the [User Guide](../user-guide/basic-usage.md)
- Ask in [GitHub Discussions](https://github.com/chen-yy20/SmartDiffusion/discussions)
- Report bugs in [GitHub Issues](https://github.com/chen-yy20/SmartDiffusion/issues)
