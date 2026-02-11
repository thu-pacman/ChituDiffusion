# Frequently Asked Questions (FAQ)

## Installation & Setup

### Q: What GPU do I need to run Smart-Diffusion?

**A**: Smart-Diffusion requires an NVIDIA GPU with:
- Compute capability 8.0+ (Ampere architecture or newer)
- At least 24GB VRAM for 14B models (can use low memory mode with less)
- CUDA 12.4 or higher

Supported GPUs:
- A100, A10, A30 (Ampere)
- H100, H20 (Hopper)
- B100, B200, RTX 5090 (Blackwell)

### Q: Can I run Smart-Diffusion on AMD or Intel GPUs?

**A**: Currently, Smart-Diffusion only supports NVIDIA GPUs with CUDA. Support for ROCm (AMD) and other platforms may be added in the future.

### Q: Do I need to compile from source?

**A**: Not necessarily. Basic installation with FlashAttention can use pre-built wheels. However, for optimal performance with SageAttention and SpargeAttention, compilation is recommended.

### Q: How long does installation take?

**A**: 
- Basic installation (pip/uv): 5-10 minutes
- Full installation with compilation: 10-20 minutes
- Depends on your internet speed and CPU cores

### Q: What if I encounter CUDA version mismatches?

**A**: 
1. Check your CUDA version: `nvcc --version`
2. Update `pyproject.toml` to match your CUDA version
3. For uv: `uv sync -v --reinstall`
4. For pip: `pip install --force-reinstall torch torchvision`

## Model & Inference

### Q: Which model should I use?

**A**: 
- **Wan2.1-T2V-1.3B**: Fast, good for testing and lower-end GPUs
- **Wan2.1-T2V-14B**: Best quality, requires more VRAM
- **Wan2.2-T2V-A14B**: Advanced two-stage model with highest quality

### Q: How much VRAM do I need?

**A**: Approximate VRAM requirements:

| Model | Resolution | Frames | Min VRAM | Recommended |
|-------|------------|--------|----------|-------------|
| 1.3B | 480x848 | 81 | 16GB | 24GB |
| 14B | 480x848 | 81 | 24GB | 40GB |
| 14B | 720x1280 | 121 | 40GB | 80GB |

Use `low_mem_level` to reduce VRAM usage:
```bash
python test_generate.py infer.diffusion.low_mem_level=2
```

### Q: How can I speed up generation?

**A**: Several strategies:

1. **Use SageAttention**: `infer.attn_type=sage` (~2x faster)
2. **Reduce inference steps**: `num_inference_steps=30` (was 50)
3. **Enable FlexCache**: `flexcache='teacache'`
4. **Lower resolution**: `height=480, width=848`
5. **Use multiple GPUs**: Enable context parallelism

### Q: What's the difference between attention backends?

**A**:
- **flash_attn**: Default, accurate, good performance
- **sage**: Quantized (INT8), ~2x faster, minimal quality loss
- **sparge**: Sparse + quantized, ~3x faster, slight quality loss
- **auto**: Automatically select best available

### Q: Why is the first generation slow?

**A**: The first generation includes:
- Model loading
- CUDA kernel compilation
- Memory allocation

Subsequent generations will be much faster. Use warmup if needed.

## Configuration

### Q: How do I set the output path?

**A**: Set `save_path` in `DiffusionUserParams`:

```python
user_params = DiffusionUserParams(
    prompt="...",
    save_path="./my_videos/output.mp4"
)
```

### Q: Can I generate multiple videos at once?

**A**: Yes! Add multiple tasks to the pool:

```python
for prompt in prompts:
    task = DiffusionTask.from_user_request(
        DiffusionUserParams(prompt=prompt)
    )
    DiffusionTaskPool.add(task)

while not DiffusionTaskPool.all_finished():
    chitu_generate()
```

### Q: How do I adjust video quality?

**A**: Key parameters:
- `num_inference_steps`: More steps = better quality (30-100)
- `guidance_scale`: Higher = more prompt adherence (5.0-15.0)
- `resolution`: Higher resolution = better detail

### Q: What does `guidance_scale` do?

**A**: Classifier-Free Guidance (CFG) scale controls how closely the model follows your prompt:
- **Low (3-5)**: More creative, less prompt adherence
- **Medium (7-9)**: Balanced (recommended)
- **High (10-15)**: Strict prompt following, may reduce creativity

## Memory & Performance

### Q: I'm getting "CUDA out of memory" errors

**A**: Try these solutions in order:

1. Enable low memory mode:
   ```bash
   infer.diffusion.low_mem_level=2
   ```

2. Reduce resolution:
   ```python
   height=480, width=848, num_frames=61
   ```

3. Use SageAttention (lower memory):
   ```bash
   infer.attn_type=sage
   ```

4. Close other programs using GPU

### Q: What is `low_mem_level`?

**A**:
- **Level 0**: All models on GPU (fastest, most VRAM)
- **Level 1**: VAE uses tiling (slight slowdown)
- **Level 2**: T5 encoder on CPU (moderate slowdown)
- **Level 3**: DiT model on CPU (significant slowdown)

### Q: How do I check VRAM usage?

**A**: During generation, Smart-Diffusion logs memory usage. You can also use:

```bash
# In another terminal
nvidia-smi -l 1  # Updates every second
```

Or in Python:
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

## Multi-GPU & Distributed

### Q: How do I use multiple GPUs?

**A**: 

**Data parallelism (future)**:
```bash
torchrun --nproc_per_node=4 test_generate.py
```

**Context parallelism**:
```bash
python test_generate.py infer.diffusion.cp_size=2
```

**CFG parallelism** (automatic if world_size >= 2 and CFG enabled)

### Q: What's the difference between CP and CFG parallelism?

**A**:
- **Context Parallelism (CP)**: Splits frames across GPUs for longer videos
- **CFG Parallelism**: Splits positive/negative prompts across 2 GPUs for faster generation

### Q: Can I combine different parallelism strategies?

**A**: Yes! For example, 4 GPUs can use:
- 2 GPUs for CFG parallelism
- 2 GPUs for context parallelism

```bash
# 4 GPUs total = 2 CFG × 2 CP
python test_generate.py infer.diffusion.cfg_size=2 infer.diffusion.cp_size=2
```

## Troubleshooting

### Q: Model loading is very slow

**A**: Possible causes:
1. **Slow disk**: Use SSD for model storage
2. **Network storage**: Copy model to local disk
3. **Large model**: 14B models take time to load

### Q: Generation stops or hangs

**A**: Check:
1. CUDA errors in logs
2. Out of memory (check `nvidia-smi`)
3. Network issues (for distributed training)
4. Ctrl+C may be caught - wait or force kill

### Q: Video quality is poor

**A**: Try:
1. Increase `num_inference_steps` (50-100)
2. Adjust `guidance_scale` (7-10)
3. Use higher resolution
4. Try different prompt phrasing
5. Use 14B model instead of 1.3B

### Q: Colors look wrong in generated video

**A**: This may be a VAE decoding issue. Try:
1. Update to latest version
2. Check if issue persists with different prompts
3. Report as a bug with sample output

## Advanced Topics

### Q: How does FlexCache work?

**A**: FlexCache reuses features from previous denoising steps:
- **TeaCache**: Reuses features when they change slowly
- **PAB**: Broadcasts high-level features to lower levels

Enables ~30-50% speedup with minimal quality loss.

### Q: Can I fine-tune models with Smart-Diffusion?

**A**: Smart-Diffusion is designed for inference only. Fine-tuning should be done with training frameworks, then load checkpoints for inference.

### Q: How do I add a custom model?

**A**: See [Custom Models](advanced/custom-models.md) guide. Summary:
1. Create model class implementing required interface
2. Register model type
3. Create configuration file
4. Load checkpoint

### Q: What's the difference between Chitu and Smart-Diffusion?

**A**:
- **Chitu**: General LLM inference framework
- **Smart-Diffusion**: Specialized for diffusion models (DiT), built on Chitu

## Contributing

### Q: How can I contribute?

**A**: See [Contributing Guide](contributing/developer-guide.md):
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation
- Share usage examples

### Q: What's the development roadmap?

**A**: Key priorities:
1. More model support (SD3, Flux, CogVideoX)
2. Better performance optimizations
3. Production serving features
4. Comprehensive benchmarks

## Getting More Help

### Q: Where can I get help not covered in FAQ?

**A**:
1. **Documentation**: Check [User Guide](user-guide/basic-usage.md)
2. **GitHub Issues**: [Report bugs](https://github.com/chen-yy20/SmartDiffusion/issues)
3. **Discussions**: [Ask questions](https://github.com/chen-yy20/SmartDiffusion/discussions)

### Q: How do I report a bug?

**A**: Open an issue with:
- System info (GPU, CUDA version, Python version)
- Installation method
- Command/code that caused the error
- Full error message
- Expected vs actual behavior

### Q: Is there a community chat?

**A**: We're considering setting up Discord/Slack. For now, use:
- GitHub Discussions for questions
- GitHub Issues for bugs
- Pull requests for contributions
