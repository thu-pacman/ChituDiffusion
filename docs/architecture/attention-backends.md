# Attention Backends

ChituDiffusion supports multiple attention backend implementations to optimize performance and memory usage.

## Overview

Attention is the most computationally intensive operation in diffusion models, accounting for 50-80% of total inference time. ChituDiffusion provides multiple attention implementations optimized for different hardware and use cases.

## Available Backends

### FlashAttention

**Type**: Default, FP16/BF16  
**Performance**: Baseline  
**Memory**: Baseline  
**Accuracy**: Full precision

FlashAttention is the default backend, providing accurate and efficient attention computation.

#### Features
- Memory-efficient via tiling
- Fast on modern GPUs (Ampere, Hopper)
- No accuracy loss

#### Configuration
```python
args.infer.attn_type = "flash_attn"
```

#### Requirements
- CUDA compute capability 8.0+
- FlashAttention 2.x installed

### SageAttention

**Type**: Quantized (INT8)  
**Performance**: Testing in progress  
**Memory**: Memory reduction expected  
**Accuracy**: Minimal loss expected

SageAttention uses INT8 quantization for faster computation and lower memory usage.

#### Features
- Performance benchmarking in progress
- Memory reduction testing in progress
- Dynamic quantization per layer
- Quality assessment ongoing

#### Configuration
```python
args.infer.attn_type = "sage"
```

#### Requirements
- SageAttention library installed
- CUDA compute capability 8.0+

#### Use Cases
- Memory-constrained systems
- High-throughput serving
- When slight quality trade-off is acceptable

### SpargeAttention

**Type**: Sparse + Quantized (INT8)  
**Performance**: Testing in progress  
**Memory**: Memory reduction expected  
**Accuracy**: Quality assessment ongoing

SpargeAttention combines sparsity and quantization for maximum performance.

#### Features
- Performance benchmarking in progress
- Memory reduction testing in progress
- Learned sparsity patterns
- Adaptive attention masking

#### Configuration
```python
args.infer.attn_type = "sparge"
```

#### Requirements
- SpargeAttention library installed
- CUDA compute capability 8.0+

#### Use Cases
- Maximum performance needed
- Very memory-constrained systems
- Batch processing scenarios

### Auto Selection

**Type**: Automatic backend selection  
**Behavior**: Chooses best available backend

The `auto` option automatically selects the best available attention backend.

#### Configuration
```python
args.infer.attn_type = "auto"
```

#### Selection Priority
1. SpargeAttention (if installed)
2. SageAttention (if installed)
3. FlashAttention (fallback)

## Performance Comparison

### Speed Benchmark

Performance benchmarking in progress on various hardware configurations.

| Backend | Time per Step | Total Time | Speedup |
|---------|--------------|------------|---------|
| FlashAttention | Baseline | Baseline | 1.0x |
| SageAttention | To be tested | To be tested | To be tested |
| SpargeAttention | To be tested | To be tested | To be tested |

### Memory Benchmark

Memory usage benchmarking in progress.

| Backend | Peak VRAM | Reduction |
|---------|-----------|-----------|
| FlashAttention | Baseline | baseline |
| SageAttention | To be tested | To be tested |
| SpargeAttention | To be tested | To be tested |

### Quality Comparison

Quality metrics testing in progress using VBench benchmark suite.

| Backend | Quality Score | Subject | Motion | Aesthetic |
|---------|--------------|---------|--------|-----------|
| FlashAttention | Baseline | Baseline | Baseline | Baseline |
| SageAttention | To be tested | To be tested | To be tested | To be tested |
| SpargeAttention | To be tested | To be tested | To be tested | To be tested |

## Implementation Details

### Attention Interface

All backends implement a common interface:

```python
class AttentionBackend:
    def forward(self, q, k, v, attn_mask=None):
        """
        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            v: Value tensor [B, H, N, D]
            attn_mask: Optional attention mask
        
        Returns:
            output: Attention output [B, H, N, D]
        """
        pass
```

### Quantization Strategy (Sage/Sparge)

#### Per-Tensor Quantization
```python
scale = max(abs(tensor)) / 127
quantized = (tensor / scale).round().clamp(-128, 127).to(int8)
```

#### Dequantization
```python
dequantized = quantized.to(float16) * scale
```

### Sparsity Pattern (Sparge)

#### Block-Sparse Attention
- Divides attention matrix into blocks (e.g., 64×64)
- Computes importance scores
- Keeps top-k important blocks
- Sets other blocks to zero

#### Dynamic Sparsity
Sparsity ratio adapts per layer based on learned patterns. Exact ratios are being optimized and will be documented.

## Context Parallelism Support

All backends support context parallelism with different communication patterns:

### FlashAttention + CP
```python
# Local attention on chunk
local_out = flash_attn(local_q, local_k, local_v)
# All-to-all for global context
global_out = all_to_all(local_out)
```

### SageAttention + CP
```python
# Quantize locally
local_q_int8 = quantize(local_q)
# Attention with reduced precision
local_out = sage_attn(local_q_int8, local_k_int8, local_v_int8)
# All-to-all (smaller data due to quantization)
global_out = all_to_all(local_out)
```

## Backend Selection Guide

### Choose FlashAttention if:
- ✅ You need maximum quality
- ✅ You have sufficient VRAM (40GB+)
- ✅ You're comparing baselines
- ✅ You're new to the system

### Choose SageAttention if:
- ✅ You need 2x speedup
- ✅ VRAM is limited (20-40GB)
- ✅ Minimal quality loss is acceptable
- ✅ You want balanced performance/quality

### Choose SpargeAttention if:
- ✅ Maximum speed is critical
- ✅ Extreme memory constraints (<20GB)
- ✅ Processing many requests
- ✅ Some quality loss is acceptable

### Choose Auto if:
- ✅ You're unsure which to use
- ✅ You want automatic optimization
- ✅ System may vary (different GPUs)

## Installation

### FlashAttention
```bash
pip install flash-attn --no-build-isolation
```

### SageAttention
```bash
git clone https://github.com/thu-ml/SageAttention
cd SageAttention
pip install -e .
```

### SpargeAttention
```bash
git clone https://github.com/thu-ml/SpargeAttention
cd SpargeAttention
pip install -e .
```

## Troubleshooting

### Import Error
**Symptom**: `ModuleNotFoundError: No module named 'flash_attn'`

**Solution**: Install the required backend:
```bash
pip install flash-attn --no-build-isolation
```

### Compilation Error
**Symptom**: CUDA compilation fails during installation

**Solution**:
1. Check CUDA version: `nvcc --version`
2. Update CUDA toolkit if needed
3. Install with verbose output: `pip install -v`

### Quality Degradation
**Symptom**: Generated videos have artifacts

**Solution**:
1. Switch back to FlashAttention
2. Check quantization calibration
3. Adjust sparsity ratio

### Performance Not Improving
**Symptom**: No speedup with Sage/Sparge

**Solution**:
1. Verify backend is actually being used (check logs)
2. Profile with `torch.profiler`
3. Check GPU utilization with `nvidia-smi`

## Advanced Configuration

### Custom Quantization Range
```python
args.infer.attn_config.quant_bits = 8  # 4, 8 supported
args.infer.attn_config.quant_symmetric = True
```

### Sparsity Control
```python
args.infer.attn_config.sparsity_ratio = 0.5  # 50% sparse
args.infer.attn_config.block_size = 64  # Block size for sparsity
```

### Attention Dropout (training only)
```python
args.infer.attn_config.dropout = 0.0  # Always 0 for inference
```

## Future Work

### Planned Features
- [ ] Flash-Decoding for parallel decoding
- [ ] PagedAttention for memory efficiency
- [ ] Custom Triton kernels
- [ ] INT4 quantization
- [ ] Adaptive sparsity learning

### Research Directions
- Dynamic backend switching per layer
- Hardware-aware attention selection
- Quality-aware sparsity patterns

## See Also

- [Architecture Overview](overview.md)
- [Parallelism](parallelism.md)
- [Performance Tuning](../user-guide/performance-tuning.md)
