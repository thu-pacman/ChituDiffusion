# Custom Models

This guide explains how to add support for custom diffusion models in Smart-Diffusion.

## Overview

Smart-Diffusion's modular architecture allows you to add new model architectures with minimal changes.

## Model Requirements

Your custom model must:
1. Be a PyTorch `nn.Module`
2. Implement the diffusion forward interface
3. Support the required input/output formats
4. Provide a configuration file

## Step-by-Step Guide

### 1. Create Model Class

Create your model in `chitu_core/models/your_model.py`:

```python
import torch
import torch.nn as nn

class YourDiffusionModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        **kwargs
    ):
        super().__init__()
        # Initialize your architecture
        
    def forward(
        self,
        x: torch.Tensor,  # [B, C, T, H, W]
        text_embeddings: torch.Tensor,  # [B, seq_len, D]
        timesteps: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Args:
            x: Latent input
            text_embeddings: Text condition
            timesteps: Diffusion timesteps
            
        Returns:
            noise_pred: Predicted noise
        """
        # Your model logic
        return noise_pred
```

### 2. Register Model Type

Add to `chitu_core/models/__init__.py`:

```python
from enum import Enum

class ModelType(Enum):
    DIFF_WAN = "diff-wan"
    YOUR_MODEL = "your-model"  # Add your type

def get_model_class(model_type: ModelType):
    if model_type == ModelType.DIFF_WAN:
        from .wan import WanDiffusionModel
        return WanDiffusionModel
    elif model_type == ModelType.YOUR_MODEL:
        from .your_model import YourDiffusionModel
        return YourDiffusionModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### 3. Create Configuration File

Create `chitu_core/config/models/your_model.yaml`:

```yaml
name: YourModel-1B
type: your-model
ckpt_dir: null  # Set at runtime
source: https://huggingface.co/your-org/your-model

transformer:
  in_channels: 16
  patch_size: [1, 2, 2]
  hidden_size: 2048
  depth: 24
  num_heads: 16
  mlp_ratio: 4.0
  
  # Your custom parameters
  custom_param_1: value1
  custom_param_2: value2
```

### 4. Update Backend (if needed)

If your model requires special loading logic, update `chitu_diffusion/backend.py`:

```python
def _build_model_architecture(args, attn_backend, rope_impl):
    model_type = ModelType(args.type)
    
    if model_type == ModelType.YOUR_MODEL:
        # Custom loading logic
        model = YourDiffusionModel(
            in_channels=args.transformer.in_channels,
            # ... your parameters
            custom_param=args.transformer.custom_param_1
        )
    else:
        model_cls = get_model_class(model_type)
        model = model_cls(...)
    
    return model
```

### 5. Test Your Model

```python
from hydra import compose, initialize

initialize(config_path="config", version_base=None)
args = compose(config_name="your_model")
args.models.ckpt_dir = "/path/to/checkpoint"

chitu_init(args)
# Test generation
```

## Advanced Customization

### Custom Text Encoder

```python
class YourTextEncoder(nn.Module):
    def forward(self, input_ids, attention_mask):
        # Your encoding logic
        return embeddings
```

Register in backend:
```python
if args.models.text_encoder == "your-encoder":
    backend.text_encoder = YourTextEncoder()
```

### Custom VAE

```python
class YourVAE(nn.Module):
    def decode(self, latent):
        # Your decoding logic
        return video
```

### Custom Attention

```python
class YourAttention(nn.Module):
    def forward(self, q, k, v):
        # Your attention logic
        return output
```

Register in attention backend:
```python
if attn_type == "your-attn":
    attn_impl = YourAttention()
```

## Checkpoint Conversion

If your checkpoint format differs, create a conversion script:

```python
# convert_checkpoint.py
import torch

def convert_checkpoint(original_path, output_path):
    """Convert your checkpoint to Smart-Diffusion format"""
    # Load original
    ckpt = torch.load(original_path)
    
    # Convert keys
    new_ckpt = {}
    for key, value in ckpt.items():
        new_key = convert_key_name(key)
        new_ckpt[new_key] = value
    
    # Save
    torch.save(new_ckpt, output_path)
```

## Best Practices

1. **Follow Existing Patterns**: Look at `WanDiffusionModel` as reference
2. **Document Parameters**: Add docstrings for all config options
3. **Test Thoroughly**: Validate outputs match expected behavior
4. **Profile Performance**: Ensure no unexpected bottlenecks
5. **Add Unit Tests**: Test forward pass, shapes, etc.

## Example: Adding Stable Diffusion 3

```python
# chitu_core/models/sd3.py
class SD3DiffusionModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # SD3 architecture
        
    def forward(self, x, text_embeddings, timesteps):
        # SD3 forward logic
        return noise_pred
```

```yaml
# chitu_core/config/models/sd3.yaml
name: StableDiffusion3
type: sd3
ckpt_dir: null

transformer:
  in_channels: 16
  hidden_size: 1536
  depth: 24
  # ... SD3 specific params
```

## Troubleshooting

### Shape Mismatches

Check tensor shapes at each step:
```python
print(f"Input shape: {x.shape}")
print(f"Expected: [B, C, T, H, W]")
```

### Loading Errors

Verify checkpoint keys:
```python
ckpt = torch.load("checkpoint.bin")
print("Checkpoint keys:", ckpt.keys())
```

### Performance Issues

Profile your model:
```python
with torch.profiler.profile() as prof:
    output = model(x, embeddings, timesteps)
print(prof.key_averages().table())
```

## See Also

- [Architecture Overview](../architecture/overview.md)
- [Backend API](../api/backend.md)
- [Contributing Guide](../contributing/developer-guide.md)
