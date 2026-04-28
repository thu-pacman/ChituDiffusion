# Code Style Guide

This document outlines the coding standards for ChituDiffusion.

## Python Style

### PEP 8 Compliance

Follow [PEP 8](https://pep8.org/) with these specifics:

- **Line Length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized with isort

### Formatting Tools

We use automated formatters:

```bash
# Format code
black .

# Sort imports
isort .

# Check style
flake8

# Type checking
mypy chitu_diffusion/
```

### Import Organization

```python
# Standard library
import os
import sys
from typing import Optional, List

# Third-party
import torch
import torch.nn as nn
import numpy as np

# Local
from chitu_core.models import ModelType
from chitu_diffusion.task import DiffusionTask
```

### Naming Conventions

```python
# Classes: PascalCase
class DiffusionBackend:
    pass

# Functions/methods: snake_case
def encode_text(prompt: str):
    pass

# Constants: UPPER_SNAKE_CASE
MAX_SEQUENCE_LENGTH = 512

# Private members: _leading_underscore
def _internal_helper():
    pass
```

## Type Hints

Use type hints for all public functions:

```python
def create_task(
    prompt: str,
    num_steps: int = 50,
    seed: Optional[int] = None
) -> DiffusionTask:
    """Create a generation task"""
    pass
```

## Docstrings

Use Google style docstrings:

```python
def generate_video(
    prompt: str,
    height: int = 480,
    width: int = 848
) -> torch.Tensor:
    """
    Generate a video from a text prompt.
    
    Args:
        prompt: Text description of the desired video
        height: Video height in pixels
        width: Video width in pixels
    
    Returns:
        video: Generated video tensor of shape [T, C, H, W]
    
    Raises:
        ValueError: If dimensions are invalid
        RuntimeError: If generation fails
    
    Example:
        >>> video = generate_video("A cat walking")
        >>> print(video.shape)
        torch.Size([81, 3, 480, 848])
    """
    pass
```

## Error Handling

### Use Specific Exceptions

```python
# Good
if height <= 0:
    raise ValueError(f"Height must be positive, got {height}")

# Bad
if height <= 0:
    raise Exception("Invalid height")
```

### Context Managers

```python
# Use context managers for resources
with torch.inference_mode():
    output = model(input)
```

## Best Practices

### 1. Fail Fast

```python
# Validate early
def process_task(task: DiffusionTask):
    if task is None:
        raise ValueError("Task cannot be None")
    
    if task.status != TaskStatus.PENDING:
        raise ValueError(f"Task must be pending, got {task.status}")
    
    # ... proceed with processing
```

### 2. Avoid Magic Numbers

```python
# Good
DEFAULT_NUM_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.0

num_steps = DEFAULT_NUM_STEPS

# Bad
num_steps = 50  # What does 50 mean?
```

### 3. Use Dataclasses

```python
from dataclasses import dataclass

@dataclass
class Config:
    num_steps: int = 50
    guidance_scale: float = 7.0
```

### 4. Keep Functions Small

```python
# Break large functions into smaller ones
def generate():
    embeddings = encode_text()
    latent = denoise_loop()
    video = decode_vae()
    return video
```

### 5. Document Complex Logic

```python
# Explain non-obvious code
# We use CFG parallelism here to split positive/negative prompts
# across 2 GPUs for 2x speedup
if cfg_size == 2:
    split_prompts()
```

## Testing Style

```python
# Clear test names
def test_task_creation_with_valid_params():
    params = DiffusionUserParams(prompt="test")
    task = DiffusionTask.from_user_request(params)
    assert task.task_id is not None

# Use fixtures
@pytest.fixture
def sample_task():
    params = DiffusionUserParams(prompt="test")
    return DiffusionTask.from_user_request(params)

def test_with_fixture(sample_task):
    assert sample_task.status == TaskStatus.PENDING
```

## Configuration Files

### YAML Style

```yaml
# Use lowercase with underscores
model_config:
  hidden_size: 3072
  num_heads: 24
  
# Comments explain purpose
# This controls memory/speed trade-off
low_mem_level: 2
```

## Git Commits

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance
- `style`: Formatting

**Examples**:
```
feat(backend): add support for SageAttention

Implements INT8 quantized attention backend for 2x speedup
with minimal quality loss.

Closes #123
```

```
fix(task): handle empty prompt gracefully

Previously crashed with empty string, now uses default prompt.
```

## See Also

- [Developer Guide](developer-guide.md)
- [Testing Guide](testing.md)
- [PEP 8](https://pep8.org/)
- [Black](https://black.readthedocs.io/)
