# Developer Guide

Welcome to Smart-Diffusion development! This guide will help you contribute to the project.

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA 12.4+
- Git
- Experience with PyTorch and diffusion models

### Development Setup

1. **Fork and Clone**:
```bash
git clone https://github.com/your-username/SmartDiffusion.git
cd SmartDiffusion
```

2. **Install Development Dependencies**:
```bash
# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev,test,docs]"
```

3. **Install Pre-commit Hooks**:
```bash
pre-commit install
```

## Project Structure

```
SmartDiffusion/
├── chitu_core/          # Core framework
│   ├── models/          # Model architectures
│   ├── config/          # Configuration files
│   └── ...
├── chitu_diffusion/     # Diffusion-specific code
│   ├── backend.py       # Backend management
│   ├── generator.py     # Generation pipeline
│   ├── task.py          # Task management
│   └── scheduler.py     # Task scheduling
├── docs/                # Documentation
├── test/                # Tests
└── script/              # Utility scripts
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the coding standards (see [Code Style](code-style.md)).

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific test
pytest test/test_task.py

# With coverage
pytest --cov=chitu_diffusion
```

### 4. Build Documentation

```bash
mkdocs serve
# Visit http://localhost:8000
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvement

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Parameter Taxonomy

**Important**: Understand the three-tier parameter system:

| Category | Life-Cycle | Location | Changeable By |
|----------|-----------|----------|---------------|
| Model params | Static | `config/models/*.yaml` | Nobody |
| User params | Per-request | `DiffusionUserParams` | End user |
| System params | Init-time | Launch args | Operator |

**Guidelines**:
- **Model params**: Tied to checkpoint, changing causes undefined behavior
- **User params**: Per-request flexibility, exposed to end users
- **System params**: Set at initialization, affects all requests

## Adding New Features

### New Model Architecture

1. Create model class in `chitu_core/models/`
2. Register in `ModelType` enum
3. Add config in `config/models/`
4. Add tests
5. Update documentation

See [Custom Models](../advanced/custom-models.md) for details.

### New Attention Backend

1. Implement attention interface
2. Register in `DiffusionAttnBackend`
3. Add configuration option
4. Benchmark performance
5. Document usage

### New Cache Strategy

1. Implement strategy class
2. Register in `FlexCacheManager`
3. Add user parameter option
4. Validate quality impact
5. Document trade-offs

## Testing

### Unit Tests

```python
# test/test_task.py
import pytest
from chitu_diffusion.task import DiffusionUserParams, DiffusionTask

def test_task_creation():
    params = DiffusionUserParams(prompt="test")
    task = DiffusionTask.from_user_request(params)
    assert task.task_id is not None
    assert task.status == TaskStatus.PENDING
```

### Integration Tests

```python
def test_full_generation():
    # Test complete pipeline
    chitu_init(args)
    task = create_test_task()
    DiffusionTaskPool.add(task)
    
    while not DiffusionTaskPool.all_finished():
        chitu_generate()
    
    assert task.status == TaskStatus.FINISHED
```

### Performance Tests

```python
def test_performance():
    import time
    
    start = time.time()
    # ... run generation
    elapsed = time.time() - start
    
    assert elapsed < expected_time
```

## Documentation

### Docstrings

Use Google style:

```python
def function(arg1: str, arg2: int) -> bool:
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When condition occurs
    
    Example:
        >>> function("test", 42)
        True
    """
```

### Markdown Documentation

- Use clear headings
- Include code examples
- Add diagrams where helpful
- Link to related docs

## Code Review

### Before Submitting

- [ ] All tests pass
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Changelog updated

### Review Checklist

Reviewers will check:
- [ ] Code quality and clarity
- [ ] Test coverage
- [ ] Documentation completeness
- [ ] Performance impact
- [ ] Breaking changes noted

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions

## License

Smart-Diffusion is licensed under the MIT License. By contributing, you agree to license your contributions under the same license.

## See Also

- [Code Style](code-style.md)
- [Testing Guide](testing.md)
- [Architecture Overview](../architecture/overview.md)
