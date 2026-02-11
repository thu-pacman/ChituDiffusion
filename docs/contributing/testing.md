# Testing Guide

This guide covers testing practices for Smart-Diffusion.

## Test Organization

```
test/
├── unit/              # Unit tests
│   ├── test_task.py
│   ├── test_scheduler.py
│   └── test_backend.py
├── integration/       # Integration tests
│   ├── test_generation_pipeline.py
│   └── test_distributed.py
├── performance/       # Performance tests
│   └── test_benchmarks.py
└── conftest.py       # Pytest fixtures
```

## Running Tests

### All Tests

```bash
pytest
```

### Specific Tests

```bash
# Single file
pytest test/test_task.py

# Single test
pytest test/test_task.py::test_task_creation

# By marker
pytest -m unit
pytest -m integration
```

### With Coverage

```bash
pytest --cov=chitu_diffusion --cov-report=html
```

## Writing Tests

### Unit Tests

Test individual components in isolation:

```python
# test/test_task.py
import pytest
from chitu_diffusion.task import DiffusionUserParams, DiffusionTask

class TestDiffusionTask:
    def test_creation(self):
        params = DiffusionUserParams(prompt="test")
        task = DiffusionTask.from_user_request(params)
        
        assert task.task_id is not None
        assert task.status == TaskStatus.PENDING
        assert task.user_params.prompt == "test"
    
    def test_invalid_params(self):
        with pytest.raises(ValueError):
            params = DiffusionUserParams(height=-1)
```

### Integration Tests

Test components working together:

```python
# test/integration/test_pipeline.py
def test_full_pipeline():
    # Setup
    chitu_init(test_args)
    
    # Create task
    params = DiffusionUserParams(
        prompt="test",
        num_frames=16,  # Minimal for speed
        num_inference_steps=5
    )
    task = DiffusionTask.from_user_request(params)
    DiffusionTaskPool.add(task)
    
    # Generate
    while not DiffusionTaskPool.all_finished():
        chitu_generate()
    
    # Verify
    assert task.status == TaskStatus.FINISHED
    assert task.buffer.video is not None
```

### Performance Tests

Measure performance:

```python
# test/performance/test_benchmarks.py
import time

def test_generation_speed():
    start = time.time()
    
    # Run generation
    generate_test_video()
    
    elapsed = time.time() - start
    
    # Should complete within threshold
    assert elapsed < 60.0  # 60 seconds
```

## Fixtures

Use fixtures for common setup:

```python
# test/conftest.py
import pytest

@pytest.fixture
def test_args():
    """Minimal args for testing"""
    return create_test_config()

@pytest.fixture
def sample_task():
    """Sample task for testing"""
    params = DiffusionUserParams(
        prompt="test",
        num_frames=16,
        num_inference_steps=5
    )
    return DiffusionTask.from_user_request(params)

@pytest.fixture
def initialized_backend(test_args):
    """Initialized backend"""
    backend = DiffusionBackend(test_args)
    yield backend
    # Cleanup
    cleanup_backend(backend)
```

## Test Markers

Mark tests by type:

```python
import pytest

@pytest.mark.unit
def test_unit():
    pass

@pytest.mark.integration
def test_integration():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.gpu
def test_requires_gpu():
    pass
```

Configure in `pytest.ini`:
```ini
[pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests
    gpu: Requires GPU
```

## Mocking

Use mocks for external dependencies:

```python
from unittest.mock import Mock, patch

def test_with_mock():
    # Mock expensive operation
    with patch('chitu_diffusion.backend.load_checkpoint') as mock_load:
        mock_load.return_value = Mock()
        
        backend = DiffusionBackend(args)
        
        assert mock_load.called
```

## Assertions

### Good Assertions

```python
# Specific
assert task.status == TaskStatus.FINISHED

# With message
assert len(pool) > 0, "Pool should not be empty"

# Multiple related checks
assert video.shape == (81, 3, 480, 848)
assert video.dtype == torch.float32
assert video.min() >= 0.0 and video.max() <= 1.0
```

### Pytest Helpers

```python
# Approximate equality
assert result == pytest.approx(expected, abs=0.01)

# Exceptions
with pytest.raises(ValueError, match="Invalid height"):
    create_task(height=-1)

# Warnings
with pytest.warns(UserWarning):
    deprecated_function()
```

## Continuous Integration

Tests run automatically on:
- Push to main
- Pull requests
- Scheduled (nightly)

See `.github/workflows/test.yml`

## Coverage Goals

- **Overall**: >80%
- **Core modules**: >90%
- **New features**: 100%

## Best Practices

1. **Test Behavior, Not Implementation**
   ```python
   # Good: Test what it does
   assert task.is_finished()
   
   # Bad: Test how it does it
   assert task._status_flag == 2
   ```

2. **Use Descriptive Names**
   ```python
   # Good
   def test_task_creation_with_empty_prompt_raises_error():
       pass
   
   # Bad
   def test1():
       pass
   ```

3. **Keep Tests Fast**
   - Use minimal configurations
   - Mock expensive operations
   - Mark slow tests

4. **One Assertion Per Test** (when possible)
   ```python
   def test_task_has_id():
       assert task.task_id is not None
   
   def test_task_starts_pending():
       assert task.status == TaskStatus.PENDING
   ```

5. **Clean Up Resources**
   ```python
   def test_with_cleanup():
       backend = DiffusionBackend(args)
       try:
           # Test code
           pass
       finally:
           backend.cleanup()
   ```

## See Also

- [Developer Guide](developer-guide.md)
- [Code Style](code-style.md)
- [Pytest Documentation](https://docs.pytest.org/)
