# API Reference - Core

This page documents the core API of ChituDiffusion.

## Main Functions

### chitu_init

```python
def chitu_init(args, logging_level=None)
```

Initialize the Chitu diffusion system.

**Parameters:**
- `args`: Configuration object containing all system parameters
- `logging_level`: Optional logging level override (default: INFO, DEBUG if CHITU_DEBUG=1)

**Initializes:**
1. Logging system
2. Environment variables
3. Distributed training groups
4. Model checkpoints
5. Backend, scheduler, and generator

**Example:**

```python
from hydra import compose, initialize
from chitu_diffusion import chitu_init

initialize(config_path="config", version_base=None)
args = compose(config_name="wan")
args.models.ckpt_dir = "/path/to/checkpoint"

chitu_init(args)
```

### chitu_start

```python
def chitu_start()
```

Mark the backend as running and ready to process tasks.

**Example:**

```python
chitu_start()
```

### chitu_generate

```python
@torch.inference_mode()
def chitu_generate()
```

Execute one generation step across all ranks.

Must be called in a loop until all tasks are finished. Rank 0 schedules and processes tasks, while other ranks participate in distributed computation.

**Example:**

```python
while not DiffusionTaskPool.all_finished():
    chitu_generate()
```

### chitu_terminate

```python
def chitu_terminate()
```

Gracefully terminate the Chitu backend.

Signals all ranks to stop processing by setting the backend state to Terminated.

**Example:**

```python
chitu_terminate()
```

### chitu_is_terminated

```python
def chitu_is_terminated() -> bool
```

Check if the Chitu backend has been terminated.

**Returns:**
- `bool`: True if terminated, False otherwise

## Task Management

### DiffusionUserParams

```python
@dataclass
class DiffusionUserParams:
    role: str = "user"
    prompt: str = ""
    height: int = 480
    width: int = 848
    num_frames: int = 81
    fps: int = 24
    num_inference_steps: int = 50
    guidance_scale: float = 7.0
    seed: Optional[int] = None
    save_path: Optional[str] = None
    flexcache: Optional[str] = None
```

User-facing parameters for video generation.

**Fields:**
- `role`: User identifier
- `prompt`: Text description of desired video
- `height`: Video height in pixels
- `width`: Video width in pixels
- `num_frames`: Number of frames to generate
- `fps`: Frames per second
- `num_inference_steps`: Number of denoising steps
- `guidance_scale`: CFG scale for prompt adherence
- `seed`: Random seed (None for random)
- `save_path`: Output path (None for auto-generated)
- `flexcache`: Cache strategy ('teacache', 'PAB', or None)

**Example:**

```python
from chitu_diffusion.task import DiffusionUserParams

params = DiffusionUserParams(
    role="user1",
    prompt="A cat walking on grass",
    num_inference_steps=50,
    guidance_scale=7.0,
)
```

### DiffusionTask

```python
class DiffusionTask:
    def __init__(self, user_request: DiffusionUserRequest)
    
    @classmethod
    def from_user_request(cls, params: DiffusionUserParams) -> "DiffusionTask"
    
    @classmethod
    def create_terminate_signal(cls, task_id: str) -> "DiffusionTask"
```

Internal task representation with buffers and status tracking.

**Methods:**
- `from_user_request(params)`: Create task from user parameters
- `create_terminate_signal(task_id)`: Create termination signal task

**Example:**

```python
from chitu_diffusion.task import DiffusionTask, DiffusionUserParams

params = DiffusionUserParams(prompt="A cat")
task = DiffusionTask.from_user_request(params)
```

### DiffusionTaskPool

```python
class DiffusionTaskPool:
    pool: Dict[str, DiffusionTask] = {}
    
    @classmethod
    def add(cls, task: DiffusionTask)
    
    @classmethod
    def all_finished(cls) -> bool
    
    @classmethod
    def get_pending_tasks(cls) -> List[str]
```

Global task pool manager.

**Methods:**
- `add(task)`: Add task to pool
- `all_finished()`: Check if all tasks are completed
- `get_pending_tasks()`: Get list of pending task IDs

**Example:**

```python
from chitu_diffusion.task import DiffusionTaskPool

DiffusionTaskPool.add(task)

while not DiffusionTaskPool.all_finished():
    chitu_generate()
```

## Configuration

### Model Configuration

Model parameters are defined in YAML files:

```yaml
# chitu_core/config/models/wan.yaml
name: Wan2.1-T2V-14B
type: diff-wan
ckpt_dir: null  # Set at runtime
source: https://huggingface.co/Wan-AI/Wan2.1-T2V-14B

transformer:
  in_channels: 16
  patch_size: [1, 2, 2]
  hidden_size: 3072
  depth: 40
  num_heads: 24
  # ... more parameters
```

### System Configuration

System parameters are set via launch arguments:

```python
args.infer.attn_type = "sage"
args.infer.diffusion.low_mem_level = 2
args.infer.diffusion.cp_size = 1
```

Or via command line:

```bash
python script.py \
    infer.attn_type=sage \
    infer.diffusion.low_mem_level=2
```

## Utilities

### Device Detection

```python
from chitu_core.device_type import (
    get_device_name,
    is_nvidia,
    is_hopper,
    has_native_fp8
)

device = get_device_name()  # "NVIDIA H100"
if is_hopper():
    print("Using Hopper GPU")
```

### Global Configuration

```python
from chitu_core.global_vars import get_global_args

args = get_global_args()
print(f"Model: {args.models.name}")
```

## Error Handling

### Common Exceptions

**ValueError**: Invalid configuration
```python
# Raised when checkpoint path is not provided
args.models.ckpt_dir = None  # Raises ValueError
```

**FileNotFoundError**: Checkpoint not found
```python
# Raised when checkpoint files don't exist
args.models.ckpt_dir = "/invalid/path"  # Raises FileNotFoundError
```

**RuntimeError**: CUDA errors
```python
# Raised when running out of GPU memory
# Use low_mem_level to reduce memory usage
```

## Complete Example

```python
from chitu_diffusion import (
    chitu_init, chitu_start, chitu_generate, chitu_terminate
)
from chitu_diffusion.task import (
    DiffusionUserParams, DiffusionTask, DiffusionTaskPool
)
from hydra import compose, initialize

# Initialize
initialize(config_path="config", version_base=None)
args = compose(config_name="wan")
args.models.ckpt_dir = "/path/to/checkpoint"

chitu_init(args)
chitu_start()

# Create task
params = DiffusionUserParams(
    prompt="A cat walking on grass",
    height=480,
    width=848,
    num_frames=81,
    num_inference_steps=50,
)

task = DiffusionTask.from_user_request(params)
DiffusionTaskPool.add(task)

# Generate
while not DiffusionTaskPool.all_finished():
    chitu_generate()

# Cleanup
chitu_terminate()

print(f"Done! Video: {task.buffer.save_path}")
```

## See Also

- [Backend API](backend.md) - DiffusionBackend details
- [Generator API](generator.md) - Generation pipeline
- [Task API](task.md) - Task management
- [Scheduler API](scheduler.md) - Task scheduling
