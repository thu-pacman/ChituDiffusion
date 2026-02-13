# Task Management API

This page documents the task management system in Smart-Diffusion, including task creation, tracking, and pooling.

## DiffusionUserParams

User-facing parameters for video generation requests.

### Class Definition

```python
@dataclass
class DiffusionUserParams:
    """
    User-facing parameters for video generation.
    
    All fields have sensible defaults and can be customized per request.
    """
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

### Fields

#### role
**Type**: `str`  
**Default**: `"user"`  
**Description**: User identifier for tracking and logging

**Example**:
```python
params = DiffusionUserParams(role="user_alice")
```

#### prompt
**Type**: `str`  
**Default**: `""`  
**Description**: Text description of the desired video

**Example**:
```python
params = DiffusionUserParams(
    prompt="A cat walking on grass in slow motion"
)
```

#### height
**Type**: `int`  
**Default**: `480`  
**Description**: Video height in pixels

**Common Values**:
- 240: Low resolution (fast)
- 480: Standard resolution
- 720: High resolution (slow, high VRAM)

#### width
**Type**: `int`  
**Default**: `848`  
**Description**: Video width in pixels

**Common Values**:
- 424: Low resolution
- 848: Standard resolution
- 1280: High resolution

#### num_frames
**Type**: `int`  
**Default**: `81`  
**Description**: Number of frames to generate

**Guidelines**:
- 16-32: Short clip (2-3 seconds at 24fps)
- 49-81: Standard video (3-4 seconds)
- 121+: Long video (5+ seconds, high VRAM)

#### fps
**Type**: `int`  
**Default**: `24`  
**Description**: Frames per second for output video

**Common Values**:
- 12: Lower frame rate
- 24: Standard (cinematic)
- 30: Standard (video)
- 60: High frame rate

#### num_inference_steps
**Type**: `int`  
**Default**: `50`  
**Description**: Number of denoising steps

**Trade-off**: More steps = better quality but slower
- 20-30: Fast, lower quality
- 40-50: Balanced (recommended)
- 60-100: Slow, highest quality

#### guidance_scale
**Type**: `float`  
**Default**: `7.0`  
**Description**: Classifier-Free Guidance scale

**Trade-off**: Higher = more prompt adherence but less creativity
- 3.0-5.0: More creative
- 6.0-8.0: Balanced (recommended)
- 9.0-15.0: Strict prompt following

#### seed
**Type**: `Optional[int]`  
**Default**: `None`  
**Description**: Random seed for reproducibility

**Example**:
```python
# Reproducible generation
params = DiffusionUserParams(
    prompt="A cat",
    seed=42
)
```

#### save_path
**Type**: `Optional[str]`  
**Default**: `None`  
**Description**: Output video path (auto-generated if None)

**Example**:
```python
params = DiffusionUserParams(
    prompt="A cat",
    save_path="./videos/cat_walking.mp4"
)
```

#### flexcache
**Type**: `Optional[str]`  
**Default**: `None`  
**Description**: Cache strategy to use

**Options**:
- `None`: No caching
- `"teacache"`: Temporal cache reuse
- `"PAB"`: Pyramid Attention Broadcast

### Usage Examples

#### Basic Usage
```python
from chitu_diffusion.task import DiffusionUserParams

params = DiffusionUserParams(
    prompt="A sunset over the ocean"
)
```

#### Custom Configuration
```python
params = DiffusionUserParams(
    prompt="A cat walking",
    height=720,
    width=1280,
    num_frames=121,
    num_inference_steps=50,
    guidance_scale=8.0,
    seed=42,
    save_path="./my_video.mp4",
    flexcache="teacache"
)
```

#### High Quality
```python
params = DiffusionUserParams(
    prompt="A detailed landscape",
    height=720,
    width=1280,
    num_inference_steps=100,
    guidance_scale=9.0
)
```

#### Fast Generation
```python
params = DiffusionUserParams(
    prompt="Quick test",
    height=240,
    width=424,
    num_frames=49,
    num_inference_steps=30,
    flexcache="teacache"
)
```

## DiffusionTask

Internal task representation with buffers and status tracking.

### Class Definition

```python
class DiffusionTask:
    """
    Internal representation of a generation task.
    
    Manages:
    - Task state and progress
    - Intermediate buffers
    - Serialization for distributed execution
    """
```

### Attributes

```python
class DiffusionTask:
    task_id: str  # Unique task identifier
    user_params: DiffusionUserParams  # User parameters
    buffer: TaskBuffer  # Intermediate results
    status: TaskStatus  # Current status
    current_step: int  # Current denoising step
    created_at: float  # Creation timestamp
    started_at: Optional[float]  # Start timestamp
    finished_at: Optional[float]  # Finish timestamp
```

### Task Status

```python
class TaskStatus(Enum):
    PENDING = "pending"  # Waiting to start
    ENCODING = "encoding"  # Text encoding phase
    DENOISING = "denoising"  # Denoising phase
    DECODING = "decoding"  # VAE decoding phase
    FINISHED = "finished"  # Completed
    FAILED = "failed"  # Error occurred
```

### Methods

#### from_user_request

```python
@classmethod
def from_user_request(cls, params: DiffusionUserParams) -> "DiffusionTask":
    """
    Create task from user parameters.
    
    Args:
        params: DiffusionUserParams instance
    
    Returns:
        task: DiffusionTask ready to execute
    """
```

**Example**:
```python
from chitu_diffusion.task import DiffusionTask, DiffusionUserParams

params = DiffusionUserParams(prompt="A cat")
task = DiffusionTask.from_user_request(params)
```

#### create_terminate_signal

```python
@classmethod
def create_terminate_signal(cls, task_id: str) -> "DiffusionTask":
    """
    Create a special task that signals termination.
    
    Args:
        task_id: Unique identifier for termination signal
    
    Returns:
        task: Termination signal task
    """
```

#### is_finished

```python
def is_finished(self) -> bool:
    """Check if task has completed"""
    return self.status in [TaskStatus.FINISHED, TaskStatus.FAILED]
```

#### get_elapsed_time

```python
def get_elapsed_time(self) -> float:
    """Get elapsed time in seconds"""
    if self.started_at is None:
        return 0.0
    
    end_time = self.finished_at or time.time()
    return end_time - self.started_at
```

### Task Buffer

Internal buffer for intermediate results:

```python
class TaskBuffer:
    text_embeddings: Optional[torch.Tensor] = None
    latent: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None
    save_path: Optional[str] = None
```

## DiffusionTaskPool

Global task pool manager.

### Class Definition

```python
class DiffusionTaskPool:
    """
    Static task pool manager.
    
    Manages all active tasks across the system.
    """
    pool: Dict[str, DiffusionTask] = {}
```

### Methods

#### add

```python
@classmethod
def add(cls, task: DiffusionTask):
    """
    Add task to pool.
    
    Args:
        task: DiffusionTask to add
    """
```

**Example**:
```python
from chitu_diffusion.task import DiffusionTaskPool

DiffusionTaskPool.add(task)
```

#### get

```python
@classmethod
def get(cls, task_id: str) -> Optional[DiffusionTask]:
    """
    Get task by ID.
    
    Args:
        task_id: Task identifier
    
    Returns:
        task: DiffusionTask or None if not found
    """
```

#### remove

```python
@classmethod
def remove(cls, task_id: str):
    """Remove task from pool"""
    if task_id in cls.pool:
        del cls.pool[task_id]
```

#### all_finished

```python
@classmethod
def all_finished(cls) -> bool:
    """
    Check if all tasks are finished.
    
    Returns:
        finished: True if all tasks complete
    """
```

**Example**:
```python
while not DiffusionTaskPool.all_finished():
    chitu_generate()
```

#### get_pending_tasks

```python
@classmethod
def get_pending_tasks(cls) -> List[str]:
    """
    Get list of pending task IDs.
    
    Returns:
        task_ids: List of pending task IDs
    """
```

#### get_statistics

```python
@classmethod
def get_statistics(cls) -> Dict[str, int]:
    """
    Get task statistics.
    
    Returns:
        stats: Dict with counts per status
    """
```

**Example**:
```python
stats = DiffusionTaskPool.get_statistics()
print(f"Pending: {stats['pending']}")
print(f"Running: {stats['encoding'] + stats['denoising'] + stats['decoding']}")
print(f"Finished: {stats['finished']}")
print(f"Failed: {stats['failed']}")
```

### Usage Examples

#### Single Task

```python
from chitu_diffusion import chitu_init, chitu_start, chitu_generate, chitu_terminate
from chitu_diffusion.task import DiffusionUserParams, DiffusionTask, DiffusionTaskPool

# Initialize
chitu_init(args)
chitu_start()

# Create and add task
params = DiffusionUserParams(prompt="A cat")
task = DiffusionTask.from_user_request(params)
DiffusionTaskPool.add(task)

# Generate
while not DiffusionTaskPool.all_finished():
    chitu_generate()

# Cleanup
chitu_terminate()
```

#### Multiple Tasks

```python
# Add multiple tasks
prompts = [
    "A cat walking",
    "A dog running",
    "A bird flying"
]

for prompt in prompts:
    params = DiffusionUserParams(prompt=prompt)
    task = DiffusionTask.from_user_request(params)
    DiffusionTaskPool.add(task)

# Process all
while not DiffusionTaskPool.all_finished():
    chitu_generate()
    
    # Show progress
    stats = DiffusionTaskPool.get_statistics()
    print(f"Finished: {stats['finished']}/{len(prompts)}")
```

#### With Progress Tracking

```python
import time

tasks = []
for i, prompt in enumerate(prompts):
    params = DiffusionUserParams(prompt=prompt, role=f"user_{i}")
    task = DiffusionTask.from_user_request(params)
    DiffusionTaskPool.add(task)
    tasks.append(task)

while not DiffusionTaskPool.all_finished():
    chitu_generate()
    
    # Print progress for each task
    for task in tasks:
        if not task.is_finished():
            progress = task.current_step / task.user_params.num_inference_steps
            elapsed = task.get_elapsed_time()
            print(f"Task {task.task_id}: {progress*100:.1f}% ({elapsed:.1f}s)")
    
    time.sleep(1)
```

## Error Handling

### Invalid Parameters

```python
try:
    params = DiffusionUserParams(
        height=-1  # Invalid
    )
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

### Task Failures

```python
task = DiffusionTask.from_user_request(params)
DiffusionTaskPool.add(task)

while not DiffusionTaskPool.all_finished():
    chitu_generate()

# Check for failures
if task.status == TaskStatus.FAILED:
    print(f"Task failed: {task.error_message}")
```

## See Also

- [Core API](core.md) - Main interface
- [Scheduler API](scheduler.md) - Task scheduling
- [Backend API](backend.md) - Backend management
- [User Guide - Basic Usage](../user-guide/basic-usage.md)
