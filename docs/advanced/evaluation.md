# Evaluation

Smart-Diffusion supports evaluation of generated videos using industry-standard metrics.

## Supported Metrics

### VBench

VBench is a comprehensive benchmark for video generation quality.

**Installation**:
```bash
pip install vbench
```

**Usage**:
```python
from vbench import VBench

evaluator = VBench()
scores = evaluator.evaluate(
    videos_path="./generated_videos",
    prompts_path="./prompts.txt"
)

print(f"Overall score: {scores['overall']}")
print(f"Subject consistency: {scores['subject_consistency']}")
print(f"Motion smoothness: {scores['motion_smoothness']}")
```

### Custom Mode

Run VBench in custom mode for your videos:

```bash
python -m vbench.evaluate \
    --videos_path ./generated \
    --mode custom \
    --output results.json
```

## Evaluation Pipeline

### 1. Generate Videos

```python
prompts = [
    "A cat walking on grass",
    "A sunset over mountains",
    # ... more prompts
]

for prompt in prompts:
    params = DiffusionUserParams(prompt=prompt)
    task = DiffusionTask.from_user_request(params)
    DiffusionTaskPool.add(task)
    
while not DiffusionTaskPool.all_finished():
    chitu_generate()
```

### 2. Run Evaluation

```python
from vbench import VBench

evaluator = VBench()
results = evaluator.evaluate_directory("./generated_videos")
```

### 3. Analyze Results

```python
# Overall quality
print(f"Mean quality: {results.mean()}")

# Per-dimension scores
for dimension in results.dimensions:
    print(f"{dimension}: {results[dimension]}")
```

## Quality Metrics

### Visual Quality
- Image quality
- Color correctness
- Sharpness

### Temporal Coherence
- Motion smoothness
- Frame consistency
- Temporal artifacts

### Semantic Alignment
- Subject consistency
- Background consistency
- Aesthetic quality

### Text Alignment
- Prompt adherence
- Concept coverage

## Comparative Evaluation

Compare different configurations:

```python
configs = [
    {"attn_type": "flash_attn", "flexcache": None},
    {"attn_type": "sage", "flexcache": None},
    {"attn_type": "sage", "flexcache": "teacache"},
]

results = {}
for config in configs:
    # Generate with config
    # Evaluate
    results[str(config)] = scores
    
# Compare
for config, scores in results.items():
    print(f"{config}: {scores['overall']}")
```

## See Also

- [Performance Tuning](../user-guide/performance-tuning.md)
- [FlexCache](flexcache.md)
- [Attention Backends](../architecture/attention-backends.md)
