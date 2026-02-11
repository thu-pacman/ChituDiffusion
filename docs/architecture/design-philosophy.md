# Why Smart-Diffusion?
[中文版](./whySmart_zh.md)

## Workload Characteristics of Diffusion

Diffusion inference is a *full-stack compute-bound* task:

1. Sample-by-sample execution: Batching hardly improves GPU utilization; pure streaming is mandatory.  
2. Long activation sequences, yet *relatively small* models: Sequence Parallelism becomes the most economical and scalable dimension.  
3. In long-sequence regimes, Full Attention accounts for ~80 % of end-to-end latency: operator-level efforts must target Attention first.  
4. Activations change mildly between denoising steps: a simple, *lossy* Feature Cache yields instant speed-ups.

## Design Philosophy of Smart-Diffusion

### Three Pillars: Parallelism × Kernels × Algorithms  
Each can be pursued independently, but *co-design* extracts the last drop of performance.  
(Technical deep-dives will be released incrementally—PRs welcome!)

### Service Framework for Multi-User, Multi-Task Workloads  
We ship a **long-running, hot-upgradable, horizontally-scalable** Diffusion service—not a frozen script that cold-starts every time.  
Key idea: decompose the Diffusion pipeline into composable stages and orchestrate them with a unified scheduler:

- Let users tune their own quality-efficiency trade-off: steps, CFG, cache ratio—all at runtime.  
- Keep *all* resources saturated: FLOPs are only the first bottleneck; memory, bandwidth and CPU must be fully utilized as well.

## Developer Guide

Thanks for joining the Smart-Diffusion open-source community! To keep code review painless, please align on the “parameter taxonomy” first:

| Category | Life-Cycle | Location | Who Can Change | Best Practice |
|---|---|---|---|---|
| Model params | Static | `chitu_core/config/models/<model>.yaml` | Nobody | Tied to weights; any change is UB |
| User params | Dynamic (per-request) | `DiffusionUserParams` | End user | Expose *necessary & sufficient* knobs; avoid parameter spam |
| System params | Semi-dynamic (init-time) | `chitu launch args` | Ops/Scheduler | No hot-edit after init; prevents distributed state explosion |

Remember:  
Every extra knob costs *documentation + tests + user mental model*. Flexibility ≠ surface area.