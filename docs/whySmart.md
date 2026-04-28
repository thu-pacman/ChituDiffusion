# Why ChituDiffusion?
[中文版](./whySmart_zh.md)

## Characteristics of Diffusion Inference

Diffusion inference is compute-intensive with these key characteristics:

1. **Sample-by-sample execution**: Batching provides minimal GPU utilization improvement; single-sample streaming is sufficient.  
2. **Long sequences, small models**: With very long activation sequences but relatively small model parameters, Context Parallelism is the most cost-effective parallelization strategy.  
3. **Attention is the bottleneck**: In long-sequence scenarios, Full Attention accounts for over 80% of end-to-end latency, making it the primary optimization target.  
4. **Small activation changes**: Activations change minimally between denoising steps, so simple Feature Cache methods can provide significant speedups.

## ChituDiffusion Design Philosophy

### Three Optimization Directions: Parallelism × Kernels × Algorithms  
Each direction can be optimized independently, but combining them yields the best results.  
(Technical details will be updated progressively—PRs are welcome!)

### Service Framework for Multi-User, Multi-Task Workloads  
We provide a **long-running, hot-upgradable, horizontally-scalable** Diffusion service, not a cold-start script.  
The core idea is to decompose the Diffusion pipeline into composable stages orchestrated by a unified scheduler:

- Let users tune their quality-efficiency tradeoff: inference steps, CFG, cache ratio—all adjustable at runtime.  
- Keep all resources fully utilized: not just compute, but also memory, bandwidth, and CPU.

## Developer Guide

Thanks for contributing to ChituDiffusion! To make code review easier, please understand our parameter taxonomy:

| Category | Lifecycle | Location | Who Can Change | Best Practice |
|---|---|---|---|---|
| Model params | Static | `chitu_core/config/models/<model>.yaml` | Nobody | Tied to weights; changes will break things |
| User params | Dynamic (per-request) | `DiffusionUserParams` | End user | Expose only necessary parameters; keep it simple |
| System params | Semi-dynamic (init-time) | `chitu launch args` | Ops/Scheduler | No changes after init; prevents distributed state issues |

Remember:  
Every extra parameter adds documentation, testing, and user complexity. Flexibility ≠ more parameters.

### Directory Structure

`/chitu_core` contains Chitu's native code. Avoid modifying `ServeConfig` and `ParallelState` unless necessary.
`/chitu_diffusion` is our diffusion framework built on Chitu's architecture. It can be modified but should maintain the basic structure.
* `chitu_diffusion_main.py`: Main parameters for system initialization, startup, and shutdown
* `backend.py`: Backend built on system parameters, stores models and schedules tasks.
