# CP/DP Hot Switch Experiments

This directory tracks the exploration of dynamic hot switching between
single-request context/sequence parallel execution and multi-request data
parallel execution for DiT inference.

## Question

When a second request arrives while one DiT request is already running with
two-GPU sequence/context parallelism, can the system switch at a denoising step
boundary into two one-GPU data-parallel replicas, one per request, and improve
serving efficiency?

In short:

```text
early phase:  request A uses GPU0+GPU1 as CP/SP for low latency
new arrival:  request B arrives while A still has denoising steps left
switch:       GPU0 continues/finishes A, GPU1 starts B as DP-style replicas
goal:         reduce queueing delay and improve throughput without hurting tail latency
```

## Working Hypothesis

The feature is likely useful only under a constrained policy:

- switch only at denoising step boundaries;
- switch only in the early part of sampling, while enough steps remain;
- switch only when the expected queueing benefit exceeds state migration,
  communicator, graph-capture, and cache-layout costs;
- start with no-FlexCache or request-local cache state, then extend to
  FlexCache-aware switching after the baseline cost model is understood.

The first experiment should be a scheduler/cost-model study before a runtime
implementation.

## Initial Artifacts

- [literature_notes.md](literature_notes.md): LLM serving papers that define
  similar dynamic-parallelism problems.

## Candidate Experiment Plan

1. Build an offline simulator from ChituBench timing data:
   - per-step latency for one-GPU DP-style execution;
   - per-step latency for two-GPU CP/SP execution;
   - synthetic or recorded request arrival traces;
   - switch cost as a parameter.
2. Evaluate scheduling policies:
   - static one-GPU DP replicas;
   - static two-GPU CP/SP for one request at a time;
   - early-window CP/SP -> DP hot switch;
   - threshold policy based on remaining steps and queue length.
3. Measure serving metrics:
   - mean latency;
   - P95/P99 latency;
   - queueing delay;
   - GPU utilization proxy;
   - number of switches and wasted/migrated work.
4. Only after the simulator shows a useful region, prototype runtime support:
   - safe transition point at DiT step boundary;
   - latent/state redistribution;
   - distributed group handling;
   - cache-state compatibility checks.

## Open Design Points

- Should the first target be Flux, Qwen-Image, or Wan?
- Is the first switching target CP/SP -> independent one-GPU requests, or
  CP/SP degree reduction with both requests still partially parallelized?
- What is the minimal state that must migrate at a step boundary?
- Do existing attention/parallel backends expose enough hooks to switch without
  reinitializing the whole engine?
- How does FlexCache state change the cost model?

