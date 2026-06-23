# Literature Notes: Dynamic Parallelism for LLM Serving

This note records LLM serving work that is closest to DiT CP/DP hot switching.
The terminology in the literature is usually not "CP/DP hot switch"; it is
closer to dynamic parallelism, elastic parallelism, on-the-fly switching, model
re-sharding, or cross-instance transformation.

## Mapping to Our DiT Problem

The DiT feature under consideration:

```text
Given N GPUs and dynamic request arrivals, use CP/SP to reduce single-request
latency when the system is lightly loaded, then switch to DP-style independent
replicas when more requests arrive and throughput/queueing delay dominate.
```

The equivalent LLM framing:

```text
TP/CP/SP improves one-request latency or context capacity, while DP improves
multi-request throughput. Static deployment chooses one point on this tradeoff.
Dynamic serving tries to change the parallelism layout at safe points without
moving too much request state.
```

The most reusable concepts for DiT are:

- state layout invariance;
- safe transition points;
- migration-aware scheduling;
- elastic parallel degree;
- cost models that compare remaining work against switching overhead.

## Shift Parallelism

Paper:

- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic
  Workloads
- arXiv: <https://arxiv.org/abs/2509.16495>
- ACM DL: <https://dl.acm.org/doi/10.1145/3779212.3790219>

Core motivation:

- TP gives low latency but communication hurts aggregate throughput.
- DP gives higher throughput but cannot reduce single-request latency.
- The paper argues that TP and DP are hard to combine directly because KV cache
  layout differs across parallelism modes.
- It introduces SP as a DP-like high-throughput mode with KV cache invariance,
  then dynamically switches between TP and SP.

Core method:

- Adapt sequence parallelism to inference.
- Switch between TP and SP depending on workload pressure.
- Preserve request state by choosing a parallelism pair whose KV cache layout is
  compatible.
- Evaluate on dynamic production traces and synthetic workloads.

Definition useful for us:

- The main feasibility condition is not just "can we switch execution kernels",
  but "can request state remain valid across layouts".
- For DiT, the analogous state is latent tensors, step scheduler state, and
  optional acceleration/cache state from FlexCache-like methods.

DiT implication:

- If CP/SP sharding can expose a request state that is cheap to materialize as a
  full single-GPU state at a step boundary, DiT may be easier than LLM because it
  lacks long-lived token KV cache.
- If FlexCache state is sharded by sequence or block, the DiT problem becomes
  closer to LLM KV layout compatibility.

## FLYING SERVING

Paper:

- FLYING SERVING: On-the-Fly Parallelism Switching for Large Language Model
  Serving
- arXiv: <https://arxiv.org/abs/2602.22593>

Core motivation:

- Production LLM traffic needs high throughput, low latency, and enough context
  capacity under non-stationary arrivals.
- Static DP/TP deployment cannot adapt to bursts, priority requests, or
  long-context requests without disruptive reconfiguration.

Core method:

- Implement online DP-TP switching in a vLLM-based system without restarting
  engine workers.
- Use a model weights manager so workers can expose TP shard views on demand.
- Use a KV cache adaptor to keep request KV state valid across DP/TP layouts.
- Pre-initialize communicator pools to avoid paying setup cost during a switch.
- Coordinate transitions with a scheduler designed to avoid deadlocks under
  execution skew.

Definition useful for us:

- Hot switching is a full systems problem:
  - weight/state virtualization;
  - cache/state adaptation;
  - communicator readiness;
  - scheduler safe points;
  - correctness under skewed worker progress.

DiT implication:

- For a first DiT prototype, we should avoid dynamic process-group creation in
  the critical path. Pre-created groups or a fixed two-GPU topology are more
  realistic.
- A DiT step boundary is the likely safe point; layer-level switching is
  probably not worth the complexity at first.

## LoongServe

Paper:

- LoongServe: Efficiently Serving Long-Context Large Language Models with
  Elastic Sequence Parallelism
- arXiv: <https://arxiv.org/abs/2404.09526>
- SOSP 2024

Core motivation:

- Long-context LLM serving has large variance across requests and across phases
  of the same request.
- Static parallelism wastes resources because short/early/late phases can need
  different degrees of parallelism.

Core method:

- Propose Elastic Sequence Parallelism.
- Adjust the degree of sequence parallelism in real time.
- Reduce KV cache migration overhead.
- Overlap partial decoding communication with computation.
- Reduce KV cache fragmentation across instances.

Definition useful for us:

- The problem is not only switching between two named modes; it is choosing an
  elastic parallelism degree per request and phase.
- The scheduler should reason about phase-specific resource demand rather than
  a single global deployment mode.

DiT implication:

- DiT denoising has a fixed number of steps but different steps may have
  different cache value or compute characteristics.
- We can define early/mid/late denoising phases and only permit hot switch in
  phases where remaining work justifies migration.

## Gyges

Paper:

- Gyges: Dynamic Cross-Instance Parallelism Transformation for Efficient LLM
  Inference
- arXiv HTML: <https://arxiv.org/html/2509.19729v1>

Core motivation:

- Many requests can be served efficiently by small instances or replicas, while
  occasional long-context requests need resources pooled across multiple GPUs.
- Static instance sizing leaves either throughput or long-context capacity on
  the table.

Core method:

- Transform between multiple small inference instances and larger parallel
  instances.
- Make scheduling transformation-aware so that migration and reshaping costs do
  not dominate.
- Focus on dynamic cross-instance parallelism, not just within-instance kernel
  changes.

Definition useful for us:

- The key object is an "instance transformation" between resource layouts.
- Evaluation should include both performance benefits and transformation costs.

DiT implication:

- CP/SP -> DP hot switch can be framed as a cross-instance transformation:
  one two-GPU instance serving request A becomes two one-GPU instances serving A
  and B.
- This framing is cleaner than treating the switch as a local kernel choice.

## Seesaw

Paper:

- Seesaw: High-throughput LLM Inference via Model Re-sharding
- arXiv: <https://arxiv.org/abs/2503.06433>

Core motivation:

- LLM prefill and decode have different compute/communication/memory behavior.
- A single static sharding choice is inefficient across both phases.

Core method:

- Re-shard the model between phases.
- Use scheduling to reduce transition overhead and avoid excessive reshards.

Definition useful for us:

- Switch at a semantically meaningful phase boundary rather than at arbitrary
  time.
- Consider transition-minimizing scheduling as part of the system, not as an
  afterthought.

DiT implication:

- DiT lacks a prefill/decode boundary, so we need to create safe boundaries
  from denoising steps or from early/mid/late sampling phases.
- Fixed checkpoint steps, such as step 1/2/4/8 or first 20%-30% of sampling,
  are likely easier to reason about than "switch whenever a request arrives".

## ReMP

Paper:

- ReMP: Runtime Model-Parallelism Reconfiguration
- arXiv: <https://arxiv.org/abs/2606.18741>

Core motivation:

- Model-parallel deployments often need to change TP/PP layout as load and
  resource availability change.
- Restarting or redeploying is too slow for online adaptation.

Core method:

- Reconfigure model parallelism at runtime.
- Decouple workers, model state, and communication topology enough to change
  the runtime layout.

Definition useful for us:

- Treat parallelism layout as a mutable runtime object.
- Separate the cost of reconfiguration from steady-state execution.

DiT implication:

- A clean DiT implementation should not bury CP/DP decisions inside model code.
  The scheduler/runtime should own the layout transition.

## AlpaServe

Paper:

- AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning
  Serving
- OSDI 2023
- Project: <https://alpa.ai/serve>

Core motivation:

- Serving multiple models with fixed parallelism can waste memory and compute.
- Statistical multiplexing can improve cluster utilization while preserving
  latency targets.

Core method:

- Search for model-parallel placements.
- Use batching and placement decisions to multiplex serving workloads.

Definition useful for us:

- Even without per-request hot switching, placement/search can define strong
  baselines.
- Static or admission-time placement may capture much of the benefit at lower
  complexity.

DiT implication:

- Our hot-switch proposal should be compared against admission control:
  wait briefly for another request, then start DP; or start CP/SP only when the
  queue is empty and an early-switch window remains.

## Candidate Problem Statement for ChituDiffusion

```text
We study dynamic CP/SP-to-DP transformation for DiT serving. Under light load,
a request can use multiple GPUs with sequence/context parallelism to reduce
single-request latency. Under bursty load, those GPUs may be better used as
independent data-parallel replicas serving different requests. The objective is
to decide whether and when to transform an in-flight CP/SP request into a
DP-style multi-request layout at denoising step boundaries, while accounting for
state migration, communicator readiness, cache compatibility, and tail-latency
risk.
```

## Cost Model Skeleton

Let:

- `S_remain`: remaining denoising steps for the active request;
- `T_sp2`: per-step latency for one request on two-GPU CP/SP;
- `T_dp1`: per-step latency for one request on one GPU;
- `C_switch`: one-time cost to transform layout;
- `Q_wait`: expected queueing delay for the newly arrived request if no switch
  happens;
- `C_tail`: penalty for increasing the active request tail latency.

Switching is plausible when:

```text
Q_wait_saved + throughput_gain_over_remaining_steps
    >
C_switch + C_tail + cache_or_graph_penalty
```

The first simulator can sweep `C_switch` instead of measuring it precisely.

## First Baselines to Compare

- `static_dp`: two one-GPU replicas, never CP/SP.
- `static_sp`: one two-GPU CP/SP request at a time, queue the rest.
- `admission_control`: if the queue is empty, wait a small batching window
  before deciding DP vs CP/SP.
- `early_hot_switch`: start CP/SP immediately; if another request arrives
  before an early-step threshold, switch to DP-style replicas.
- `oracle`: choose the best layout with future arrival knowledge.
