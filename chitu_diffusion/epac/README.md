# EPAC Core

[English](README.md) | [简体中文](README.zh-CN.md)

`chitu_diffusion.epac` contains only model-independent Elastic Parallel Caching
Engine code. It must not import Z-Image, FastAPI, torchrun launchers, or the old
ChituDiffusion backend.

EPAC turns a fixed distributed stage into one or more context-parallel lanes.
It profiles execution costs at startup, plans request placement against
throughput and SLO objectives, and can resize lanes at pulse boundaries. Static
generation and persistent serving use the same executor and request state
contract.

## Design principles

- **Model-independent core:** model tensor conventions belong in
  `chitu_diffusion.models`, never in the planner or worker protocol.
- **One executor contract:** static CP, static DP, and elastic scheduling differ
  in lane constraints, not in model execution semantics.
- **Measured decisions:** the planner uses startup and online measurements
  keyed by workload shape and lane width.
- **Collective safety:** every rank in a lane receives the same lease and
  executes identical distributed control flow.
- **Explicit capabilities:** unsupported scheduling, transfer, or decode modes
  fail early instead of falling back to a hidden pipeline call.

## Execution model

```text
request queue
    |
request profile + measured cost table
    |
EpeSchedulingPolicy
    |
StepPlan(request, lane_ranks, K)
    |
PulseCoordinator ---- LaneLease ---- lane-local workers
    ^                                      |
    +------------- LaneReport -------------+
```

## Ownership

| Module | Responsibility |
| --- | --- |
| `request.py` | Framework-neutral request, status, result, and metrics |
| `adapters/` | Model adapter protocol and registry |
| `engine.py` | Request-local state and synchronous reference execution |
| `api.py` | Shared typed-request helpers and synchronous Diffusers facade |
| `image_decoder.py` | Stage world, executor, transfer, completion, and embedded API types |
| `model_executor.py` | Reusable executor lifecycle and stage-world construction |
| `model_scheduling.py` | Measured cost state, online calibration, and planner facade |
| `executor.py` | One denoise step and optimization boundaries |
| `optimization.py` | FlexCache-compatible model/prediction hooks |
| `cost.py` | Startup-measured sequence-length/lane-width cost table |
| `scheduling.py` | Request profiles, step plans, and lane constraints |
| `epe.py` | Cost- and SLO-aware layout planner |
| `pulse.py` | Epoch, lane lease, deadline, lane report, and pull protocol |
| `worker.py` | Lane-local minimum-K, opportunistic-step, and pull loop |
| `worker_pool.py` | Fixed-lane pools and point-to-point rank exchange transport |
| `lane_broker.py` | Rank-local progress aggregation, pull, and pulse replanning |
| `timeline.py` | Buffered rank-local execution events and shutdown-time trace flush |

The public integration boundary is the executor lifecycle in
`model_executor.py` and `image_decoder.py`. See
[`../INTEGRATING_IMAGE_DECODER.md`](../INTEGRATING_IMAGE_DECODER.md) before
adding a model adapter.

## Unified Strategies

The strategies are constraints over one planner and one executor, not separate
runtimes:

| Strategy | Allowed lane widths | Running-lane rule |
| --- | --- | --- |
| `static_dp` | `{1}` | Keep every running request on its singleton lane |
| `elastic` | Configured divisors, for example `{1,2,4}` | Resize at a pulse |
| `static_cp` | `{world_size}` | Run one full-world lane |

`LaneConstraints.create()` is the only place that maps a strategy name to these
topology limits. Model or serve code must not implement another static-DP/CP
planner branch.

## Pulse Protocol

A pulse is a control-plane epoch. It is not a `dist.barrier(WORLD)` after a
fixed number of steps.

1. `PulseCoordinator.open()` asks the common scheduling policy for the initial
   request/lane/K layout.
2. Every disjoint rank set receives a `LaneLease` with a shared wall-clock
   deadline. Unassigned ranks receive an idle lease and can pull work.
3. A lane executes its minimum K, then may run extra steps while
   `LaneLease.can_start_step()` predicts completion before expiry.
4. If a request completes, the lane asks the coordinator for another globally
   ordered request through `choose_pull_request()`.
5. At lease expiry, each lane leader submits one epoch-checked `LaneReport`.
   Only after all reports arrive can the next pulse reclaim and repartition all
   ranks.

The protocol therefore permits producer-consumer work between pulses without
making the lane pattern permanent. CP collectives remain synchronous inside a
lane; lane leaders participate in the pulse control plane.

## Current State

The typed pulse/lease/report protocol, deadline admission, pull ordering,
lane-local producer-consumer loop, and unified strategy constraints are
implemented and covered by CPU tests. Static-DP is connected to the distributed
singleton-lane worker pool: every rank completes a request and immediately
pulls the next globally ordered item without a world collective.

Static-CP uses the same work/result contract with a full-world CP lane. Every
rank executes the DiT collectively, the lane leader publishes the result, and
the next request starts without opening a pulse.

Elastic uses `DistributedRankExchange` and `PulseLaneBroker`. Rank 0 aggregates
progress per lane rather than through a world collective. A lane may execute
extra steps or pull a pending request before its lease expires; after every
lease reports, the cost/SLO planner can repartition cp1/cp2/cp4 lanes. A resize
transfers latents only from the previous owner to newly joined ranks.

`default_deadline_ms` defines the default end-to-end SLO from request arrival;
an explicit request `deadline_ms` overrides it. The service config reserves
`deadline_guard_ms` (default 3000ms) from each deadline before scoring layouts.
Startup warmup currently measures DiT steps, while control, state transfer,
VAE, D2H, and CPU publication still consume wall time; the guard prevents
throughput work from consuming that unmodeled tail. The client-visible deadline
is unchanged.

The elastic policy keeps the complete pending queue visible while enforcing
`max_inflight_requests` only at state admission. For every candidate layout it
forward-predicts queue completion with the measured cost table, then compares
SLO misses and tardiness before starvation, concurrent admission, cp1-normalized
useful throughput, slowdown, and flow time. Normalized throughput is
`sum(T_cp1 / T_lane)`, so raw short-sequence steps cannot monopolize the pool.
This rolling prediction is recomputed at every pulse.

Dense queues use a bounded layout search. The planner first inserts
deterministic cp-width/fairness layouts, then explores at most 256 unique
candidates; rank permutations that are equivalent for requests without an
existing placement are deduplicated. `last_plan_stats` exposes queue size,
candidate count, and planning time, and the distributed runtime records those
fields on each `pulse_wait` timeline event.

Startup cost profiling requires at least three measured steps so a cold first
forward cannot define the median. Online calibration is exact-keyed by sequence
length, lane width, batch size, and condition count. An unseen key uses the
startup table (including its interpolation/extrapolation), while the first
lane-command observation enables correction for the next scheduling pulse.
Updates inside a 5% relative-error deadband retain the current factor; factors
from other resolutions are never used as a fallback.

## Generate and serve behavior

`generate` creates one full-world static-CP lane and performs no warmup or
elastic planning. Every rank enters the executor; the leader owns the final
Diffusers result.

`serve` performs startup profiling, admits requests into request-local state,
and repeatedly plans work at pulse boundaries. The stage leader owns ingress
and completion publication while all ranks participate in lane execution.

```bash
# One distributed request.
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate --model zimage --model-path /path/to/Z-Image \
  --output outputs/zimage.png

# Persistent runtime from a native stage configuration.
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  serve --stage-config /path/to/stage.yaml
```

## Extending EPAC

For a new model family:

1. Implement the shared executor operations under
   `chitu_diffusion/models/<family>/`.
2. Declare capabilities and normalize public requests without leaking model
   fields into EPAC scheduling types.
3. Use lane ranks supplied by `StepPlan`; never cache a fixed CP world size in
   the model backend.
4. Keep scheduler, latents, timesteps, and step cursor request-local.
5. Add CPU contract tests, then validate single-GPU and multi-rank output
   against the native Diffusers pipeline.

Changes to scheduling should extend `LaneConstraints` or the common policy.
Do not create model-specific static-DP, static-CP, or elastic planner branches.

## Validation

From the repository root:

```bash
python -m pytest -q
python -m ruff check chitu_diffusion/epac chitu_diffusion/models test
```

CPU tests cover request transitions, lane constraints, planner ordering,
leases, reports, transfer contracts, and executor lifecycle. GPU acceptance
must additionally verify collective order, deterministic request ownership,
same-seed quality, lane resize, and leader-only result publication.

## Current limitations

- Arbitrary rank failure and distributed fatal-state recovery are not yet
  implemented.
- Running-request cancellation is not yet broadcast at denoise-step
  boundaries.
- Startup and online cost calibration model known workload keys; deployments
  should warm the resolutions and lane widths they intend to serve.
- FlexCache is deliberately disabled for persistent EPE serving. It is a
  request-local optimization for static `generate`.

EPAC is therefore a developer-preview runtime, not a production-GA fault
tolerance layer.
