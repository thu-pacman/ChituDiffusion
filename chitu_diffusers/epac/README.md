# EPAC Core

`chitu_diffusers.epac` contains only model-independent Elastic Parallel Caching
Engine code. It must not import Z-Image, FastAPI, torchrun launchers, or the old
ChituDiffusion backend.

## Ownership

| Module | Responsibility |
| --- | --- |
| `request.py` | Framework-neutral request, status, result, and metrics |
| `adapters/` | Model adapter protocol and registry |
| `engine.py` | Request-local state and synchronous reference execution |
| `image_decoder.py` | Stage world, executor, transfer, completion, and embedded API types |
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
