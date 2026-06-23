# Simulator Plan

The first implementation target should be an offline simulator, not a runtime
hot-switch prototype. The simulator decides whether the feature has a useful
operating region before we pay the runtime complexity cost.

## Inputs

- Per-step latency:
  - `sp_2gpu_step_ms`: one request with two-GPU CP/SP.
  - `dp_1gpu_step_ms`: one request on one GPU.
  - optional `sp_1gpu_step_ms` if a model-specific baseline differs from DP.
- Workload:
  - request arrival timestamps;
  - number of denoising steps per request;
  - optional priority or deadline.
- Switch model:
  - `switch_cost_ms`;
  - `switch_allowed_until_step`;
  - `graph_recapture_cost_ms`;
  - `cache_migration_cost_ms`;
  - `communicator_cost_ms`.

## Policies

- `static_dp`: keep two independent one-GPU replicas.
- `static_sp`: use both GPUs for one CP/SP request and queue the rest.
- `admission_control`: wait a short window before deciding DP or CP/SP.
- `early_hot_switch`: start CP/SP, then switch to DP if another request arrives
  before the early threshold.
- `oracle`: choose the best action with future arrivals, for an upper bound.

## Outputs

- mean latency;
- P50/P95/P99 latency;
- queueing delay;
- GPU busy-time proxy;
- switch count;
- wasted or migrated work;
- policy trace for debugging.

## Next Code Artifact

Add `simulate.py` with a deterministic event simulator and a small JSON example
trace. The first run can use synthetic Poisson arrivals and ChituBench-derived
step latencies.

