# EPAC Serve

This directory contains service and launcher concerns separated from EPAC core
and model implementations:

- `protocol.py`: HTTP request and response schemas;
- `app.py`: FastAPI routes;
- `config.py`: stage/service configuration parsing;
- `diffusion_runtime.py`: the compatibility distributed runtime (model operations
  are injected through `DiffusionBackend`);
- `zimage.py`: conversion from public serve config to the generic runtime;
- `torchrun.py`: process lifecycle, signal handling, and HTTP startup.
- `embedded.py`: host-owned stage lifecycle (`start/submit/poll/cancel/stop`).

A `DiffusionBackend` supplies model-specific request preparation, denoise,
state-transfer, and finalization hooks to EPAC worker pools. Static-DP uses
independent singleton consumers, static-CP uses one full-world CP consumer, and
elastic uses rank-local progress exchange plus pulse-time lane repartitioning.
The Gloo control transport and scheduling broker live in `chitu_diffusion.epac`;
this package owns only service lifecycle and connectors.

`EmbeddedDiffusionRuntime` is the primary runtime boundary. `torchrun.py` starts
it and attaches FastAPI as a connector; SGLang-Omni can instead translate its
inbox/outbox directly to `submit()` and `poll()`. The compatibility
`EpeDiffusionServiceRuntime` consumes a model-neutral `DiffusionBackend`; new
host integrations should use `EmbeddedDiffusionRuntime`.
