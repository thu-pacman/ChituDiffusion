# EPE Serve

This directory contains service and launcher concerns separated from EPE core
and model implementations:

- `protocol.py`: HTTP request and response schemas;
- `app.py`: FastAPI routes;
- `config.py`: stage/service configuration parsing;
- `runtime.py`: model-independent distributed runtime;
- `runner.py`: conversion from public serve config to the generic runtime;
- `torchrun.py`: process lifecycle, signal handling, and HTTP startup.
- `embedded.py`: host-owned stage lifecycle (`start/submit/poll/cancel/stop`).

A `DiffusionBackend` supplies model-specific request preparation, denoise,
state-transfer, and finalization hooks to EPE worker pools. Static-DP uses
independent singleton consumers, static-CP uses one full-world CP consumer, and
elastic uses rank-local progress exchange plus pulse-time lane repartitioning.
The Gloo control transport and scheduling broker live in `chitu_diffusion.epe`;
this package owns only service lifecycle and connectors.

`DiffusionServiceRuntime` owns queueing and distributed execution.
`EmbeddedDiffusionRuntime` provides the host-owned lifecycle; `torchrun.py`
starts it and attaches FastAPI, while another host can call `submit()` and
`poll()` directly.
