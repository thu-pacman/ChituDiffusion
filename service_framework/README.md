# ChituDiffusion Service Framework

This is a lightweight web UI plus persistent GPU worker around the existing
`system_config.yaml` and `run.sh` launch path. The HTTP UI is opened by the
launcher process on the VSCode remote host so VSCode can detect and forward the
port. The GPU worker is started through `run.sh`, allocates resources, loads the
model, then marks the UI ready for requests.

The static launch fields stay in the selected system config:

- hardware resources: `launch.num_nodes`, `launch.gpus_per_node`, `parallel.cfp`
- model loading: `model.name`, `model.ckpt_dir`
- attention backend: `infer.attn_type`

The browser can be opened immediately, but request submission is disabled until
the GPU worker reports ready. Requests are processed serially by the already
loaded instance. The browser can override only request fields and output/logging
fields such as `output.run_log`, `output.memory`, `output.timer`, and
`output.log_ranks`.

## Remote Cluster Start

```bash
./.venv/bin/python service_framework/server.py --config system_config.yaml --host 127.0.0.1 --port 7860
```

This command opens the UI on the VSCode remote host and generates a temporary
service config under `service_framework/runs/`, then launches the GPU worker:

```bash
bash run.sh service_framework/runs/<service_id>/system_config.yaml
```

The temporary config points `launch.python_script` at
`service_framework/persistent_service.py`. The page shows a loading state until
the worker has completed `chitu_init()`.

With VSCode Remote, the `Service UI: http://127.0.0.1:7860` line should trigger
the usual local-open/port-forward prompt. Without VSCode forwarding, keep the
service bound to `127.0.0.1` on the cluster and open an SSH tunnel from your
local machine:

```bash
ssh -L 7860:127.0.0.1:7860 <cluster-login-host>
```

Then open this URL in your local browser:

```text
http://127.0.0.1:7860
```

If port `7860` is already used on the cluster, either stop the old service,
choose another port, or let the service pick the next free port:

```bash
./.venv/bin/python service_framework/server.py --config system_config.yaml --host 127.0.0.1 --port 7860 --auto-port
```

Use the actual printed port in the SSH tunnel command.

If the local browser cannot load the page, first verify the service from the
remote terminal:

```bash
curl http://127.0.0.1:7860/api/config
```

If this returns JSON, the service is healthy on the cluster and the missing
piece is local forwarding. In VSCode Remote, open the Ports panel and manually
forward remote port `7860`, then open the forwarded local URL. If using plain
SSH, keep the `ssh -L 7860:127.0.0.1:7860 <cluster-login-host>` tunnel open.

Each submitted request creates:

- `service_framework/runs/<job_id>/request.json`
- a normal ChituDiffusion output directory under `output.root_dir`
