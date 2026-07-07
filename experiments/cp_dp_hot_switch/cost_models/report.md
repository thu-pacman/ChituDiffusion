# Cost Models (per-hardware) — CP/DP Hot-Switch

This directory holds **hardware-specific** decoupled cost models (worker roofline +
communication) for Z-Image, produced by
[`../profile_worker.py`](../profile_worker.py). The simulator and the static
baseline runner select one **by parameter** so the same trace/policy analysis can
be re-run on different GPUs without editing code.

| file | hardware | interconnect | provenance |
| --- | --- | --- | --- |
| [`h20.json`](h20.json) | NVIDIA H20 (bf16) | NVLink | M1 calibration (copy of the legacy top-level `cost_model.json`) |
| [`rtx4090.json`](rtx4090.json) | 4× NVIDIA RTX 4090 48GB (bf16) | PCIe P2P (no NVLink) | re-profiled from `/dockerdata/Z-Image` on this machine |

The legacy top-level `../cost_model.json` is kept unchanged (H20) so existing
report links keep working; `h20.json` here is an identical copy.

## How the simulator reads it (by parameter)

Both entry points accept `--cost-model`, which takes either a **hardware name**
(resolved to `cost_models/<name>.json`) or an **explicit JSON path**. Omitting it
falls back to the legacy `cost_model.json`.

```bash
# Hot-switch policy simulator (Z-Image profiles derived from the cost model)
python3 simulate.py --cost-model rtx4090
python3 simulate.py --cost-model h20

# All static + dynamic serve policies + GPU-usage SVG timelines
python3 run_serve_policies.py --cost-model rtx4090 --output-dir out/serve_policies_4090
```

Resolution order for the value: existing file path → `cost_models/<name>.json` →
`cost_models/<name>` → treated as a literal path.

## Reproduce a hardware profile

```bash
# 1) DP (k=1) compute grid — single GPU, real Z-Image DiT forward
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  experiments/cp_dp_hot_switch/profile_worker.py --mode compute \
  --ckpt /dockerdata/Z-Image \
  --out experiments/cp_dp_hot_switch/cost_models/rtx4090.json

# 2) Communication (all-to-all) calibration — multi-GPU via torchrun
torchrun --nproc_per_node=4 \
  experiments/cp_dp_hot_switch/profile_worker.py --mode comm \
  --out experiments/cp_dp_hot_switch/cost_models/rtx4090.json
```

`--mode comm` merges into the same file, so run compute first, then comm.

## Calibrated numbers

### Compute grid (single GPU, bf16, flash_attn), median ms / denoise-step forward

**RTX 4090 (48GB)** — B=1 is already compute-bound (`batch_saturation` knee = 1 at
every resolution; batching scales ~linearly, no memory-bound plateau):

| resolution | seq_len | B=1 | B=2 | B=4 | B=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512×512 | 1536 | 147.4 | 305.9 | 679.5 | 1368.5 |
| 768×768 | 2816 | 296.6 | 648.0 | 1319.2 | 2615.3 |
| 1024×1024 | 4608 | 543.3 | 1131.1 | 2245.9 | 4478.8 |
| 1280×1280 | 6912 | 898.1 | 1791.2 | 3556.8 | 7075.7 |
| 1536×1536 | 9728 | 1344.1 | 2678.2 | 5314.0 | 10603.1 |

For reference, H20 B=1 is 184.3 / 350.8 / 596.9 / 975.4 / 1518.1 ms at the same
resolutions — i.e. **this 4090 is ~1.1–1.25× faster in raw DiT compute** for these
shapes.

### Communication (Ulysses all-to-all)

| hardware | effective all-to-all BW | note |
| --- | ---: | --- |
| H20 | 123.4 GB/s | single-node NVLink |
| RTX 4090 | **14.2 GB/s** | PCIe P2P, no NVLink — ~9× lower |

### Composed step latency @ B=1: DP(k=1) vs SP(k=2) vs SP(k=4) — RTX 4090

Parenthesized = speedup vs k=1; `c=` = per-step comm cost (ms).

| resolution | img_tok | k=1 | k=2 | k=4 |
| --- | ---: | ---: | ---: | ---: |
| 512×512 | 1024 | 147.4 | 156.9 (×0.94, c=9.5) | 154.7 (×0.95, c=7.3) |
| 768×768 | 2304 | 296.6 | 182.9 (×1.62, c=20.6) | 163.0 (×1.82, c=15.6) |
| 1024×1024 | 4096 | 543.3 | 302.8 (×1.79, c=36.0) | 174.6 (×3.11, c=27.2) |
| 1280×1280 | 6400 | 898.1 | 475.9 (×1.89, c=56.0) | 256.6 (×3.50, c=42.1) |
| 1536×1536 | 9216 | 1344.1 | 702.4 (×1.91, c=80.3) | 357.0 (×3.77, c=60.4) |

## Key findings (RTX 4090 vs H20)

1. **No NVLink dominates SP economics.** At 14.2 GB/s the all-to-all cost is
   ~9× higher than H20's NVLink. It is still < a few % of a 1024²+ step (compute is
   large), so SP remains a strong latency lever at high resolution — but at **512²
   SP is net-negative** (k=2 → ×0.94), because comm (~9.5 ms/step) exceeds the
   compute saved by sharding a short sequence.
2. **The DP↔SP crossover point is essentially unchanged** on the serving traces
   (see `out/baselines_4090` vs `out/baselines_h20`): low-res / low-load traces
   still prefer independent **DP** replicas; high-res / high-load traces still
   prefer **SP4** for tail latency + SLO attainment. The higher 4090 comm cost
   nudges the balance toward DP but does not flip any trace's best policy.
3. **Raw throughput is slightly higher on 4090** for compute-bound shapes because
   its per-step DiT compute is faster (e.g. `poisson_512_r6n16` DP 1.067 → 1.328
   req/s; `poisson_1024_r3n12` DP 0.330 → 0.362 req/s).
4. Practical takeaway for the hot-switch scheduler: on 4090-class boxes, the
   **SP-vs-DP decision should weight comm more heavily for short/low-res requests**
   — exactly the shape-aware ratio the experiment targets. Re-profile comm per box
   since PCIe topology varies.

## GPU-usage visualization (this hardware)

`run_serve_policies.py --cost-model rtx4090` emits per-(trace, policy) execution
timelines showing GPU lanes (compute blocks vs idle, dark blocks for CP↔DP
switches) and the request arrival→queue→compute→finish lifecycle:

- Markdown index + inline SVGs: `out/serve_policies_4090/serve_policy_timelines.md`
- Individual SVGs: `out/serve_policies_4090/serve_policy_timelines/*.svg`
- Aggregate metrics table: `out/serve_policies_4090/serve_policies.md`
- Machine-readable: `out/serve_policies_4090/serve_policies.json`

(`out/` is gitignored; regenerate with the command above.)
