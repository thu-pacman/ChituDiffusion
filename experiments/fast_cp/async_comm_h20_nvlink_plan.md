---
name: Async QKV-Comm Overlap on Slurm H20 NVLink
overview: Turn the existing async-comm sketch into a Slurm/H20 execution plan. Debug the overlap in isolation first via a standalone harness that drives the real model-agnostic CP core with synthetic tensors; prove the NVLink baseline, bit-equivalence, and overlap ceiling there; then implement the smallest safe AGKV/Ulysses projection-communication overlap behind a flag, migrate it into Z-Image, and validate with correctness, wall-clock, CP profile, and torch trace evidence. CUDA-graphing the overlap schedule is a later phase to cut CPU launch jitter and tighten cross-rank collective alignment.
todos:
  - id: h20_baseline
    content: Capture Slurm/H20 topology, P2P/NVLink status, NCCL path, and serial CP baselines for AGKV/Ulysses/unified.
    status: pending
  - id: trace_baseline
    content: Produce pre-change torch traces showing the serial QKV -> collective -> attention schedule and collect CP profile buckets.
    status: pending
  - id: agkv_mvp
    content: Implement the minimal CHITU_CP_OVERLAP_QKV AGKV path: K projection -> K all-gather on comm stream, V/Q projection on compute stream, event wait, serial attention.
    status: pending
  - id: ulysses_mvp
    content: Implement input-side Ulysses overlap only: stagger K/V/Q projections with their all-to-all collectives; leave output collectives serial in phase 1.
    status: pending
  - id: validation
    content: Run correctness, trace, and Slurm benchmark matrix; compare overlap on/off under the same node allocation.
    status: pending
  - id: harness
    content: Build a standalone CP-core harness in experiments/fast_cp/harness/ that drives the real ImageContextParallelAttention with synthetic tensors for bit-equiv / trace / wall-clock A/B, before touching Z-Image.
    status: pending
  - id: cudagraph
    content: After eager overlap is proven, capture the AGKV/Ulysses overlap schedule (comm stream + events + NCCL) into a CUDA graph to cut CPU launch jitter and tighten cross-rank collective alignment.
    status: pending
  - id: decide_next
    content: Decide from H20 data whether to extend to output-collective overlap, fp8 KV, KV cache, unified, Qwen-Image, or CUDA graph/compile integration.
    status: pending
isProject: false
---

# H20 NVLink Async Comm Plan

## 0. What changes from the PCIe plan

The PCIe worklog established that no-P2P PCIe is communication-bound and that overlap only hid about 4% wall clock. On Slurm-managed H20 NVLink, the assumption flips:

- NVLink should make K/V all-gather and Ulysses all-to-all much shorter than PCIe, so Q/K/V projection, norm, RoPE, and launch overhead become a larger fraction of the CP attention path.
- The target is not "move fewer bytes" first. The target is "hide the remaining collective latency behind projection work" while keeping the current exact CP math and existing serial path untouched.
- We should not assume theoretical H20 bandwidth. First record the actual Slurm node topology, P2P status, NCCL path, and profile buckets under the same job allocation that will be used for A/B.

This plan complements `async_comm.md`; it is the execution plan for the H20 Slurm environment.

## 1. Success criteria

Minimum acceptance:

- Correctness: `CHITU_CP_OVERLAP_QKV=1` matches `=0` for the same CP mode, seed, size, steps, and guidance. For short runs, image PSNR vs serial CP should be effectively unchanged; tensor-level checks are preferred if we add a small unit harness.
- Trace evidence: with `CHITU_TORCH_TRACE=/path/trace.json`, rank 0 trace shows NCCL all-gather/all-to-all kernels on a non-default stream overlapping with Q/V/QKV projection GEMMs or norm/RoPE kernels on the compute stream.
- Wall-clock: H20 cp4 or cp8 Z-Image 1024x1024 guidance=4, 50 steps improves beyond run-to-run noise. Use at least 3 repeated runs inside the same Slurm partition. A practical go/no-go threshold is >=3% denoise or end-to-end improvement for AGKV, or a clearly positive attention-stage reduction with neutral end-to-end.
- Fallback safety: profiling, fp8 KV, KV-cache stale steps, ring/ring_graph/unified, masks, non-CUDA, and unsupported head divisibility all continue through the existing serial implementation.

Stretch:

- Ulysses input-side overlap is positive or neutral.
- The implementation can be reused by Qwen-Image later through the model-agnostic `ImageContextParallelAttention` core.

## 1.5 Harness-first workflow (debug async in isolation, then migrate)

Do **not** debug the async schedule against a full Z-Image denoise run. The CP
core is model-agnostic and self-contained: `ImageContextParallelAttention`
([chitu_diffusion/modules/attention/image_cp_attention.py](../../chitu_diffusion/modules/attention/image_cp_attention.py))
needs only

- a comm `group` exposing `group_size`, `gpu_group`, `rank_in_group`, and
  `all_to_all(t, scatter_dim, gather_dim)` -- exactly what the real
  `CommGroup` provides, and what a tiny harness stand-in can wrap around a raw
  `torch.distributed` process group; and
- an `attn_backend(q, k, v, causal=False) -> (out, ...)` callable -- a plain
  SDPA/flash call on packed `[B, S, H, D]`.

So we can build a standalone harness that feeds **synthetic** packed Q/K/V into
the *real* core and exercises `_attn_agkv` / `_attn_ulysses` (serial) vs the new
`attention_fused()` (overlap) without loading any model weights, scheduler, or
config. All async debugging, profiling, and A/B happens here in
`experiments/fast_cp/harness/` until overlap is proven; only then do we
wire `attention_fused()` into `z_image_attention.py` and re-validate on the real
workload.

Why this is the right boundary:

- **Fast iteration**: a single attention call per step instead of a 50-step
  diffusion loop. Seconds, not minutes, per A/B.
- **Clean correctness signal**: the harness fixes the seed and compares serial
  vs overlap outputs directly (max-abs-diff / PSNR), so a stream-ordering or
  `record_stream` bug shows up as a numerical mismatch immediately, with no
  model noise on top.
- **Honest profiling**: the harness owns its own timing/trace and never calls
  the synchronized CP `profile` path, so overlap is not defeated by measurement.
- **Reusable on H20**: the same `torchrun`/`srun` launch that runs Z-Image can
  run the harness on the target partition, giving a topology-faithful overlap
  ceiling before any model integration.

Harness layout (see `harness/README.md`):

```text
experiments/fast_cp/harness/
  cp_async_harness.py   # builds synthetic Q/K/V, drives the real CP core
  run_harness.sh        # torchrun / srun launcher (cp4, cp8, ...)
  README.md             # how to run, what each mode/flag means
```

What the harness must cover (mirror of section 1 success criteria, but at the
core level):

- correctness: `--check` asserts overlap output == serial output for the same
  synthetic seed (target max-abs-diff ~0, PSNR > 60 dB) per mode.
- trace: `CHITU_TORCH_TRACE=...` produces a rank-0 chrome trace where the NCCL
  all-gather / all-to-all kernel runs on a side stream concurrently with the
  next projection GEMM (here, a synthetic projection matmul stands in for the
  real Q/K/V proj).
- wall-clock: `--bench` reports serial vs overlap median over N iters at
  realistic shapes (Z-Image 1024x1024: `S_img`, `S_txt`, `H`, `D` taken from the
  actual config) for cp4 / cp8.

Migration gate: only after the harness shows (a) bit-equivalence and (b) visible
overlap in the trace and (c) a non-negative wall-clock delta on H20, port
`attention_fused()` into the Z-Image processor and repeat the section 6 matrix.

## 2. Slurm/H20 preflight

Run these once per target partition/node type and save logs under `outputs/h20_async_comm_preflight/`.

Topology and CUDA/NCCL sanity, inside a Slurm allocation:

```bash
export SRUN_PARTITION=<h20-partition>
export SRUN_JOB_NAME=h20-async-preflight
export SRUN_EXTRA_ARGS="--exclusive"

script/srun_direct.sh 1 8 test/test_z_image.py \
  models=Z-Image \
  models.ckpt_dir=<Z-Image-ckpt> \
  infer.diffusion.cfg_size=1 \
  infer.diffusion.cp_size=8 \
  infer.diffusion.up=8 \
  infer.attn_type=flash_attn \
  output.timer=True \
  output.memory=False \
  output.run_log=True
```

Also run a tiny standalone job or add a temporary preflight command to capture:

```bash
nvidia-smi topo -m
nvidia-smi nvlink -s || true
python3 - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
n = torch.cuda.device_count()
print("device_count", n)
for i in range(n):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
print("p2p")
for i in range(n):
    print(i, [torch.cuda.can_device_access_peer(i, j) for j in range(n)])
PY
```

NCCL logging for one short run:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL
export TORCH_NCCL_ENABLE_MONITORING=0
```

Keep the log. We need to know whether NCCL is using NVLink/P2P inside the node and whether Slurm rank order maps local ranks to the expected GPU order. If rank order is not topology-friendly, fix that before measuring overlap.

## 3. Baseline matrix before code changes

Use the current serial code. Keep `CHITU_CP_OVERLAP_QKV` unset.

For Z-Image, use the same prompt/seed already in `test/test_z_image.py`:

- size: `CHITU_Z_IMAGE_SIZE=1024,1024`
- steps: `CHITU_Z_IMAGE_STEPS=50` for final numbers, `4` or `8` for iteration
- guidance: existing Z-Image config path should keep guidance=4 unless explicitly overridden
- output: `output.timer=True`, `output.memory=False`, `output.run_log=True`

Collect two kinds of baselines:

1. Synchronized CP profile, for analytical upper bound:

```bash
export CHITU_Z_IMAGE_CP_PROFILE=1
export CHITU_Z_IMAGE_CP_MODE=agkv   # repeat: ulysses, unified
export CHITU_Z_IMAGE_CP_UP=8        # for pure Ulysses/unified as needed
chitu run <h20-zimage-config.yaml> --gpus-per-node 8 --cfp 1
```

Use the log lines from `_report_cp_profile()`:

- `qkv_seconds`: maximum input-side work that can hide communication
- `kv_seconds` or `a2a_seconds`: communication that can be hidden
- `attn_seconds`: should remain almost unchanged
- `out_seconds`: not part of phase-1 overlap

Expected upper bound for phase 1:

```text
AGKV removable per layer <= min(K_gather, V_proj + Q_proj + norm/rope gap)
Ulysses removable per layer <= min(a2a_K + a2a_V + a2a_Q exposed gaps, staggered projection windows)
```

2. Unsynchronized wall-clock and trace:

```bash
unset CHITU_Z_IMAGE_CP_PROFILE
export CHITU_TORCH_TRACE=$PWD/outputs/h20_async_comm_traces/agkv_serial_rank0.json
export CHITU_Z_IMAGE_CP_MODE=agkv
chitu run <h20-zimage-config.yaml> --gpus-per-node 8 --cfp 1
```

Repeat for `ulysses`. The serial trace should show QKV projection before the collectives. This gives the visual baseline for the later overlap trace.

## 4. Implementation shape

Keep the change model-agnostic in `image_cp_attention.py`, with only closure construction in `z_image_attention.py`.

### 4.1 Core API

Add:

```python
def attention_fused(self, *, img_len, produce_q, produce_k, produce_v):
    ...
```

where each closure returns the current packed local tensor `[B, image_chunk + text, H, D]`, already projected, normalized, RoPE-applied where relevant, cast to value dtype, and contiguous. The core owns splitting, collectives, stream/event ordering, and fallback dispatch.

Fallback if any of these is true:

- `not self.enabled`
- `self.profile`
- `not self._overlap_qkv`
- `self.mode not in {"agkv", "ulysses"}`
- `self._fp8_kv`
- `self._kvcache_enabled`
- mask path is not the fast path
- tensors are not CUDA
- Ulysses heads not divisible by CP size

The fallback calls closures in the original order and then calls existing `attention(...)`, preserving behavior.

### 4.2 Stream helpers

Add lazy helpers:

```python
def _get_comm_stream(self):
    if self._comm_stream is None:
        self._comm_stream = torch.cuda.Stream(priority=0)
    return self._comm_stream

def _record_for_comm(self, tensor, stream):
    tensor.record_stream(stream)
    return tensor

def _event_after(self, stream):
    ev = torch.cuda.Event()
    ev.record(stream)
    return ev
```

Stream discipline:

- Before launching a collective on `comm_stream`, call `comm_stream.wait_stream(current_stream)` so the projected tensor is ready.
- For every tensor produced on compute and consumed by comm, call `record_stream(comm_stream)`.
- For every tensor produced on comm and consumed by compute, make compute wait on a CUDA event recorded on `comm_stream`, then call `record_stream(current_stream)`.
- Do not use `torch.cuda.synchronize()` in the overlap path.
- Do not time overlap sub-ops with the existing synchronized `time()` / `_comm_time()`.

### 4.3 AGKV MVP schedule

Use separate K/V/Q projections instead of the current combined `_qkv_proj()`.

The schedule should be:

```text
compute stream: produce_k
comm stream   : wait compute -> all_gather K image
compute stream: produce_v while K gather runs
comm stream   : after K gather launch, wait compute -> all_gather V image
compute stream: produce_q while V gather runs
compute stream: wait K/V events -> concat -> attention
```

Important details:

- Gather only image K/V. Text K/V remain local replicated slices from the packed K/V.
- `produce_k()` and `produce_v()` return packed `[image, text]`; split inside core using `img_len`.
- Existing `all_gather_seq()` allocates list pieces and calls `dist.all_gather`; it can be reused inside `with torch.cuda.stream(comm_stream)` because NCCL launch binds to the current stream.
- In phase 1, keep bf16 only. `CHITU_CP_FP8_KV=1` falls back because fp8 adds scale all-gathers and local-bf16 substitution that complicate stream ownership.
- KV-cache falls back. Stale steps have no gather, so the overlap schedule is not the right abstraction.

### 4.4 Ulysses MVP schedule

Only overlap input all-to-all with projection. Keep output image all-to-all and text all-gather-heads serial after attention.

Schedule:

```text
compute stream: produce_k
comm stream   : wait compute -> all_to_all K image
compute stream: produce_v while K a2a runs
comm stream   : wait compute -> all_to_all V image
compute stream: produce_q while V a2a runs
comm stream   : wait compute -> all_to_all Q image
compute stream: wait Q/K/V a2a events -> slice text heads -> attention
compute stream: output img all_to_all -> txt all_gather_heads, serial as today
```

Reason to keep output serial in phase 1:

- Both output collectives use the same CP communicator as the input collectives.
- Concurrent collectives on one communicator are easy to get wrong if launch order diverges across ranks.
- Output overlap opportunity is smaller and less directly tied to QKV projection.
- Once input-side overlap is proven, output overlap can be tested with a dedicated flag, not mixed into the MVP.

### 4.5 Z-Image closure construction

In `z_image_attention.py`, replace the eager:

```python
query, key, value = self._cp.time(_qkv_proj, "qkv")
...
self._cp.attention(...)
```

with two paths:

- serial/profile path: preserve the current code exactly, including the `qkv` timing bucket
- overlap fast path: build `produce_q`, `produce_k`, `produce_v`

Closure helper:

```python
def _produce(which):
    proj = {"q": attn.to_q, "k": attn.to_k, "v": attn.to_v}[which](hidden_states)
    x = proj.unflatten(-1, (attn.heads, -1))
    if which == "q" and attn.norm_q is not None:
        x = attn.norm_q(x)
    if which == "k" and attn.norm_k is not None:
        x = attn.norm_k(x)
    if which in ("q", "k") and freqs_cis is not None:
        x = self._apply_rotary_emb(x, freqs_cis)
    if which in ("q", "k"):
        x = x.to(value_dtype).contiguous()
    else:
        x = x.contiguous()
    return x
```

Because `value_dtype` is only known after V projection in the current code, choose one of:

- use `hidden_states.dtype` / projection output dtype as the cast target, and assert it matches V dtype on first overlap call; or
- have `produce_v()` set a local dtype cache before Q/K cast. This is more awkward and can accidentally serialize.

Prefer the first option unless H20 reveals mixed projection dtypes.

## 5. Warmup

The first NCCL call on a side stream can include connection setup and distort both correctness debugging and performance. Add a one-time overlap warmup after CP enable or on the first `attention_fused()` call:

- allocate a tiny CUDA tensor with the same dtype/device as the first K/V tensor
- on `comm_stream`, run one `dist.all_gather` for AGKV or one `group.all_to_all` for Ulysses
- synchronize only during warmup, not during measured denoise
- guard by shape-independent `_overlap_warmed[(mode, group_size, dtype, device)]`

Simpler acceptable MVP: rely on the model warmup path (`warmup_diffusion_engine(args)`) to exercise `attention_fused()` once before timing. Still record a trace to verify the first measured denoise is not paying side-stream NCCL init.

## 6. Slurm benchmark matrix

Use one exclusive allocation style and avoid mixing partitions in the same table.

Recommended first pass:

| GPUs | CFP | CP | mode | overlap | steps | purpose |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 1 | 1 | 1 | n/a | off | 50 | single-GPU reference |
| 4 | 1 | 4 | agkv | off/on | 50 | primary AGKV signal |
| 4 | 1 | 4 | ulysses | off/on | 50 | Ulysses signal |
| 8 | 1 | 8 | agkv | off/on | 50 | H20 full-node scaling |
| 8 | 1 | 8 | ulysses | off/on | 50 | H20 full-node Ulysses |
| 8 | 2 | 4 | agkv | off/on | 50 | real cfp2 x cp4 composition |

Optional controls:

- `CHITU_Z_IMAGE_STAGE_PROFILE=1` for stage-level denoise breakdown without CP bucket synchronization.
- `CHITU_Z_IMAGE_CP_PROFILE=1` with overlap off only, for analytical profile buckets.
- `CHITU_CP_FP8_KV=1` off in the first async runs. Test fp8 composition later.
- `infer.diffusion.compile_mode=off` and `infer.diffusion.ring_cudagraph=false` for the first implementation validation.

Example A/B command pattern:

```bash
export SRUN_PARTITION=<h20-partition>
export SRUN_EXTRA_ARGS="--exclusive"
export CHITU_Z_IMAGE_SIZE=1024,1024
export CHITU_Z_IMAGE_STEPS=50
export CHITU_Z_IMAGE_CP_MODE=agkv
unset CHITU_Z_IMAGE_CP_PROFILE
unset CHITU_CP_FP8_KV
unset CHITU_CP_KV_CACHE

export CHITU_CP_OVERLAP_QKV=0
chitu run <h20-zimage-config.yaml> --gpus-per-node 8 --cfp 1

export CHITU_CP_OVERLAP_QKV=1
chitu run <h20-zimage-config.yaml> --gpus-per-node 8 --cfp 1
```

Trace one short run:

```bash
export CHITU_Z_IMAGE_STEPS=4
export CHITU_TORCH_TRACE=$PWD/outputs/h20_async_comm_traces/agkv_overlap_rank0.json
export CHITU_CP_OVERLAP_QKV=1
chitu run <h20-zimage-config.yaml> --gpus-per-node 4 --cfp 1
unset CHITU_TORCH_TRACE
```

## 7. Measurement rules

- Do not compare CP-profile timed runs against overlap runs. CP profiling synchronizes CUDA events and defeats overlap.
- Use CP profile only to compute the possible ceiling and to understand whether H20 is compute-heavy or still communication-heavy.
- Use wall-clock and torch trace for actual overlap validation.
- Run overlap off/on back-to-back on the same Slurm node allocation when possible. If the launcher always creates a new allocation, repeat at least 3 times and report median.
- Record: partition, node name, driver, CUDA/PyTorch version, `NCCL_DEBUG=INFO` init path, `nvidia-smi topo -m`, model commit, config, env vars.

## 8. Risks and mitigations

- Stream lifetime bugs: fixed by strict `record_stream` on every cross-stream tensor and event waits before compute consumes comm results.
- NCCL launch-order mismatch: keep one collective sequence per mode, identical on all ranks. Do not branch per-rank between K/V/Q collectives. Keep output collectives serial in phase 1.
- Allocator pressure: async path temporarily keeps projected K/V/Q and gathered K/V alive together. Watch peak memory on H20; if needed, release text slices after concat or reuse existing gather buffers later.
- Projection closure dtype mismatch: assert dtypes/shapes in debug mode for first few calls.
- Trace overhead: use 4-step trace only; do not use trace timings as benchmark timings.
- Compile/CUDA graph interaction: keep the async path outside graph/compile
  experiments **for the eager MVP** so a missing overlap can't be confused with a
  capture bug. CUDA-graphing the overlap schedule is a deliberate later phase
  (section 9a), not an MVP concern.
- Slurm rank/GPU mapping: if `LOCAL_RANK` order does not match NVLink-friendly GPU order, fix the launcher or set `CUDA_VISIBLE_DEVICES` ordering before judging algorithms.

## 9. Data-driven next decisions

After AGKV and Ulysses MVP:

1. If AGKV improves and Ulysses does not, keep async default candidate limited to AGKV on NVLink.
2. If Ulysses input overlap improves but output collectives dominate, add `CHITU_CP_OVERLAP_U_OUTPUT=1` as a separate experiment that overlaps image output all-to-all with text head all-gather only after proving launch-order safety.
3. If H20 communication is already tiny and qkv dominates, look next at projection fusion / block compile / text-token splitting rather than more communication tricks.
4. If communication remains visible, test composition with `CHITU_CP_FP8_KV=1` after implementing stream-safe fp8 scale movement.
5. If cp8 AGKV memory is tight because full K/V gather is large, revisit unified on H20. NVLink may make Ulysses less punitive than PCIe, but the message-fragmentation lesson from PCIe still needs measurement.
6. Once Z-Image is stable, reuse `attention_fused()` from Qwen-Image by adding model-specific Q/K/V closures there, without changing the core schedule.

## 9a. CUDA graph as an overlap multiplier (later phase)

Once eager overlap is proven (trace shows real concurrency) but the measured
speedup is smaller than the profile ceiling, the gap is likely **CPU-side**: on
NVLink the collectives are short, so per-op launch latency, stream-wait
bookkeeping, and event record/wait overhead become a visible share of the
attention path, and they also let the per-rank kernel submission times drift
apart -- which loosens collective alignment across ranks. A CUDA graph attacks
both:

- **Cut launch jitter**: capture `produce_k -> all_gather K (comm stream) -> produce_v -> ... -> event waits -> attention`
  once and `replay()` it. Replay submits the whole multi-stream subgraph with
  almost no Python/CPU overhead, so the projection GEMM and the NCCL kernel start
  back-to-back every step instead of waiting on the launch queue.
- **Tighten cross-rank alignment**: a replayed graph issues kernels in a fixed,
  CPU-jitter-free order, so all ranks enter the collective at nearly the same
  time -- less skew, less exposed wait.

This is feasible in-repo: `_capture_ring_graph` / `_attn_ring_graphed`
([image_cp_attention.py:764-826](../../chitu_diffusion/modules/attention/image_cp_attention.py#L764))
already capture a multi-stream schedule containing NCCL P2P into a graph, so the
pattern (warmup -> capture on a side stream -> replay) is established here.

Hard constraints, which is why it is a separate phase, not the MVP:

- Static shapes and static input/output addresses. The harness must capture per
  (mode, cp_size, B, S_img, S_txt, H, D) and copy fresh Q/K/V into the captured
  input buffers before each replay. Z-Image's per-step shapes are stable within a
  run, so this holds, but it must be asserted.
- The comm stream, its `wait_stream`/event edges, and the NCCL collective must be
  established *inside* the capture region under the documented graph+NCCL rules
  (warmup collective first so connection setup is not captured).
- KV-cache (variable schedule), fp8 (extra scale collectives), and Ulysses output
  collectives stay out of the first graph capture.

Sequencing: prove eager overlap in the harness (section 1.5) -> A/B eager
overlap on H20 (section 6) -> only if CPU overhead is the residual bottleneck,
add a `CHITU_CP_OVERLAP_GRAPH=1` capture path in the harness, validate
bit-equivalence + trace there, then port. Do not graph before eager overlap is
visible: with both turned on at once, "no speedup" is unattributable between a
scheduling bug and a capture bug.

## 10. Implementation order

1. Build the standalone harness (section 1.5): synthetic Q/K/V driving the real
   CP core, with `--check` (bit-equiv vs serial), `--bench`, and trace support.
   This needs `attention_fused()` to exist as a fallback-only no-op first.
2. Add flag and fallback-only `attention_fused()`; the harness `--check` must show
   overlap-off == overlap-on (both serial) before any real overlap code.
3. Add AGKV async path and stream helpers. Validate in the harness: cp4 AGKV
   bit-equivalence + trace shows overlap. Iterate here, not on Z-Image.
4. Add Ulysses input async path. Validate head-divisibility fallback and trace in
   the harness.
5. Route Z-Image through `attention_fused()` only when the fast path is enabled
   (no-op behaviorally otherwise). Re-confirm bit-equivalence on the real model.
6. Run 50-step cp4/cp8 AGKV + Ulysses A/B on H20 (section 6). Stop and fix if the
   trace does not show overlap.
7. Run the full benchmark matrix and write results back into this directory,
   either as an appendix here or a new `h20_async_comm_results.md`.
8. Only if eager overlap is proven but CPU-launch-bound, add the CUDA graph
   capture path (section 9a), validating in the harness first.
9. Only after that, consider output overlap, fp8 composition, KV-cache
   composition, unified, and Qwen-Image reuse.
