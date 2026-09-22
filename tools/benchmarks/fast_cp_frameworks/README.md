# Fast CP framework benchmark

This directory is the clean benchmark boundary for comparing Chitu Fast CP,
xDiT, and SGLang Diffusion. It replaces the archived exploratory harnesses under
`outputs/fast-cp-archive-20260917`.

## Benchmark contract

The primary comparison is implementation-faithful. Chitu, xDiT, and SGLang
Diffusion must execute their own context-parallel attention entry point, native
communication schedule, attention backend, stream/event dependencies, layout
transforms, and padding policy. The harness standardizes only the semantic
input, model shape, BF16 dtype, H20 node, warmup, timing, and correctness check.
It must not replace a repository's native path with a shared collective or a
shared attention call merely to make the code look uniform.

xDiT and SGLang use their automatic native attention-backend selectors in the
primary sweep. Set `FAST_CP_XDIT_ATTENTION_BACKEND=cudnn` or
`FAST_CP_SGLANG_ATTENTION_BACKEND=torch_cudnn_sdpa` only for a separately
labeled controlled-backend experiment. Chitu's initial Fast CP implementation
uses PyTorch cuDNN SDPA on this QKV-to-O boundary.

Every timed sample uses CUDA events and the maximum latency across ranks. The
report records the exact repository revision, Python/Torch/CUDA/cuDNN versions,
GPU name, node, attention backend, transport, graph mode, and whether the
native path supports the requested CP degree. An unsupported native
configuration is reported as unsupported; the benchmark must not add padding
or another algorithm unless that repository does so itself.

Speedup always uses CP1 from the same repository, software environment, model
shape, and benchmark boundary. Cross-repository absolute latency is valid only
for runs collected on the same H20 node under the same allocation. Native
dependency differences remain part of the primary result and are disclosed in
the output. A controlled common-backend experiment, if needed, is a separate
secondary table.

## Required breakdown

Each repository and shape produces four measurements from the same native
implementation:

1. **Communication:** replay the native input and output exchanges with the
   exact tensors, layouts, packing, padding, and synchronization used by the
   full path. Input exchange and output exchange are also reported separately.
2. **Attention compute:** run the repository's native attention backend on the
   exact post-communication local shape. Chitu planner on/off is an additional
   ablation, not silently included in the communication result.
3. **Overlap:** run the full native path, then run the same path with an explicit
   synchronization at each communication/compute boundary. Report
   `overlap_gain = serial_ms - native_ms` and
   `overlap_efficiency = overlap_gain / min(comm_ms, compute_ms)`. The forced
   serial run changes dependencies only; it does not substitute kernels.
4. **End to end:** time the unmodified native CP attention invocation, including
   projections, Q/K normalization, RoPE, layout transforms, communication,
   attention, merge when present, and output projection.

The main attribution table contains native end-to-end time, communication time,
attention-compute time, overlap gain, and the residual work outside those two
components. Nsight Systems traces validate stream overlap and kernel ordering;
the reported latency still comes from rank-max CUDA-event samples. Standalone
communication plus standalone compute is not treated as an end-to-end model,
because allocator state, launch order, and resource contention can change the
sum.

## Model-shaped workload layer

The end-to-end layer instantiates each repository's real attention module and
configuration with deterministic synthetic tensors and randomly initialized
parameters. It does not download checkpoint weights, run text encoders, VAE, or
the denoising loop. This preserves the production projections, normalization,
RoPE, joint-attention layout, CP dispatch, communication, and output projection
while keeping the test reproducible and small enough for rapid iteration.

| Model profile | Attention shape | Workload semantics | CP degrees |
| --- | --- | --- | --- |
| Wan 2.1 1.3B | H12, D128, hidden 1536 | image-token self-attention | 2, 4, 8 |
| Wan 2.1 14B | H40, D128, hidden 5120 | image-token self-attention | 2, 4, 8 |
| Z-Image | H30, D128, hidden 3840 | image and caption joint attention | 2, 4, 8 |
| Qwen-Image | H24, D128, hidden 3072 | image and text joint attention | 2, 4, 8 |

The core image-token sweep is global `S = 4096, 8192, 16384`. Each model also
has one required production profile derived from its native default image or
video resolution, frame count, VAE compression, and patchification rules; the
runner records the resulting token count. Z-Image and Qwen-Image include their
real caption/text segment length and replication or sharding policy instead of
reducing the workload to image-only attention. Batch size, mask validity,
rotary tensors, and padding are generated from the model config and recorded in
every result.

Some head counts do not divide every CP degree. For example, Wan 1.3B H12 does
not divide CP8, and Z-Image H30 does not divide CP4 or CP8. AGKV can still be a
native valid path, while an Ulysses implementation may reject the shape or use
its own head padding. These cells remain in the matrix and expose that behavior
rather than being removed or repaired by the harness.

For each model, CP2/CP4/CP8 results are reported in two views:

- a repository-native comparison containing every supported native method
  (Chitu Fast AGKV/Fast Ulysses, xDiT Ulysses, and SGLang Diffusion K/V gather
  and Ulysses);
- a best-native-method table that selects the lowest median latency within each
  repository and shows latency, speedup versus that repository's CP1, and peak
  memory.

Correctness is checked before timing against a dense CP1 reference generated
from the same synthetic inputs and parameters. Attention-only output is checked
in FP32 error metrics; the model-shaped path checks the final projected output.
No performance cell is published if correctness fails.

## Execution order

1. Finish the current pre-projected Q/K/V smoke layer to validate environments,
   distributed initialization, rank ordering, and report schema.
2. Add native instrumentation and forced-serial controls separately in each
   repository adapter; validate kernel order with one Nsight Systems trace per
   method.
3. Run the communication, attention, serial, and native-overlap breakdown for
   `S = 4096, 8192, 16384` at CP2/CP4/CP8.
4. Add the four weight-free model attention fixtures and run their native
   end-to-end matrix on the same exclusive 8xH20 allocation.
5. Restore the Chitu CP attention planner only as an explicit compute ablation
   after the initial Fast CP baseline has been recorded.

## Pinned sources and environments

`versions.env` records immutable upstream commits. Build two isolated
environments from scratch:

```bash
tools/benchmarks/fast_cp_frameworks/setup_envs.sh
```

The default locations are:

- sources: `refs/fast-cp-frameworks/{xdit,sglang}`
- environments: `.venv-fast-cp-frameworks/{xdit,sglang}`

Both can be overridden with `FAST_CP_BENCH_SOURCE_ROOT` and
`FAST_CP_BENCH_ENV_ROOT`. Chitu runs with `.venv` because its Fast CP extension
and NVSHMEM runtime are installed there.

## H20 execution

The launcher requests an exclusive node using the cluster default partition.
Set `SRUN_PARTITION` for srun launchers or `SBATCH_PARTITION` for sbatch launchers
to select a partition. Examples:

```bash
# Wan 14B attention shape: H40/D128, supports CP2/4/8 Ulysses.
tools/benchmarks/fast_cp_frameworks/run_case.sh chitu fast_agkv 8 4096 40
tools/benchmarks/fast_cp_frameworks/run_case.sh chitu fast_ulysses 8 4096 40
tools/benchmarks/fast_cp_frameworks/run_case.sh xdit ulysses 8 4096 40
tools/benchmarks/fast_cp_frameworks/run_case.sh sglang ulysses 8 4096 40
tools/benchmarks/fast_cp_frameworks/run_case.sh sglang kv_gather 8 4096 40

# Wan 1.3B attention shape: H12/D128, Ulysses supports CP2/4.
tools/benchmarks/fast_cp_frameworks/run_case.sh chitu fast_agkv 2 4096 12
tools/benchmarks/fast_cp_frameworks/run_case.sh xdit ulysses 2 4096 12
tools/benchmarks/fast_cp_frameworks/run_case.sh sglang kv_gather 2 4096 12
```

Use CP1 with the same framework, shape, and environment for the scaling
denominator. Fast transports are unnecessary at CP1; use `nccl_agkv` for Chitu.

Each JSON under `outputs/fast-cp/qkvo` records raw rank-max samples, repeat
medians, accuracy, actual CP degree, tensor shape, GPU, Torch/CUDA/cuDNN versions,
upstream revision, transport, attention backend, Slurm job, nodes, and benchmark
boundary. Use the same node allocation for CP1/CPn results used in a final
scaling claim; the node field makes accidental cross-node comparisons visible.

Generate a Markdown table after the CP1 and multi-GPU cases complete:

```bash
.venv/bin/python tools/benchmarks/fast_cp_frameworks/summarize.py \
    outputs/fast-cp/qkvo --output outputs/fast-cp/qkvo/results.md
```

The summarizer rejects mixed benchmark boundaries, attention shapes, or GPU
models. It computes speedup only from a CP1 result produced by the same
framework and software revision. Chitu uses `nccl_agkv` as its default CP1
baseline because Fast CP communication is unnecessary on one GPU. SGLang uses
its Ulysses CP1 result for both Ulysses and K/V-gather scaling because both
communication paths reduce to the same local attention at CP1.

Submit the complete native end-to-end sweep for `S = 4096, 8192, 16384, 32768`
and CP2/CP4/CP8. The job first records a framework-local CP1 denominator for
every sequence length, then runs all seven native methods in one exclusive H20
allocation. Completed JSON cells are skipped when a preempted job is resumed.

```bash
tools/benchmarks/fast_cp_frameworks/submit_e2e_sweep.sh \
    outputs/fast-cp/e2e-native-sweep
```

The job writes its log to the output directory and generates `results.md` after
the final cell completes. Override the matrix with
`FAST_CP_BENCH_SEQUENCES`, `FAST_CP_BENCH_CP_DEGREES`, or
`FAST_CP_BENCH_HEADS` when running a focused subset.

Run all native QKV-to-O components sequentially in one exclusive allocation,
then generate the component table:

```bash
FAST_CP_BENCH_NODE=bjdb-h20-node-021 \
    tools/benchmarks/fast_cp_frameworks/submit_breakdown_matrix.sh 4096 40 128 8
.venv/bin/python tools/benchmarks/fast_cp_frameworks/summarize_breakdown.py \
    outputs/fast-cp/breakdown-same-node \
    --output outputs/fast-cp/breakdown-same-node/results.md
```

The reported delta is `E2E - communication - attention` across independent
runs. It is useful for finding missing work or timing-boundary problems, but is
not interpreted as overlap. Native versus forced-serial model runs and Nsight
Systems timelines provide the overlap measurement.

Capture a short post-warmup timeline with CUDA Profiler API bracketing:

```bash
FAST_CP_BENCH_NODE=bjdb-h20-node-021 \
    tools/benchmarks/fast_cp_frameworks/submit_trace.sh \
    chitu fast_ulysses 8 4096 40 128 e2e
```

Run native and forced-serial Wan attention in one allocation:

```bash
FAST_CP_BENCH_NODE=bjdb-h20-node-021 \
    tools/benchmarks/fast_cp_frameworks/submit_wan_matrix.sh wan14b 4096 8
.venv/bin/python tools/benchmarks/fast_cp_frameworks/summarize_wan.py \
    outputs/fast-cp/model-attention-same-node \
    --output outputs/fast-cp/model-attention-same-node/results.md
```

Collect a CP1 reference and the CP2/4/8 native, serialized-CE, synchronous,
and NCCL matrix under one allocation:

```bash
FAST_CP_BENCH_NODE=bjdb-h20-node-038 \
    tools/benchmarks/fast_cp_frameworks/submit_wan_scaling.sh \
    wan14b 4096 outputs/fast-cp/wan14b-scaling-node038
```

Run the native Qwen-Image or Z-Image joint-attention boundary, including the
replicated text stream:

```bash
FAST_CP_BENCH_NODE=bjdb-h20-node-021 \
    tools/benchmarks/fast_cp_frameworks/submit_joint_scaling.sh \
    qwenimage 4096 512 outputs/fast-cp/qwenimage-scaling-node021
.venv/bin/python tools/benchmarks/fast_cp_frameworks/summarize_joint.py \
    outputs/fast-cp/qwenimage-scaling-node021 \
    --output outputs/fast-cp/qwenimage-scaling-node021/summary.md
```

Run xDiT's native Wan attention fixture with its isolated environment:

```bash
srun --exclusive --nodes 1 --ntasks 1 --gres gpu:8 \
    tools/benchmarks/fast_cp_frameworks/run_local_xdit_wan_case.sh \
    wan14b 8 4096 outputs/fast-cp/xdit-wan
```

The launcher writes an `unsupported` report instead of changing the algorithm
when the native xDiT Ulysses divisibility requirement is not met.

Run xDiT's native attention processors for all four model fixtures at
CP1/CP2/CP4/CP8. Qwen-Image shards its image and text streams independently;
Z-Image shards the unified image/text sequence, matching each xDiT wrapper.

```bash
tools/benchmarks/fast_cp_frameworks/submit_xdit_model_sweep.sh \
    outputs/fast-cp/xdit-model-native-sweep
```

Run SGLang's four model-shaped attention fixtures. Qwen-Image and Z-Image call
their native attention modules; Wan uses the same projections, cross-head
RMSNorm, RoPE, `USPAttention`, and output projection as the self-attention
subpath embedded in `WanTransformerBlock`. The script first checks CP2 smoke
cases, then records the formal CP1/CP2/CP4/CP8 matrix for native Ulysses and
K/V gather.

```bash
tools/benchmarks/fast_cp_frameworks/submit_sglang_joint_sweep.sh \
    outputs/fast-cp/sglang-joint-native-sweep
```

Combine the framework-local CP1 scaling results after both model jobs finish:

```bash
.venv/bin/python tools/benchmarks/fast_cp_frameworks/summarize_model_native.py \
    outputs/fast-cp/wan1.3b-scaling-node021 \
    outputs/fast-cp/wan14b-scaling-node040 \
    outputs/fast-cp/qwenimage-scaling-node021 \
    outputs/fast-cp/zimage-scaling-node021 \
    outputs/fast-cp/xdit-model-native-sweep \
    outputs/fast-cp/sglang-joint-native-sweep \
    --output outputs/fast-cp/model-native-summary.md
```

Generate raw-latency and normalized workload radar charts for each CP degree plus
a combined overview. Model labels include the global attention QKV shape in
`B x S x H x D` order:

```bash
.venv/bin/python tools/plotting/fast_cp_workload_radar.py \
    --native-sweep-dir outputs/fast-cp/cp-attention-native-20260918
```

The four sequence vertices use the native QKV-to-O sweep; the four model vertices
use their weight-free native attention fixtures. The normalized radius is the
fastest latency for the workload divided by the method latency, so 100% is the
fastest available implementation. The script also writes raw latency and
normalized scores as CSV.
The sequence vertices compare the six requested methods, including xDiT USP.
xDiT USP model vertices remain gaps because there is no model-level native fixture.
