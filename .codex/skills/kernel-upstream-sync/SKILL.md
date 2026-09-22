---
name: kernel-upstream-sync
description: Tracks kernel changes from SGLang, vLLM, HPC-Ops, FlashAttention, FlashInfer, SageAttention, and diffusion engines; evaluates diffusion relevance; integrates reusable kernels through kernel_adapters; benchmarks correctness and performance; and updates the kernel worklog. Use for kernel refreshes, attention upgrades, sparse or quantized operator evaluation, and performance-baseline maintenance.
---

# ChituDiffusion Kernel Upstream Sync

## Rules

- Treat upstream speed claims as hypotheses until measured on ChituDiffusion's DiT shapes and target GPUs.
- Do not port decode-only KV-cache, paged-attention, MoE, or sampling kernels without a demonstrated diffusion hot path.
- Prefer a maintained package/API adapter in `kernel_adapters/`. Vendor source only when the API cannot satisfy the use case; retain license and provenance.
- Never make a lossy or architecture-specific kernel the unconditional default. Gate by capability, dtype, shape, and correctness.
- Reconcile `.gitmodules` with `pyproject.toml` git sources before changing Sage/Sparge; reference submodules and installed packages currently use different upstream URLs.
- Append to `kernels/worklog.md` on every investigation, including rejected candidates.

## Workflow

1. Read `kernels/worklog.md`, `kernels/upstreams.toml`, and the current attention/operator integration.
2. Run `python script/kernel_watch.py scan`. An exit code of 1 means reviewed heads changed, not a tool failure.
3. Materialize only relevant sources:
   `python script/kernel_watch.py sync <source...>`.
4. Compare the locked commit with the checkout. Inspect focused paths from the manifest, tests, benchmarks, supported GPU architectures, dtypes, shapes, and license.
5. Record each candidate as:
   - applicable now,
   - needs an adapter or shape change,
   - benchmark-only,
   - rejected with a technical reason.
6. Add external API normalization to `kernel_adapters/`; keep model semantics in `chitu_diffusion/`.
7. Run the smallest correctness case first:
   `python kernels/benchmark_attention.py --preset smoke --backends torch_sdpa <candidate>`.
8. Benchmark representative image/video shapes on the same GPU and software stack. Require repeated latency measurements and report unavailable backends instead of silently falling back.
9. Run a real ChituBench model A/B. Kernel-only wins are insufficient when preprocessing, layout conversion, compile, or communication erases them.
10. Update `kernels/worklog.md` with commands, hardware, commits, correctness, kernel latency, end-to-end latency, decision, and follow-up.
11. Only after review is complete, run:
    `python script/kernel_watch.py lock <source...>`.

## Acceptance gates

- Output shape, dtype, and model semantics match the reference.
- Exact kernels pass the configured numerical threshold; lossy kernels include quality evaluation.
- The candidate beats the current baseline on at least one production-relevant shape without unacceptable regressions on supported shapes.
- Capability checks fail clearly on unsupported GPUs.
- Context parallelism is tested when the backend must return logsumexp.
- Provenance, license, upstream commit, build requirements, and rollback path are documented.
