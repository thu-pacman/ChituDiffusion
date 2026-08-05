# FLUX.2-klein Diffusers Integration

Date: 2026-07-27

This package keeps the official Diffusers pipeline as the single-GPU baseline
and adds a fixed, full-world AGKV/USP context-parallel transformer. It intentionally
does not expose EPAC or elastic lane switching: the four-step distilled schedule
leaves too little scheduling interval to justify the state-lifecycle changes
that an elastic adapter would require.

## Supported Scope

- distilled FLUX.2-klein checkpoints accepted by `Flux2KleinPipeline`;
- native single-process Diffusers generation;
- fixed full-world context parallelism over image tokens;
- replicated text tokens and request output on every rank;
- batch size one, no attention mask, and no fused double-stream QKV.

The context-parallel implementation shards image tokens before the transformer,
runs the selected AGKV or USP backend inside the fixed lane, and gathers projected
image tokens before the Diffusers pipeline resumes. AGKV remains the default.
Dynamic topology, CFG parallelism, parallel VAE,
EPAC serving, and non-distilled checkpoints are outside this integration.

## Validation Snapshot

On one H20 node with the local distilled checkpoint, BF16, a 512x512 image,
four denoise steps, and seed 7:

- native Diffusers and four-GPU fixed CP produced byte-identical PNG files;
- native generation took 1.844 s and four-GPU fixed CP took 1.344 s;
- the single-run timing is an integration smoke result, not a throughput claim.

Run the two retained entry points with:

```bash
python chitu_diffusion/examples/flux2_klein_native.py \
  --model-path /path/to/flux2-klein \
  --output outputs/flux2_klein/native_512_4step.png

torchrun --standalone --nproc-per-node 4 \
  chitu_diffusion/examples/flux2_klein_cp.py \
  --model-path /path/to/flux2-klein \
  --output outputs/flux2_klein/fixed_cp_4gpu_512_4step.png
```

Images and JSON timing records are generated under `outputs/` and remain
untracked.
