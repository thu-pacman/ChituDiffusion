# Hunyuan Image 3 parallel inference

ChituDiffusion runs Hunyuan Image 3 on a single-node parallel stage instead of
the released `device_map="auto"` layout. The 80B MoE decoder is sharded over
tensor, CFG, context, and expert parallelism chosen at launch — TP2×CFG2×CP2×EP2
on the reference machine — the VAE decodes spatial tiles across an independent
VAEP group, and
the released prompt tokenization, conditioning-image encode, and input assembly
stay on the official rank-synchronous path.

The validated reference is single-node 8×RTX PRO 5000, SDPA attention, eager
expert execution, and NCCL collectives. Both a distributed CLI and the static-CP
EPE HTTP service share one pipeline implementation.

This is foundational support rather than complete feature coverage. The first
release covers fixed-size Base text-to-image and the Instruct direct and
image-to-image paths. Text generation, `recaption`/`think_recaption`, automatic
size selection, elastic EPE lane resize, and multi-node execution are not
supported.

## Supported topologies

Any single-node shape that satisfies these constraints will run:

- `world = TP × CFG × CP`, every degree a positive integer, and `CFG ∈ {1, 2}`.
  CFG2 gives each guided branch its own context-parallel lane inside a TP plane;
  CFG1 keeps both conditions in one local batch.
- `EP` divides `CFG × CP`, and the checkpoint's routed-expert count divides `EP`.
  An `EP` below the plane width gives each contiguous slice of the plane its own
  copy of the routing table, trading memory for a narrower dispatch group.
- The attention and key/value head counts divide `TP × CP`, and every
  feed-forward width divides `TP`.
- `1 ≤ VAEP ≤ world`, independent of the decoder geometry.

For the EPE service the pool's `allowed_lane_widths` must list every width the
stage collects over — the CP lane, the EP slice, and the TP plane — because that
is what decides which process groups exist.

Only the shapes listed under "Measured results" have been run on real weights.
The others satisfy the constraints and should work, but no claim is made about
their latency or memory, and the shipped shape stays the recommendation.

## Parallel layout

The shipped configuration and its description below are TP2×CFG2×CP2×EP2 on
eight cards. An unset expert degree falls back to the full plane width (EP4),
which is the cheapest shape in memory rather than the fastest.

- Decoder: TP2×CFG2×CP2 over eight ranks. Each TP plane splits into a conditional
  CP2 lane and an unconditional CP2 lane. The two branches run concurrently and
  exchange their final predictions through shard-aligned rank pairs. Under CFG1
  a rank denoises both conditions and combines them without any exchange.
- Attention: variable-length AGKV. Queries stay on their contiguous CP shard;
  only K/V are padded by at most one token, gathered through the configured AGKV
  transport, and trimmed back into rank order. The output stays sequence-sharded,
  so there is no inverse all-to-all. `--attention-mode ulysses` is retained as a
  correctness and performance fallback. The validated CP2 subgroup uses the
  torch/NCCL transport; Fast AGKV still requires subgroup-capable NVSHMEM setup.
- Attention mask: read once per step rather than handed to every layer. A
  denoising step queries the generated image block, whose only restriction is
  the token that closes it, so those rows attend to a narrowed key sequence with
  no mask and take the fused kernel. Only the single causal row in front of the
  block still carries the released mask. Prefill keeps the masked path.
- Experts: EP2 splits each tensor-parallel plane in half, and both halves span
  the two CFG branches, so tokens from both branches participate in one
  variable-length all-to-all and each rank owns 32 of the 64 routed experts.
  Any EP that divides the plane width works; a narrower one replicates the
  routing table across more ranks and shortens the dispatch, which is exactly
  the trade the shipped shape takes.
- MoE dispatch: variable-length all-to-all over deduplicated rows. Routing is
  top-8 of 64, so a token's experts often share an owner; the row crosses the
  wire once per destination rank rather than once per expert, and the owner
  applies the router weights and sums its own experts before the inverse
  exchange. Splits and expert ownership are the only values the host needs, and
  they arrive in two transfers per layer, so the routing arithmetic never
  stalls the queue to read a device tensor. The released eager MoE builds a dense
  `[tokens, experts, capacity]` dispatch mask instead, which is why the official
  path cannot run conditioning images at this resolution on this machine (see
  below).
- VAE: independent VAEP8 by default, orthogonal to the decoder geometry. 384px
  tiles with at least 48px of overlap, one round-robin tile assignment, and
  blending on the group leader.
- Checkpoint: every rank materializes only its tensor-parallel shards and
  expert-parallel slices, so no process ever holds the full 158 GiB checkpoint.
  Resident memory is 44.3 GiB per card at EP2 and 26.3 GiB at EP4.

## Measured results

Single node, 8×RTX PRO 5000, 1024×1024, 50 steps, `guidance_scale=5.0`,
`flow_shift=3.0`, seed 1234, warm cache.

| Path | Denoise | VAE decode | End to end |
| --- | --- | --- | --- |
| Released `device_map="auto"` | 281.3 s (5.60 s/step) | included above | 281.3 s |
| Chitu TP2×CFG2×CP2×EP2 AGKV + VAEP8 | 18.0 s (0.36 s/step) | 0.27 s | 18.3 s |
| Chitu TP2×CFG2×CP2×EP4 AGKV + VAEP8 | 20.5 s (0.41 s/step) | 0.27 s | 20.8 s |

That is a 15.4× end-to-end speedup. The released layout is pipeline-serial, so
only one card computes at a time and one card stays idle; the Chitu stage keeps
all eight busy.

Four eight-card topologies were run on real weights. Pixel differences below
use the EP4/CP2 result with the same prompt and seed as their reference:

| Topology | Step | Resident | Difference from EP4/CP2 |
| --- | --- | --- | --- |
| TP2×CFG2×CP2×EP2 (shipped) | 0.36 s | 44.3 GiB | 2.77/255 MAE, 30.3 dB (50 steps) |
| TP2×CFG2×CP2×EP4 | 0.41 s | 26.3 GiB | reference |
| TP2×CFG1×CP4×EP4 | 0.47 s | 26.3 GiB | 1.43/255 MAE, 41.9 dB (2 steps) |
| TP4×CFG2×CP1×EP2 | 0.58 s | 25.1 GiB | 1.76/255 MAE, 39.5 dB (2 steps) |

Each produces the same picture; the residual is reduction order, which changes
with the number of context-parallel shards and with expert ownership. The last
two are slower, so the shipped shape keeps CFG2 and CP2: CFG1 pays for a
four-way K/V gather where CFG2 pays for one paired prediction exchange, and TP4
adds an all-reduce to every projection while CP1 gives up sequence parallelism
altogether.

EP is the one axis where the shipped shape spends memory to buy speed. A token
is routed to eight of 64 experts, so it reaches nearly every group that owns
one of them: at EP4 its row lands on 3.7 of the four destinations on average,
against 2.0 of two at EP2. Halving the group therefore halves the rows on the
wire and narrows the collective, which the dispatch probe sees as 1.22 ms of
exchange per layer against 3.04 ms, and the whole dispatch as 3.18 ms against
5.24 ms. Peak memory is 45.7 GiB of the 72 GiB card. EP4 remains the right
choice on smaller cards and is what an unset degree resolves to.

Three measurements set that step time, all taken per layer on the real process
groups by `experiments/hunyuan_image3_pro5000/cp_transport_probe.py` and
`ep_dispatch_probe.py`. Expert dispatch dominates at 8.74 ms per layer before
row deduplication, of which 6.36 ms was hidden rows on the wire; deduplication
and a single fused gather-scatter on the receive side bring it to 5.49 ms, and
folding the routing arithmetic into two host transfers brings EP4 to 5.24 ms.
Attention was second: a masked SDPA call costs 1.27 ms per layer against 0.42 ms
for the same shapes unmasked, which is what the query split recovers. Together
they moved the non-CFG-parallel step from 0.59 s to 0.46 s. CFG2 moves the EP4
layout to 0.44 s/step, and replacing its four Ulysses exchanges with two K/V
gathers reaches 0.43 s/step. On the production CFG2/CP2 subgroup the isolated
transport drops from 0.50 to 0.21 ms per layer (15.9 to 6.8 ms per step), while
the end-to-end step drops by about 8 ms. The AGKV and Ulysses T2I outputs are
pixel-identical at both 2 and 50 steps.

What is left in the synchronous path is the wire itself. The eight cards have
no NVLink, so a dispatch group is a PCIe all-to-all inside one NUMA node at
roughly 30 GB/s per direction; at EP2 that is still 39 ms of the 358 ms step.

VAEP8 decode takes 0.27 s against 0.92 s for the same tile plan on one card
(3.4×). The tiled result differs from a single full-canvas decode by 0.94/255
mean absolute error (45.7 dB PSNR), which is seam blending only.

Memory at EP2 is 44.3 GiB resident on every card, peaking at 45.7 GiB. EP4 is
26.3 GiB resident, peaking at 34.5 GiB on rank 0 (which owns postprocessing) and
27.6 GiB elsewhere. The released layout needs up to 41.9 GiB on its busiest card.

### Parity against the released path

Fixed seed, same prompt, same schedule:

| Steps | Mean absolute error | PSNR |
| --- | --- | --- |
| 2 | 2.22/255 | 38.1 dB |
| 50, EP2 | 4.97/255 | 29.1 dB |
| 50, EP4 | 6.22/255 | 27.1 dB |

Per-step agreement is at bfloat16 level. Over 50 steps the trajectories drift
apart because tensor-parallel and context-parallel reductions run in a different
order, so the images stay compositionally identical while fine texture differs.
Bitwise reproduction of the released path is not a goal and is not achievable
without matching its reduction order.

### Image-to-image

The image-to-image path runs end to end and honors the conditioning image, and
the official comparison could not be produced on this machine: the released
eager MoE tried to allocate a 61 GiB dense dispatch mask for the longer
conditioned sequence. Output quality is also not validated, because the local
checkpoint is the Base model while the released image-to-image flow expects the
Instruct checkpoint with its `en_unified` system prompt.

## Launch

Any degree left unset is derived from the launched world size, which on eight
GPUs gives TP2×CFG2×CP2×EP4. Ask for EP2 to trade 18 GiB per card for the
shorter dispatch measured above.

```bash
/dockerdata/chitudiffusion-venv-sm120/bin/torchrun \
  --standalone --nproc-per-node=8 \
  -m chitu_diffusion.cli generate --model hunyuan-image3 \
  --model-path /dockerdata/HunyuanImage-3 \
  --prompt "a red cube on a white table, studio lighting" \
  --steps 50 --seed 1234 --expert-parallel-degree 2 \
  --output outputs/chitu/hunyuan_image3.png
```

Pass `--tensor-parallel-degree`, `--cfg-parallel-degree`,
`--context-parallel-degree`, or `--expert-parallel-degree` to select another
shape; the launch fails with the offending constraint rather than silently
falling back.

```bash
/dockerdata/chitudiffusion-venv-sm120/bin/torchrun \
  --standalone --nproc-per-node=8 \
  -m chitu_diffusion.commands.serve \
  --stage-config examples/stage-hunyuan-image3.yaml
```

The service takes the same geometry from `parallelism.model`
(`tensor_parallel_degree`, `cfg_parallel_degree`, `expert_parallel_degree`) with
`parallelism.cp` holding the rank space each TP plane schedules and
`parallelism.vae.degree: 8` decoding across the whole stage.

## Native request

```bash
curl --fail-with-body --request POST \
  http://127.0.0.1:18084/v1/image-decode \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "prompt": "a red cube on a white table, studio lighting",
    "width": 1024,
    "height": 1024,
    "seed": 1234,
    "num_steps": 20
  }'
```

Poll `/v1/image-decode/{request_id}` and download the PNG from
`/v1/image-decode/{request_id}/image`. A 20-step request completes in 10.1
seconds including admission and postprocessing.

Requested sizes are snapped onto the released resolution group, so 512×512
resolves to the warmed 1024×1024 shape. A request whose snapped shape is not in
`warmup_resolutions` is rejected at admission, for example a 2:1 request that
resolves to 704×1344.

## Limits

Fixed size, batch 1, single node, SDPA attention, eager local experts. The
topology is configurable but only the three shapes above have been measured, and
only on eight cards. Prompt encode, conditioning-image encode, and text
generation are unoptimized and rank-synchronous. The key/value cache is lane
resident and expert ownership is fixed, so the stage requires `static_cp` and a
request cannot migrate between lanes: `export_state` raises. Automatic size
selection, FlashInfer expert execution, multi-node execution, and elastic EPE
lanes are not supported.
