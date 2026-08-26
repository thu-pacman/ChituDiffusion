# MiniMax-H3 parallel inference service

ChituDiffusion supports the foundational native MiniMax-H3 FL2VA/T2VA inference
service: Qwen3-VL layer-50 prompt encoding, deterministic video/audio noise,
EPE DiT execution, independent parallel video VAE decode, audio VAE decode, and
asynchronous H.264/AAC MP4 packaging. Offline conditioning and deterministic
synthetic latent modes remain available for parity and performance tests.

The validated reference is single-node 8×RTX PRO 5000, TP4×CP2, FA4,
synchronous Fast Ulysses, and VAEP8. NCCL remains the CP transport fallback.

This is foundational service support rather than complete MiniMax-H3 feature
coverage: arbitrary Ref2VA reference inputs, multi-node execution, and rank
failure recovery are not yet supported. FL2VA can optionally condition on first
and/or last frames.

## Parallel layout

The checked-in SM120 stage configuration uses:

- DiT: TP4×CP2 across all eight ranks.
- CP transport: synchronous Fast Ulysses, with NCCL fallback.
- Attention: FA4 on SM120.
- Video VAE: independent VAEP8, orthogonal to DiT TP/CP.
- Audio VAE and MP4 assembly: VAEP leader.

EPE schedules in CP-rank space and expands every logical lane across all TP
planes, so a TP group is never split between requests. `vae_parallel_degree`
configures VAE decode independently; it defaults to the stage world size when
`parallel_vae` is enabled.

The reference 768×768, 5-second, 24 FPS, 20-step request completed in 49.51
seconds. VAEP8 reduced isolated video VAE decode from 8.14 seconds to 1.11
seconds (7.34×), with a maximum absolute difference of `2.38e-7` from the
release tiled decoder.

## Step cost model

The H3 scheduler models DiT step latency using padded sequence length and the
discrete CP width. Text/media composition remains request metadata rather than
an independent model feature. Fit data can be loaded with
`factory_args.cost_profile_path`.

```bash
python script/profile_h3_sparse_cost.py
python script/analyze_h3_cost_profile.py \
  outputs/chitu-api/h3-cost-profile-sparse/h3_sparse_profile.json
```

## Launch

```bash
export NVSHMEM_HOME=$PWD/refs/nvshmem-developer
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NCCL_IB_DISABLE=1
export CHITU_ULYSSES_TRANSPORT=fast_ulysses
export CHITU_FAST_ULYSSES_ASYNC_CE=0

/dockerdata/chitudiffusion-venv-sm120/bin/torchrun \
  --standalone --nproc-per-node=8 \
  -m chitu_diffusion.commands.serve \
  --stage-config \
  experiments/chitu_api/stage_config.h3.tp4cp2.fast-ulysses.sm120.yaml
```

## Native request

```bash
curl --fail-with-body --request POST \
  http://127.0.0.1:18088/v1/image-decode \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "prompt": "A red fox running through snow",
    "width": 768,
    "height": 768,
    "seed": 11,
    "num_steps": 20,
    "model_inputs": {
      "duration_s": 5.0,
      "fps": 24,
      "output_type": "mp4"
    }
  }'
```

Poll `/v1/image-decode/{request_id}` and download completed MP4 bytes from
`/v1/media/{request_id}`. Optional `first_frame_path` and `last_frame_path`
inside `model_inputs` select FL2VA endpoint-frame conditioning.

## Synthetic and offline requests

For a deterministic synthetic request, set `model_inputs.synthetic=true` and
provide `latent_t`, `latent_h`, `latent_w`, `audio_t`, `audio_channels`, and
`text_length`.

For offline latent inputs, set `model_inputs.condition_path` to a safetensors
file containing `prompt_embeds`, `video_latents`, and `audio_latents`. Shape
metadata is validated when present.

The H3 schedule matches the SGLang contract: video and audio use independent
shifted sigma schedules (defaults 12 and 3), the DiT receives `1 - sigma`, and
the eta-zero Euler update uses `(sigma_current - sigma_next) * velocity`.

Performance methodology and results:

- [`experiments/vae_parallel_pro5000/README.md`](../../../experiments/vae_parallel_pro5000/README.md)
- [`experiments/fast_cp_pro5000/README.md`](../../../experiments/fast_cp_pro5000/README.md)
