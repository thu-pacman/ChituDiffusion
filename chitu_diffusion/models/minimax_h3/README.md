# MiniMax-H3 native FL2VA service

The service supports the complete native FL2VA path: Qwen3-VL layer-50 prompt
encoding, deterministic video/audio noise, EPE DiT execution, lane-owner video
and audio VAEs, and asynchronous H.264/AAC MP4 packaging. The previous offline
conditioning and deterministic synthetic latent modes remain available for
parity and performance tests.

## Parallel layout

The checked-in stage configuration uses eight ranks as TP4 × elastic CP{1,2}:

- CP2: one request uses all eight ranks.
- CP1 + CP1: two requests each use one complete four-rank TP group.

EPE schedules in CP-rank space and expands every logical lane across all TP
planes, so a TP group is never split between requests.

## Step cost model

The H3 scheduler models DiT step latency using only padded sequence length and
the discrete CP width. Text/media composition is retained as request metadata
but is not an independent model feature. Fit data can be loaded with
`factory_args.cost_profile_path`.

Generate five fit points and three held-out validation points per CP width:

```bash
python script/profile_h3_sparse_cost.py
python script/analyze_h3_cost_profile.py \
  outputs/chitu-api/h3-cost-profile-sparse/h3_sparse_profile.json
```

## Launch

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /dockerdata/chitudiffusion-venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=8 \
  --module experiments.chitu_api.serve \
  --stage-config experiments/chitu_api/stage_config.h3.yaml
```

## Native request

```json
{
  "prompt": "A red fox running through snow",
  "seed": 11,
  "num_steps": 50,
  "model_inputs": {
    "width": 1344,
    "height": 768,
    "duration_s": 5.0,
    "fps": 24,
    "output_type": "mp4"
  }
}
```

Optional `first_frame_path` and `last_frame_path` select FL2VA endpoint-frame
conditioning. Completed MP4 bytes are available from `/v1/media/{request_id}`;
the legacy `/v1/image-decode/{request_id}/image` route remains an alias.

## Synthetic latent request

```json
{
  "prompt": "offline",
  "seed": 11,
  "num_steps": 50,
  "model_inputs": {
    "synthetic": true,
    "latent_t": 4,
    "latent_h": 128,
    "latent_w": 128,
    "audio_t": 8,
    "audio_channels": 2,
    "text_length": 64
  }
}
```

For offline latent inputs, set `model_inputs.condition_path` to a safetensors file
with `prompt_embeds`, `video_latents`, and `audio_latents`. Shape metadata is
validated when present.

The H3 schedule matches the SGLang contract: video and audio use independent
shifted sigma schedules (defaults 12 and 3), the DiT receives `1 - sigma`, and
the eta-zero Euler update uses `(sigma_current - sigma_next) * velocity`.
