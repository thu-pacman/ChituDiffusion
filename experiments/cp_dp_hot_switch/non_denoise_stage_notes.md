# Non-Denoise Stage Scheduling Notes

These notes collect follow-up ideas around text encode, VAE decode, image save,
and CPU/GPU staging. They are intentionally **not** part of the main
`slo_elastic` denoise scheduler implementation. The current scheduler should stay
focused on denoise-time DP/SP layout, SLO, tardiness, and FlexCache decisions.

## Current Position

Keep the main implementation simple:

- `slo_elastic` remains the canonical denoise scheduler.
- Text encode / VAE decode / save should enter SLO budgeting as estimated stage
  costs, not as full peers in the DP/SP layout optimizer.
- Stage-level optimizations should be implemented as small, isolated queues or
  caches only after the denoise scheduler is stable.

## Measured 4090 Stage Costs

From `out/stage_profile_4090.json`:

| stage | B=1 | B=2 | B=4 | B=8 | takeaway |
| --- | ---: | ---: | ---: | ---: | --- |
| text encode ms/item | 41.17 | 34.17 | 32.72 | 34.09 | modest GPU batching benefit |
| VAE decode 512 ms/item | 31.83 | 36.37 | 35.53 | 34.74 | batching hurts |
| VAE decode 1024 ms/item | 151.30 | 198.54 | 192.18 | 188.69 | batching hurts |

CPU text encode benchmark on the same checkpoint:

| batch | CPU total | CPU per item | GPU per item |
| --- | ---: | ---: | ---: |
| B=1 | 1778 ms | 1778 ms | 41 ms |
| B=2 | 3809 ms | 1904 ms | 34 ms |
| B=4 | 7552 ms | 1888 ms | 33 ms |

CPU text encode is therefore not a low-latency path, but it can overlap with GPU
queueing when all GPUs are busy.

## Memory Facts

From the same profile:

- Text encoder full parameters: about 8.04 GB.
- Text encoder transformer weights: about 7.27 GB.
- VAE full parameters: about 0.17 GB.
- VAE decoder parameters: about 0.10 GB.

The memory pressure is primarily the text encoder, not the VAE weights. VAE
activations can still create short-lived peaks, especially at high resolution.

## Candidate Ideas

### Negative Prompt Cache

Many requests share the same negative prompt. Cache:

```text
negative_prompt -> text embedding
```

This is a low-risk optimization because it avoids recomputing identical encoder
work without changing denoise scheduling or image quality.

### GPU Text Encode Micro-Batching

4090 text encode has a small GPU batching benefit around B=2..4. If implemented,
use opportunistic batching only:

```text
max_text_batch_wait_ms ~= 5..20 ms
```

Do not delay a request substantially just to form a text batch; the absolute gain
is only a few milliseconds per item.

### CPU Text Encode Fallback

CPU text encode is around 1.8 s per item, so it should not be the default. It may
be useful when GPU encode cannot start soon:

```text
cpu_ready = now + cpu_text_encode_ms
gpu_ready = predicted_gpu_encoder_available_time + gpu_text_encode_ms

choose CPU only if cpu_ready < gpu_ready
```

With current measurements, CPU becomes plausible only when the predicted GPU text
encoder wait is roughly greater than:

```text
1778 ms - 41 ms ~= 1.7 s
```

Limit CPU concurrency, for example:

```text
max_cpu_text_encode_workers = 1 or 2
```

### VAE Decode

Do not batch VAE decode on 4090; measured per-image latency worsens with batch.
Prefer:

```text
GPU VAE decode + async queue + overlap with later denoise work
```

CPU VAE decode should be treated as an emergency memory fallback, not a serving
default. GPU decode is already about 32 ms at 512 and 151 ms at 1024.

### Stage-Aware SLO Budget

The denoise scheduler can remain unchanged if requests are converted into:

```text
denoise_ready_time = arrival + estimated_text_encode_wait + estimated_text_encode
denoise_deadline = user_deadline - estimated_vae_decode - estimated_save
```

This lets non-denoise stages affect SLO accounting without expanding the main
DP/SP layout search.

## Guardrails

- Do not move encoder/VAE decisions into the main DP/SP layout optimizer until a
  measured bottleneck requires it.
- Avoid CPU text encode by default; it is useful only if it overlaps a long GPU
  wait.
- Avoid VAE batching on 4090.
- Keep the text encoder pinned on the main rank/GPU if memory allows; offloading
  an 8 GB encoder per request is likely worse than keeping it resident.
- If memory does not allow text encoder + denoise coexistence, consider a stage
  worker GPU as a fallback, but expect reduced denoise scheduling flexibility.

