# TracePlanner Context

## Current Mental Model

FreeCache/MeanCache-style step-level caching can be framed as a small discrete search:

- per step: `F / R0 / R1`;
- or per edge: anchor transition `t -> s` with an order/span label;
- if the policy is prompt/seed agnostic, an offline probe should make the search tractable.

The key missing piece is a good error state:

- expressive enough to predict final image degradation;
- compact enough to be composed across edges;
- stable enough across prompt/seed;
- able to explain model-specific behavior:
  - Qwen likes order1 and has phase sensitivity;
  - Wan is order-sensitive and often prefers order0;
  - Flux is comparatively insensitive.

## What Has Been Verified

### Hand Phase Is Strong On Qwen

Qwen-Image, `coffee_sign`, seed 42, 50 steps, 1328x1328, cfp=2.

Post-CFG high-quality phase:

| case | fresh/reuse | speedup | PSNR | SSIM | 1-LPIPS |
| --- | --- | ---: | ---: | ---: | ---: |
| hqphase order0 | 25/25 | 3.615x | 25.759 | 0.9550 | 0.9869 |
| hqphase order1 | 25/25 | 3.623x | 29.746 | 0.9619 | 0.9879 |
| hqphase order2 | 25/25 | 3.634x | 30.035 | 0.9678 | 0.9907 |

Fresh schedule:

```text
[0,1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,25,28,31,34,38,42,46,48,49]
```

### TracePlanner V1 Failed

V1 searches actions `F/R0/R1` in a GPU-resident beam. It uses latent relative MSE against the full-compute trajectory as the scoring proxy.

Real FreeCache replay:

| case | fresh/reuse | speedup | PSNR | SSIM | 1-LPIPS |
| --- | --- | ---: | ---: | ---: | ---: |
| TracePlanner B20 | 20/30 | 4.509x | 23.208 | 0.9039 | 0.9536 |
| TracePlanner B25 | 25/25 | 3.631x | 22.802 | 0.9381 | 0.9638 |
| TracePlanner B30 | 30/20 | 3.028x | 23.032 | 0.9178 | 0.9474 |

This is far worse than hand hqphase at the same speed.

### All-R1 Ablation Did Not Fix It

Keeping TracePlanner fresh positions and replacing all reuse orders with R1 barely changed quality.

Conclusion: the main failure is not just order assignment. It is fresh placement/phase or the scoring proxy.

### Checkpoint Propagate V0 Ran But Did Not Help

Implemented real checkpoint forward for Qwen cfp=2:

- rank0 selects candidate `x_hat[t]`;
- latent is broadcast to CFG ranks;
- each rank computes its CFG branch at the same timestep;
- guided `v(x_hat[t])` is gathered and fed back to TracePlanner.

Conservative run:

- `checkpoint_interval=8`
- `checkpoint_topk=1`
- actual DiT forwards: `50 full + 6 checkpoint`
- denoise time: `72.575s`

It ran successfully, but produced exactly the same B20/B25/B30 policies as V1. Real replay was therefore unchanged.

Interpretation:

- checkpoint propagation is technically viable;
- selecting only the current top fresh candidate every 8 steps is too narrow to affect the beam;
- checkpointing should probably be used to estimate edge costs or state transitions more systematically, not as a sparse patch on the current beam.

### Important Bug Fixed

`compute_jvp()` previously used scalar sigma broadcast in a way that could expand latent shape from `[B, S, C]` into extra singleton dimensions. This only became obvious when checkpoint latents were sent back through DiT.

Now scalar sigma broadcast is fixed in:

- `chitu_diffusion/flexcache/freecache_core.py`
- `chitu_diffusion/flexcache/strategy/traceplanner.py`

Tests now strictly check shape preservation.

## MeanCache Insight

MeanCache is not an online controller. It does:

1. full-compute probe on original trajectories;
2. construct a multigraph over timesteps;
3. edge cost is average-velocity approximation error for a candidate transition and JVP span/order;
4. solve a peak-suppressed shortest path under budget `B`;
5. hard-code the resulting fresh table and span/order table.

Release code evidence:

- `calc_dict`: fresh compute table;
- `edge_source`: per-interval order/span table;
- Qwen-Image, Qwen-Image-2512, and Z-Image all use different tables.

This suggests their strength is not a universal law, but a constrained offline edge-cost search with peak suppression and multi-prompt averaging.
