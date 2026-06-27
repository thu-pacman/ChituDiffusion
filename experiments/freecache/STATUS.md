# FreeCache / TracePlanner Status

Last update: 2026-06-27

This exploration is paused. The repository now keeps the production-facing Z-Image / step-level FlexCache integration, while this directory keeps only the research memory needed to restart later.

## Milestones

1. **Step-level FreeCache runtime path landed.**
   `freecache`, `steptrace`, and `traceplanner` are wired into the runtime as step-level cache strategies. Forced schedules and reuse orders can be replayed through the normal generator path.

2. **Z-Image was adapted to ChituDiffusion.**
   Basic T2I runtime, single-GPU FlashAttention path, CFG, VAE decode, ChituBench config, MeanCache/FreeCache replay, and TracePlanner probes were connected.

3. **Qwen / Flux edge-DP validated the main offline idea.**
   Integrated velocity costs plus a max-gap structural prior explain the hand-tuned high-quality schedules much better than peak-only costs. The useful abstraction is an edge-DP over fresh anchors, not a free-form beam search over latent MSE.

4. **Single prompt/seed Z-Image closed loop can converge.**
   On one Z-Image case, DP -> replay -> DP with a trust-region update found a small schedule edit (`+49, -11`) that improved PSNR / LPIPS and then reached Hamming-0 on the next DP round. This is evidence that the loop can calibrate a specific path.

5. **Unified/global policy search failed under the current objective.**
   Multi prompt/seed aggregation did not produce a better global schedule. The unified policies degraded both train and validation samples, so this was not simple overfitting.

## Core Conclusions

- A FreeCache schedule is a path through the denoising transport, and the best path can depend on prompt and seed.
- DP is the path search mechanism. Replay-vs-DP state gap is a map-correction signal; it should not become the direct search objective by itself.
- Positive gap area over-rewards late/tail steps. Promoting the final step can improve the state-gap score while barely helping final image quality, because there is little or no downstream propagation left.
- More data is not the next fix. The objective must be fixed before scaling prompt/seed coverage.
- The likely useful objective needs a notion of downstream influence on final latent/image quality, or at least a tail-downweighted gap/integrated-velocity cost.

## Preserved Core Workflow

The kept scripts under `tools/` cover the minimum reusable pieces:

1. Load fresh steptrace vector payloads.
2. Compute edge costs between candidate fresh anchors.
3. Run budgeted edge-DP to propose `forced_compute_steps`.
4. Aggregate edge costs across prompt/seed traces.
5. Compare an automatic DP schedule against a known good schedule.

Everything else, including concrete run outputs, policy JSON, prompt JSON, plots, copied configs, and generated images, has been removed.

## Restart Plan

If this thread is revived:

1. Generate a fresh small steptrace dataset with the current runtime.
2. First test objective changes, not bigger data:
   - downweight steps near the tail;
   - penalize demoting mid-stage anchors;
   - estimate each step's downstream effect on final latent/image quality;
   - keep trust-region updates small.
3. Only after the objective stops degrading train and validation together, scale to multi prompt/seed.
4. Recreate result artifacts in `ChituBench/results/` as disposable outputs, and only summarize conclusions back into this directory.

## What Not To Revive Blindly

- TracePlanner V1 latent-MSE beam search without replay validation.
- Pure positive-gap maximization.
- Unified schedule voting before the objective is corrected.
- Reusing old `.pt` vectors or old policy JSON; they were intentionally deleted.
