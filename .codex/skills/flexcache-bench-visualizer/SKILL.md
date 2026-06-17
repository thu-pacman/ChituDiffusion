---
name: flexcache-bench-visualizer
description: Use when Codex benchmarks or visualizes FlexCache strategies, updates ChituBench result.md, creates speed-quality trade-off plots, Pareto/frontier summaries, or visual contact sheets for cache acceleration methods such as TeaCache, MeanCache, PAB, BlockDance, Cubic, TaylorSeer, or similar.
---

# FlexCache Bench Visualizer

Use this skill to plan, run, summarize, and visualize FlexCache benchmark sweeps. The goal is to produce a reusable speed-quality comparison, not a pile of one-off points.

## Benchmark Discipline

1. Reuse completed runs before launching new jobs. Prefer symlinking existing run dirs into a consolidated experiment dir and rerunning collection/evaluation over reloading the model.
2. Minimize queue and load cost. When new points are needed, batch multiple requests/strategy settings into one model load whenever the benchmark script supports it or can be cheaply extended to support it.
3. Do not present a strategy as characterized by one point unless the user explicitly asks for a smoke test. For a FlexCache trade-off summary, aim for about 3 points per method at different speed/quality settings.
4. Keep baseline explicit. Use the same prompt, seed set, resolution, step count, attention backend, and parallel strategy across comparable points. If any of these differ, state it in the notes and avoid over-comparing.
5. Record what was reused, what was newly run, and what was skipped. If a metric such as HPSv3 is skipped, say so.

## Sweep Coverage

For each method, choose settings that span low, medium, and high acceleration. Exact knobs depend on the implementation, but the result should make each method's curve visible.

- TeaCache or MeanCache: vary fresh-step count, cache ratio, threshold, or equivalent reuse aggressiveness.
- PAB: vary skip ranges, warmup/cooldown, or block selection so at least three quality/speed points exist.
- BlockDance: vary boundary/group/fraction parameters so it has a real curve, not a single decorative marker.
- Cubic: vary target speedup, tau, block size/split settings, or warmup/cooldown to produce conservative, medium, and aggressive points.
- TaylorSeer: vary order, step interval, or warmup/cooldown if available.

If a method cannot reasonably produce three valid points, label it as under-swept or experimental instead of implying a complete frontier.

## Trade-Off Plot

Make one primary data visualization in the same spirit as the existing ChituBench FlexCache plots:

- Use a two-panel speed-quality plot, typically PSNR vs speedup and 1-LPIPS vs speedup.
- Encode each method with a stable label, color, and marker across all plots and visual sheets.
- Connect points from the same method so the method curve and the Pareto/frontier behavior are visible in this plot.
- Do not create a separate Pareto frontier plot unless the user explicitly asks. The frontier should be readable from the quality-latency trade-off chart itself.
- Do not draw text labels next to individual points by default. Use the legend, color, marker, and connected method curve to identify methods; if exact settings are needed, put them in the table or caption.
- Keep the origin/baseline visible with a neutral color and a vertical speedup reference line where appropriate.

Recommended stable families:

| family | label | color |
| --- | --- | --- |
| origin / baseline | Origin or Torch SDPA | neutral gray |
| TeaCache | TeaCache | purple |
| MeanCache | MeanCache | purple |
| PAB | PAB | red |
| BlockDance | BlockDance | blue |
| Cubic | Cubic | green |
| TaylorSeer | TaylorSeer | orange |

## Visual Contact Sheet

The visual sheet is for checking generated images, not for explaining the whole experiment.

- Use the representative prompt/image set requested by the user. If the user says one coffee prompt is enough, use only that prompt.
- Show the full prompt exactly once, not once per wrapped row. Put it in a title/subtitle area or a dedicated text band above the grid.
- Do not truncate the prompt. Wrap it to multiple lines if needed.
- Do not show every generated image in the visual sheet. Show at most two representative images per method unless the user explicitly requests a full grid.
- For methods that are not on, or close to, the speed-quality frontier, show only one representative image.
- Prefer one balanced/frontier point plus one aggressive point when a method deserves two images.
- Prefer filling each visual sheet row. If adding one extra representative image from a useful frontier method avoids a mostly empty row, include it.
- Keep the total image count at `4n` in normal visual checks so the sheet forms complete rows without redundant whitespace.
- Draw the speedup and one typical quality metric directly on each image, usually `speedup_vs_origin` and `1-LPIPS`.
- Use 4 images per row only when a full grid is explicitly requested.
- Group cases by method with colored outlines or headers when showing more than one method. Use the same family colors and labels as the trade-off plot.
- Use short case labels matching the data plot labels.
- If a method has multiple sweep points, keep them adjacent inside the method group.
- Include the baseline in the first position unless the user asks otherwise.

## Result.md Requirements

When updating `ChituBench/result.md`:

- Add the consolidated run ID and exact command snippets used to collect/evaluate/visualize.
- State reused runs and newly launched runs separately.
- Include a summary table with speed, speedup, and quality metrics.
- Include only the primary speed-quality plot unless another plot is explicitly requested.
- Include the contact sheet after the data plot.
- Note under-swept methods and avoid strong frontier claims until each method has enough points.

## Acceptance Checklist

Before finalizing:

- The consolidated result dir contains summary/raw rows and quality rows.
- The primary speed-quality plot exists and has stable method colors, connected method curves, and readable short labels.
- No separate Pareto image is left behind unless explicitly requested.
- The contact sheet uses a `4n` image count, avoids redundant empty space by filling rows where reasonable, shows one image for non-frontier methods, full prompt once, speedup plus one quality metric on each image, and matching colors/labels.
- The final response states whether any strategies still need more sweep points.
