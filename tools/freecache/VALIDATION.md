# Publication validation — 2026-09-18

This is toolchain acceptance, not a new calibration or quality result for the
released profiles. Runtime schedules, coefficients and benchmark plots were
not changed.

## CPU and documentation

- `python -m tools.freecache.preprocess reproduce --check-runtime --output <new-file>`:
  all 150 profiles match, including every budget-extension ranking and the nine
  historical overrides. Canonical input hashes pass. Also executed from an
  index-only checkout without local research files or ignored outputs.
- `python -m pytest tests/unit/flexcache -q`: 218 passed, including 13 focused
  preprocessing tests. CPU environment: declared Torch 2.10 /
  Diffusers 0.38 dependencies.
- `ruff check`, `git diff --check`, Python compilation, and
  `mkdocs build --strict` pass.

Tests cover direct-vector versus Gram objective equality, protected warmup
cardinality, zero terminal sigma span, legacy deviation-versus-gain semantics,
partial/mismatched input rejection, no overwrite, hook restoration after an
exception, native-resolution PSNR, and runtime dictionary cache statistics.

## GPU smoke

Single NVIDIA H20, Slurm `debug`, Z-Image at 512², 50-step Euler, guidance 5,
BF16. Torch 2.9.1+cu130 / Diffusers 0.37.0 / Transformers 4.56.2. This smoke does
not validate new GPU collection for FLUX/Qwen or their quality under new fits;
their shipped profiles are covered by exact CPU reproduction.

Inputs (JSON `{id,prompt}` lists): training prompt `A red cube on a white table.`
with seed 111; validation prompt `A blue ceramic vase on a wooden desk.` with
seed 222. `MODEL_PATH` below denotes the local Z-Image checkpoint.

Run these GPU commands inside a single-GPU `debug` allocation; fit runs on CPU.
All output paths must be new. The validation retry used a new directory.

```bash
python -m tools.freecache.collect trace --model zimage --model-path "$MODEL_PATH" \
  --prompts train.json --seeds 111 --width 512 --height 512 --guidance 5 \
  --output outputs/smoke/trace
python -m tools.freecache.collect propagation --model zimage --model-path "$MODEL_PATH" \
  --prompts train.json --seeds 111 --width 512 --height 512 --guidance 5 \
  --relative-epsilon .1 --inject-steps 4 40 --output outputs/smoke/propagation
python -m tools.freecache.preprocess fit --traces outputs/smoke/trace/traces.json \
  --propagation outputs/smoke/propagation --coherence 0 --warmup 10 --budgets 25 \
  --output outputs/smoke/candidates.json
python -m tools.freecache.evaluate --model zimage --model-path "$MODEL_PATH" \
  --prompts holdout.json --seeds 222 --width 512 --height 512 --guidance 5 \
  --candidates outputs/smoke/candidates.json --lpips --meancache \
  --output outputs/smoke/validation
```

Trace and two-position propagation collection completed in job 245141; observer
parity passed. Following a metadata-order fix, CPU fitting reused those complete
data without resampling. Job 245146 completed validation successfully in 1m13s:
F50 equals no-cache RGB exactly; candidate F25, warmup-periodic ZOH F25 and
MeanCache F25 all generated and were scored with native 512² Alex LPIPS.
The two-position probe is an API smoke only, not evidence that sparse probes
adequately describe early propagation.

Failures remain in cost accounting:

| Job | Allocation | Result |
| --- | --- | --- |
| 245138 | 1s | Import failed before model load: third-party `tools` package shadowing; fixed by explicit source package |
| 245139 | 48s | Z-Image has no transformer-wide `attn_processors`; metadata now discovers processors from modules |
| 245141 | 2m06s | Trace/probe completed; CPU fit then rejected equivalent settings due to unordered Diffusers default-key metadata; comparison now normalizes only that order |
| 245143 | 1m18s | F50 parity passed; evaluator assumed object stats, while runtime returned a dict; both forms now supported |
| 245146 | 1m13s | Validation completed, exit 0 |

Total GPU allocation including failures: 5m26s. This is integration-test cost,
not a bound on real model preprocessing. Ignored local run records are under
`outputs/preprocess-smoke/` and `outputs/preprocess-smoke-r2/`; they are not
dependencies of the public reproduction command.

## Custom step counts and one-command entry point — 2026-09-18

229 FlexCache tests pass, including custom 4/40/60-step schedules, all-Fresh
parity, step mismatch rejection, fitting and evaluator coverage. Exact CPU
reproduction still matches all 150 released profiles. Documentation builds
with `mkdocs build --strict`; Ruff and `git diff --check` pass.

The 40-step GPU smoke uses the same H20 environment, training/validation
prompts and seeds as above. Within a single-GPU Slurm `debug` allocation:

```bash
python -m tools.freecache.run --model zimage --model-path "$MODEL_PATH" \
  --prompts train.json --seeds 111 --steps 40 --width 512 --height 512 \
  --guidance 5 --warmup 8 --budgets 16 20 28 --relative-epsilon .1 \
  --inject-steps 3 32 --output outputs/preprocess-smoke-40
python -m tools.freecache.evaluate --model zimage --model-path "$MODEL_PATH" \
  --prompts holdout.json --seeds 222 --steps 40 --width 512 --height 512 \
  --guidance 5 --candidates outputs/preprocess-smoke-40/candidates.json \
  --lpips --output outputs/preprocess-smoke-40/validation
```

Collection and fitting completed in 194.31 seconds. The three emitted profiles
have `reference_steps=40`, exactly 16/20/28 Fresh steps, and protect steps 0–7.
This small two-probe smoke checks integration, not calibration quality or a
recommended probe count. Default probes are spread over the requested step
grid; CPU tests also verify their bounds for 4/40/60 steps.

Job 245166 completed successfully (exit 0) in 4m50s, including collection,
fitting and evaluation. Held-out evaluation completed all six paired cells
(three candidates and their equal-budget warmup-ZOH controls), with expected
Fresh counts and native 512² PSNR/LPIPS. All-Fresh F40 equals no-cache RGB
exactly. Raw outputs remain ignored under `outputs/preprocess-smoke-40/`.
