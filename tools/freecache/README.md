# FreeCache offline preprocessing

Source-checkout tools, independent of the archived research directory:

- `preprocess reproduce`: CPU reconstruction of all 150 released profiles.
- `run`: collect traces, measure propagation, and fit profiles in one command.
- `collect trace`: record full-compute, scheduler-consumed velocities as Grams.
- `collect propagation`: measure finite ZOH-direction perturbation responses.
- `preprocess fit`: produce experimental profiles with an explicit Fresh prefix.
- `evaluate`: paired no-cache/all-Fresh/candidate/warmup-ZOH validation, optional
  MeanCache/MagCache and native-resolution Alex LPIPS.

Run modules from the repository root with `python -m tools.freecache.<module>`.
GPU entry points require a single GPU allocated through Slurm `debug`. CPU
reproduction needs NumPy; collection needs the normal project dependencies;
LPIPS validation additionally needs the `eval` extra and cached AlexNet weights.

See [usage and parameters](../../docs/features/flexcache.md#freecache-preprocess) and
[canonical data provenance](data/README.md). New collection/evaluation output
directories must not exist. Generated runs belong under ignored `outputs/`;
canonical reproduction inputs under `data/` remain versioned. Nothing here
automatically modifies the runtime's bundled profiles.

All GPU commands accept `--steps` (default 50). Custom profiles must use the
same step count at inference; built-in Preview and MeanCache profiles remain
50-step profiles. Collection currently supports deterministic FlowMatch Euler
for FLUX, Qwen-Image and Z-Image.

Validate on separate prompts and seeds using the same generation settings:

```bash
python -m tools.freecache.evaluate --model zimage --model-path /path/to/Z-Image \
  --prompts holdout.json --seeds 456 --steps 40 --guidance 5 \
  --candidates outputs/freecache-40/candidates.json --lpips \
  --output outputs/freecache-40-validation
```

Run evaluation in a single-GPU Slurm `debug` allocation too. `--lpips` is optional;
PSNR is always computed. Both use native resolution. For CPU-only reproduction:

```bash
python -m tools.freecache.preprocess reproduce --check-runtime \
  --output outputs/freecache-reproduction/profiles.json
```

The [validation record](VALIDATION.md) includes exact reproduction, tests,
single-H20 smoke scope, and failed-attempt cost accounting.
