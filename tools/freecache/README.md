# FreeCache offline preprocessing

Source-checkout tools, independent of the archived research directory:

- `preprocess reproduce`: CPU reconstruction of all 150 released profiles.
- `collect trace`: record full-compute, scheduler-consumed velocities as Grams.
- `collect propagation`: measure finite ZOH-direction perturbation responses.
- `preprocess fit`: produce experimental profiles with an explicit Fresh prefix.
- `evaluate`: paired no-cache/F50/candidate/warmup-ZOH validation, optional
  MeanCache/MagCache and native-resolution Alex LPIPS.

Run modules from the repository root with `python -m tools.freecache.<module>`.
GPU entry points require a single GPU allocated through Slurm `debug`. CPU
reproduction needs NumPy; collection needs the normal project dependencies;
LPIPS validation additionally needs the `eval` extra and cached AlexNet weights.

See [standard workflow](../../docs/usage/freecache-preprocess.md) and
[canonical data provenance](data/README.md). New collection/evaluation output
directories must not exist. Generated runs belong under ignored `outputs/`;
canonical reproduction inputs under `data/` remain versioned. Nothing here
automatically modifies the runtime's bundled profiles.

The [validation record](VALIDATION.md) includes exact reproduction, tests,
single-H20 smoke scope, and failed-attempt cost accounting.
