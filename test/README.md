# ChituDiffusion Test Launches

These files are smoke-test launchers for real `chitu run` paths. They are meant
to run on the Slurm GPU partition configured in each YAML file.

Install optional backends before running the full matrix:

```bash
uv sync --extra flash --extra flashinfer --extra sage --extra sparge
```

Run Flux attention backends:

```bash
test/run_flux_attention_backends.sh
CHITU_ATTN_BACKENDS="sage sparge" test/run_flux_attention_backends.sh
```

Run Flux CP Ulysses backends:

```bash
test/run_flux_cp_backends.sh
CHITU_CP_ATTN_BACKENDS="sage sparge" test/run_flux_cp_backends.sh
```

Run Wan attention and CP smoke tests:

```bash
test/run_wan_attention_backends.sh
test/run_wan_cp_backends.sh
```

Run FlexCache smoke tests:

```bash
test/run_flexcache_smoke.sh
CHITU_FLEXCACHE_STRATEGIES="blockdance,taylorseer" test/run_flexcache_smoke.sh
CHITU_FLUX_FLEXCACHE_STRATEGIES="teacache pab" test/run_flexcache_smoke.sh
```

Useful environment overrides:

- `CHITU_BIN=/path/to/chitu`
- `CHITU_ATTN_BACKENDS="torch_sdpa flash flashinfer sage sparge"`
- `CHITU_CP_ATTN_BACKENDS="sage sparge"`
- `CHITU_WAN_STEPS=3`
- `CHITU_FLEXCACHE_STRATEGIES="teacache,pab,blockdance,taylorseer,cubic"`
- `CHITU_FLUX_FLEXCACHE_STRATEGY=blockdance`

The config files live under `test/configs/` and can also be launched directly:

```bash
.venv/bin/chitu run test/configs/flux_attention_sparge.yaml
.venv/bin/chitu run test/configs/wan_cp_sage.yaml
```
