# Chitu API compatibility entry point

`experiments/chitu_api` now contains only the compatibility shell required by
the original SGLang-Omni image-decoder experiment. The reusable runtime, EPAC
scheduler, Z-Image adapter, parallel attention implementations, benchmark
harness, and documentation live in `chitu_diffusers/`.

## Retained files

| Path | Purpose |
| --- | --- |
| `serve.py` | Torchrun and `chitu run` compatibility entry point |
| `launch.py` | Launch a StageConfig with `torch.distributed.run` |
| `stage_config.example.yaml` | Generic image-decoder StageConfig example |
| `benchmark_epe.yaml` | Four-GPU Z-Image benchmark configuration |
| `requirements.md` | SGLang-Omni image-decoder integration contract |
| `test_service.py` | StageConfig and HTTP compatibility tests |
| `test_zimage_standalone.py` | Native Diffusers and EPAC consistency tests |

The previous local service, protocol, Z-Image model fork, benchmark client,
timeline parser, and process result log were removed after promotion. Do not
add model/runtime implementations back under this experiment directory.

## Launch

Update the model path and GPU placement in a StageConfig, then run:

```bash
../ChituDiffusion/.venv/bin/torchrun \
  --standalone \
  --nproc-per-node=4 \
  --module experiments.chitu_api.serve \
  --stage-config experiments/chitu_api/stage_config.example.yaml
```

The same config can be launched through the compatibility helper:

```bash
../ChituDiffusion/.venv/bin/python -m experiments.chitu_api.launch \
  --stage-config experiments/chitu_api/stage_config.example.yaml
```

Rank 0 prints `CHITU_API_READY` and serves the HTTP API. All implementation and
benchmark commands are documented in `chitu_diffusers/README.md` and
`chitu_diffusers/benchmarks/README.md`.

## Validation

```bash
PYTHONPATH="$PWD" ../ChituDiffusion/.venv/bin/pytest -q \
  experiments/chitu_api/test_service.py \
  experiments/chitu_api/test_zimage_standalone.py
```
