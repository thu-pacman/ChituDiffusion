# Examples

These files demonstrate public library APIs; command-line implementation lives
under `chitu_diffusion.commands`.

- `generate_zimage_flexcache.py`: `ZImagePipeline`/`ZImageRequest` plus MeanCache
- `generate_flux1.py`: `Flux1Pipeline`/`Flux1Request`
- `generate_flux2_klein.py`: fixed-CP FLUX.2-klein pipeline
- `generate_qwen_image.py`: `QwenImagePipeline`/`QwenImageRequest`
- `generate_llada_image.py`: `LLaDAImagePipeline`/`LLaDAImageRequest`
- `generate_wan.py`: `WanPipeline`/`WanRequest`
- `epe_embedded.py`: embedded EPE lifecycle without HTTP
- `epe_serve.py`: pipeline-backed EPE HTTP service
- `stage-zimage.yaml`: four-GPU `chitu serve` configuration
- `stage-minimax-h3.yaml`: eight-GPU elastic `chitu serve` at TP4×CP2
- `stage-hunyuan-image3.yaml`: eight-GPU static-CP `chitu serve` at
  TP2×CFG2×CP2×EP2; the `parallelism.model` degrees select other single-node
  topologies
- `static_cp.py`: static NCCL (`torch`) and Fast AGKV CP

Run generation examples directly for one rank or under `torchrun`. For example:

```bash
export ZIMAGE_MODEL_PATH=/path/to/Z-Image
torchrun --standalone --nproc-per-node=4 examples/static_cp.py --transport torch
torchrun --standalone --nproc-per-node=4 examples/static_cp.py --transport fast_agkv
```

Other Python examples use the model-path environment variable named in the file.
Edit `factory_args.model_path` before using `stage-zimage.yaml`.
