"""Generate with the Z-Image wrapper and TeaCache."""

import os

import torch

from chitu_diffusion import CacheConfig, TeaCacheConfig, ZImagePipeline, ZImageRequest

model_path = os.environ["ZIMAGE_MODEL_PATH"]
pipeline = ZImagePipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    cache=CacheConfig(
        strategy="teacache",
        params=TeaCacheConfig(threshold=0.2),
    ),
)
try:
    result = pipeline.generate(
        ZImageRequest(prompt="a red cube on a white table", seed=7)
    )
    if pipeline.parallel_context.rank == 0:
        result.images[0].save("zimage.png")
        print(pipeline.last_cache_stats)
finally:
    pipeline.close()
