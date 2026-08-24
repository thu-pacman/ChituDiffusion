"""Generate with the Z-Image wrapper and MeanCache."""

import os

import torch

from chitu_diffusion import CacheConfig, MeanCacheConfig, ZImagePipeline, ZImageRequest

model_path = os.environ["ZIMAGE_MODEL_PATH"]
pipeline = ZImagePipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    cache=CacheConfig(
        strategy="meancache",
        params=MeanCacheConfig(fresh_steps=25),
    ),
)
try:
    result = pipeline.generate(
        ZImageRequest(
            prompt="a red cube on a white table",
            num_steps=50,
            seed=7,
        )
    )
    if pipeline.parallel_context.rank == 0:
        result.images[0].save("zimage.png")
        print(pipeline.last_cache_stats)
finally:
    pipeline.close()
