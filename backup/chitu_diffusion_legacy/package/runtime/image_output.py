from __future__ import annotations

import os

from einops import rearrange
from PIL import Image
from torch import Tensor


def save_image_as_png(tensor: Tensor, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = tensor.clamp(-1, 1)
    x = rearrange(x, "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    img.save(save_path, quality=95, subsampling=0)
