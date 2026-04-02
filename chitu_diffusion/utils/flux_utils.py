import math
import os

import torch
from einops import rearrange
from PIL import Image
from torch import Tensor


def prc_txt(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _l, _ = x.shape
    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),
        "w": torch.arange(1),
        "l": torch.arange(_l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids.to(x.device)


def prc_img(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _, h, w = x.shape
    x_coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(h),
        "w": torch.arange(w),
        "l": torch.arange(1),
    }
    x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids.to(x.device)


def batched_wrapper(fn):
    def batched_prc(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)
        return torch.stack(x), torch.stack(x_ids)
    return batched_prc


batched_prc_img = batched_wrapper(prc_img)
batched_prc_txt = batched_wrapper(prc_txt)


def compress_time(t_ids: Tensor) -> Tensor:
    assert t_ids.ndim == 1
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(
        len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype
    )
    t_ids_compressed = t_remap[t_ids]
    return t_ids_compressed


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    """Using position ids to scatter tokens back into spatial layout."""
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)

        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
    return x_list


def generalized_time_snr_shift(t: Tensor, mu: float, sigma: float) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def save_image_as_png(tensor: Tensor, save_path: str):
    """Save a (C, H, W) tensor in [-1, 1] range as a PNG image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = tensor.clamp(-1, 1)
    x = rearrange(x, "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    img.save(save_path, quality=95, subsampling=0)