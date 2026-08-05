# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Naive data-parallel helpers, kept model-agnostic on purpose.

Data parallel is the *outermost* serving layer: the same request is broadcast
to every rank and its ``n_sample`` batch is split evenly across ``dp_size``
replicas (``dp_size`` must divide ``n_sample``). Each replica is a self-contained
``cfg_size * cp_size`` block that runs the full model on its own seed slice, so
everything below DP (model path, attention/CP/CFG backends, VAE decode within a
replica) is unaware of data parallelism.

Model runtime adapters only need two touch points:
  * input side  -- call :func:`dp_shard_sample_seeds` to learn which seeds this
    replica should build/denoise;
  * output side -- the generator calls :func:`dp_gather_sample_batch` after
    ``decode_latents`` to reassemble the full ``n_sample`` batch on rank 0.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from chitu_diffusion.core.distributed.parallel_state import (
    get_dit_dp_group,
    get_dit_dp_size,
    get_dit_replica_index,
)


def dp_is_active() -> bool:
    """True when more than one data-parallel replica is configured."""
    return get_dit_dp_size() > 1


def dp_shard_sample_seeds(params, fallback: int = 0) -> list[int]:
    """Return the seed slice this replica is responsible for.

    Replica ``r`` owns the contiguous slice ``seeds[r*k : (r+1)*k]`` with
    ``k = n_sample // dp_size``. When ``dp_size == 1`` this returns all sample
    seeds unchanged, so single-replica behavior is identical to before.
    """
    all_seeds = params.sample_seeds(fallback)
    dp_size = get_dit_dp_size()
    total = len(all_seeds)
    if total % dp_size != 0:
        raise ValueError(
            f"n_sample ({total}) must be divisible by dp_size ({dp_size}) "
            f"for naive data parallel."
        )
    per_replica = total // dp_size
    replica_index = get_dit_replica_index()
    return all_seeds[replica_index * per_replica : (replica_index + 1) * per_replica]


def dp_gather_sample_batch(batch: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """All-gather per-replica sample batches into the full ``n_sample`` batch.

    Each replica decodes only its seed slice, so this concatenates the slices
    along the batch dim across the DP group. The DP group orders members by
    replica index (``rank_in_group == replica index``), so the samples come back
    in seed order. No-op when ``dp_size == 1`` or the tensor is missing (e.g. a
    non-leader rank that produced nothing).
    """
    if batch is None:
        return batch
    dp_group = get_dit_dp_group()
    if dp_group.group_size <= 1:
        return batch
    batch = batch.contiguous()
    gathered = [torch.empty_like(batch) for _ in range(dp_group.group_size)]
    dist.all_gather(gathered, batch, group=dp_group.gpu_group)
    return torch.cat(gathered, dim=0)
