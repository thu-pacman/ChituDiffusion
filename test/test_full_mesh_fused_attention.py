from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from chitu_diffusion.parallel.fast_cp.agkv import (
    FastAgkvAttention,
)


@dataclass
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.bfloat16
    is_cuda: bool = True

    @property
    def ndim(self) -> int:
        return len(self.shape)


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("head_dim", [8, 64, 72, 96, 104, 128, 136, 192, 200, 256])
def test_supports_dynamic_batch_and_sm90_head_dims(
    batch: int,
    head_dim: int,
) -> None:
    shape = (batch, 17, 12, head_dim)
    query = _FakeTensor(shape)
    key = _FakeTensor(shape)
    value = _FakeTensor(shape)

    assert FastAgkvAttention.supports(query, key, value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "shape",
    [
        (0, 17, 12, 128),
        (1, 17, 0, 128),
        (1, 17, 12, 7),
        (1, 17, 12, 65),
        (1, 17, 12, 264),
    ],
)
def test_rejects_unsupported_shapes(shape: tuple[int, ...]) -> None:
    query = _FakeTensor(shape)
    key = _FakeTensor(shape)
    value = _FakeTensor(shape)

    assert not FastAgkvAttention.supports(  # type: ignore[arg-type]
        query,
        key,
        value,
    )


def test_shards_and_slot_offsets_are_token_granular() -> None:
    lengths = FastAgkvAttention.shard_lengths(13, 4)
    assert lengths == (4, 3, 3, 3)

    attention = FastAgkvAttention.__new__(FastAgkvAttention)
    attention.world_size = 4
    attention.rank = 1
    attention.source_chunk_streaming = True
    attention.chunk_count = 1
    assert attention._slot_starts(lengths) == [0, 3, 6, 10]

    attention.chunk_count = 2
    assert attention._slot_starts(lengths) == [0, 1, 2, 4, 5, 7, 9, 11]


def test_shards_require_only_one_token_per_rank() -> None:
    assert FastAgkvAttention.shard_lengths(4, 4) == (1, 1, 1, 1)
    with pytest.raises(ValueError, match="one token per rank"):
        FastAgkvAttention.shard_lengths(3, 4)
