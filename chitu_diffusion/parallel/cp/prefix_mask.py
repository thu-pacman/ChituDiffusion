"""Split prefix-shaped bool masks so most query rows can use fused SDPA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class PrefixMaskPlan:
    """Query rows from ``start`` on attend to exactly the first keys."""

    start: int
    visible_keys: int

    def shard_queries(self, *, offset: int, length: int) -> "PrefixMaskPlan":
        """Translate a full-query plan onto one contiguous CP shard."""

        local_start = min(max(self.start - int(offset), 0), int(length))
        return PrefixMaskPlan(start=local_start, visible_keys=self.visible_keys)


def shard_query_mask(
    mask: torch.Tensor | None, *, offset: int, length: int, total_length: int
) -> torch.Tensor | None:
    """Keep this rank's query rows while retaining every key column."""

    if mask is None:
        return None
    if mask.shape[-2] == length:
        return mask
    if mask.shape[-2] != total_length:
        raise ValueError(
            f"attention mask has {mask.shape[-2]} query rows; expected "
            f"{total_length} full rows or {length} local rows"
        )
    return mask.narrow(-2, offset, length)


def plan_prefix_mask(mask: torch.Tensor | None) -> PrefixMaskPlan | None:
    """Describe a mask as leading masked rows and a maskless prefix-key tail.

    Returns ``None`` when the mask has no such shape or when the tail is too
    short to pay for an extra attention call. This performs one device-to-host
    synchronisation, so callers should plan once per step rather than per layer.
    """

    if mask is None or mask.dtype != torch.bool or mask.ndim != 4:
        return None
    rows = mask.flatten(0, 1)
    queries, keys = rows.shape[-2], rows.shape[-1]
    if queries < 2:
        return None

    first_false = rows.to(torch.uint8).argmin(-1)
    visible = torch.where(rows.all(-1), torch.full_like(first_false, keys), first_false)
    is_prefix = rows.sum(-1) == visible
    lengths = visible[0]
    shared = (
        is_prefix.all(0) & visible.eq(lengths).all(0) & lengths.eq(lengths[-1])
    )

    divergent = (~shared).nonzero()
    start = int(divergent[-1, 0]) + 1 if divergent.numel() else 0
    visible_keys = int(lengths[-1])
    if visible_keys == 0 or queries - start < queries // 2:
        return None
    return PrefixMaskPlan(start=start, visible_keys=visible_keys)


def prefix_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    plan: PrefixMaskPlan | None,
) -> torch.Tensor:
    """Run SDPA while routing the query rows described by ``plan`` around a mask."""

    if plan is None:
        if mask is not None and query.device.type == "cuda":
            query, key, value = (
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
            )
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0
        )
    tail = F.scaled_dot_product_attention(
        query[:, :, plan.start :].contiguous(),
        key[:, :, : plan.visible_keys].contiguous(),
        value[:, :, : plan.visible_keys].contiguous(),
        dropout_p=0.0,
    )
    if plan.start == 0:
        return tail
    leading = F.scaled_dot_product_attention(
        query[:, :, : plan.start].contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None if mask is None else mask[:, :, : plan.start],
        dropout_p=0.0,
    )
    return torch.cat([leading, tail], dim=2)


__all__ = [
    "PrefixMaskPlan",
    "plan_prefix_mask",
    "prefix_masked_attention",
    "shard_query_mask",
]
