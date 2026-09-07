import torch

from chitu_diffusion.parallel.cp import (
    PrefixMaskPlan,
    plan_prefix_mask,
    prefix_masked_attention,
    shard_query_mask,
)


def _denoise_mask(batch: int = 2, queries: int = 9, keys: int = 10) -> torch.Tensor:
    mask = torch.ones(batch, 1, queries, keys, dtype=torch.bool)
    mask[:, :, 0, keys - queries :] = False
    mask[:, :, 1:, keys - 1] = False
    return mask


def test_denoise_mask_leaves_every_image_row_maskless() -> None:
    assert plan_prefix_mask(_denoise_mask()) == PrefixMaskPlan(
        start=1, visible_keys=9
    )


def test_prefix_plan_localizes_onto_each_cp_query_shard() -> None:
    plan = PrefixMaskPlan(start=1, visible_keys=9)

    assert plan.shard_queries(offset=0, length=5) == PrefixMaskPlan(1, 9)
    assert plan.shard_queries(offset=5, length=4) == PrefixMaskPlan(0, 9)


def test_query_mask_sharding_keeps_all_key_columns() -> None:
    mask = _denoise_mask()

    local = shard_query_mask(mask, offset=5, length=4, total_length=9)

    assert local is not None and local.shape == (2, 1, 4, 10)
    torch.testing.assert_close(local, mask[:, :, 5:9])


def test_a_fully_visible_mask_needs_no_rows_at_all() -> None:
    assert plan_prefix_mask(
        torch.ones(2, 1, 6, 6, dtype=torch.bool)
    ) == PrefixMaskPlan(start=0, visible_keys=6)


def test_a_causal_prefill_mask_is_not_worth_splitting() -> None:
    causal = torch.ones(8, 8, dtype=torch.bool).tril().expand(2, 1, 8, 8)
    assert plan_prefix_mask(causal) is None


def test_rows_that_see_past_a_gap_keep_the_mask() -> None:
    mask = _denoise_mask()
    mask[:, :, -1, 4] = False
    assert plan_prefix_mask(mask) is None


def test_batch_entries_must_agree_before_a_row_goes_maskless() -> None:
    mask = _denoise_mask()
    mask[1, :, 3:, -2] = False
    assert plan_prefix_mask(mask) is None


def test_non_boolean_or_absent_masks_are_left_alone() -> None:
    assert plan_prefix_mask(None) is None
    assert plan_prefix_mask(torch.zeros(2, 1, 4, 4)) is None
    assert plan_prefix_mask(torch.ones(2, 4, 4, dtype=torch.bool)) is None


def test_split_attention_matches_the_masked_reference() -> None:
    torch.manual_seed(0)
    batch, heads, queries, keys, head_dim = 2, 3, 9, 10, 8
    mask = _denoise_mask(batch, queries, keys)
    query = torch.randn(batch, heads, queries, head_dim)
    key = torch.randn(batch, heads, keys, head_dim)
    value = torch.randn(batch, heads, keys, head_dim)

    plan = plan_prefix_mask(mask)
    split = prefix_masked_attention(query, key, value, mask=mask, plan=plan)
    reference = prefix_masked_attention(query, key, value, mask=mask, plan=None)

    assert plan is not None
    torch.testing.assert_close(split, reference)


def test_a_fully_visible_plan_skips_the_leading_call() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 2, 5, 4)
    key = torch.randn(1, 2, 5, 4)
    value = torch.randn(1, 2, 5, 4)

    split = prefix_masked_attention(
        query, key, value, mask=None, plan=PrefixMaskPlan(start=0, visible_keys=5)
    )

    torch.testing.assert_close(
        split, prefix_masked_attention(query, key, value, mask=None, plan=None)
    )
