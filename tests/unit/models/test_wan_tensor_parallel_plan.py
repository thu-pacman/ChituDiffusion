"""The Wan plan against every Wan graph it claims to cover.

The Wan 2.1 sizes and tasks share ``WanTransformer3DModel``, so one table
serves all of them and this is where that claim is checked. The plan is
validated rather than applied, which needs no process group and lets every
degree be tested on a meta-device model.
"""

from __future__ import annotations

import pytest
import torch

from chitu_diffusion.models.wan.tensor_parallel import WAN_TENSOR_PARALLEL_PLAN
from chitu_diffusion.parallel.tp import (
    TensorParallelPlan,
    validate_tensor_parallel_plan,
)

transformer_wan = pytest.importorskip(
    "diffusers.models.transformers.transformer_wan"
)

SHARED = {
    "num_layers": 1,
    "patch_size": (1, 2, 2),
    "text_dim": 4096,
    "freq_dim": 256,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "rope_max_seq_len": 64,
}

# Widths and head counts from the published checkpoints; the depth is cut to
# one block because the patterns are per-block and coverage does not depend on
# how many there are.
# The degrees each variant can actually run: the head count and the
# feed-forward width both have to divide.
VARIANTS = {
    "wan2.1-1.3b": (
        {
            "num_attention_heads": 12,
            "attention_head_dim": 128,
            "ffn_dim": 8960,
            "in_channels": 16,
            "out_channels": 16,
        },
        (2, 4),
    ),
    "wan2.1-14b-t2v": (
        {
            "num_attention_heads": 40,
            "attention_head_dim": 128,
            "ffn_dim": 13824,
            "in_channels": 16,
            "out_channels": 16,
        },
        (2, 4, 8),
    ),
    "wan2.1-14b-i2v": (
        {
            "num_attention_heads": 40,
            "attention_head_dim": 128,
            "ffn_dim": 13824,
            "in_channels": 36,
            "out_channels": 16,
            "image_dim": 1280,
            # The image embedder lifts the image context to the hidden width
            # before the blocks see it, so this is 40 * 128, not image_dim.
            "added_kv_proj_dim": 5120,
        },
        (2, 4, 8),
    ),
}

DEGREES = sorted(
    {(variant, degree) for variant, (_, degrees) in VARIANTS.items() for degree in degrees}
)


def _meta_model(variant: str):
    with torch.device("meta"):
        return transformer_wan.WanTransformer3DModel(**SHARED, **VARIANTS[variant][0])


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_plan_covers_every_linear_in_every_wan_variant(variant: str) -> None:
    is_i2v = "added_kv_proj_dim" in VARIANTS[variant][0]
    rewrite = validate_tensor_parallel_plan(
        _meta_model(variant), WAN_TENSOR_PARALLEL_PLAN, degree=1
    )

    assert rewrite.column == (
        "blocks.0.attn1.to_k",
        "blocks.0.attn1.to_q",
        "blocks.0.attn1.to_v",
        *(("blocks.0.attn2.add_k_proj", "blocks.0.attn2.add_v_proj") if is_i2v else ()),
        "blocks.0.attn2.to_k",
        "blocks.0.attn2.to_q",
        "blocks.0.attn2.to_v",
        "blocks.0.ffn.net.0.proj",
    )
    assert rewrite.row == (
        "blocks.0.attn1.to_out.0",
        "blocks.0.attn2.to_out.0",
        "blocks.0.ffn.net.2",
    )
    assert rewrite.norm == (
        "blocks.0.attn1.norm_k",
        "blocks.0.attn1.norm_q",
        *(("blocks.0.attn2.norm_added_k",) if is_i2v else ()),
        "blocks.0.attn2.norm_k",
        "blocks.0.attn2.norm_q",
    )
    # Conditioning stays whole: it is small next to the blocks, and the
    # modulation and unpatchify that consume it want the full width.
    assert rewrite.replicated == (
        *(
            (
                "condition_embedder.image_embedder.ff.net.0.proj",
                "condition_embedder.image_embedder.ff.net.2",
            )
            if is_i2v
            else ()
        ),
        "condition_embedder.text_embedder.linear_1",
        "condition_embedder.text_embedder.linear_2",
        "condition_embedder.time_embedder.linear_1",
        "condition_embedder.time_embedder.linear_2",
        "condition_embedder.time_proj",
        "proj_out",
    )


@pytest.mark.parametrize(("variant", "degree"), DEGREES)
def test_every_wan_variant_divides_evenly_at_the_degrees_it_supports(
    variant: str, degree: int
) -> None:
    rewrite = validate_tensor_parallel_plan(
        _meta_model(variant), WAN_TENSOR_PARALLEL_PLAN, degree=degree
    )

    heads = VARIANTS[variant][0]["num_attention_heads"]
    assert {local for _, _, local in rewrite.head_counts} == {heads // degree}


def test_wan21_1_3b_is_the_one_variant_that_stops_at_tp4() -> None:
    with pytest.raises(ValueError, match="attn1.heads=12"):
        validate_tensor_parallel_plan(
            _meta_model("wan2.1-1.3b"), WAN_TENSOR_PARALLEL_PLAN, degree=8
        )


def test_the_feed_forward_width_is_what_bounds_the_degree() -> None:
    # 40 heads would allow TP5 on the 14B models, but 13824 is not divisible by
    # it, so the usable set of degrees is narrower than the head count suggests
    # and the error has to name the feed-forward layer rather than the heads.
    for variant, degree in (("wan2.1-14b-t2v", 5), ("wan2.1-1.3b", 3)):
        assert VARIANTS[variant][0]["num_attention_heads"] % degree == 0
        with pytest.raises(ValueError, match="ffn.net.0.proj"):
            validate_tensor_parallel_plan(
                _meta_model(variant), WAN_TENSOR_PARALLEL_PLAN, degree=degree
            )


def test_qk_norm_across_heads_must_be_declared_as_a_reducing_norm() -> None:
    # This is the trap Wan sets. norm_q normalizes over heads * head_dim, so
    # leaving it replicated divides by a shard's sum of squares and returns a
    # wrong answer without raising. Dropping it from the plan must be refused.
    without_q_norm = TensorParallelPlan(
        column=WAN_TENSOR_PARALLEL_PLAN.column,
        row=WAN_TENSOR_PARALLEL_PLAN.row,
        replicated=WAN_TENSOR_PARALLEL_PLAN.replicated,
        norm=tuple(
            pattern
            for pattern in WAN_TENSOR_PARALLEL_PLAN.norm
            if not pattern.endswith("norm_q")
        ),
        head_counts=WAN_TENSOR_PARALLEL_PLAN.head_counts,
    )

    with pytest.raises(ValueError, match="does not list them under norm"):
        validate_tensor_parallel_plan(
            _meta_model("wan2.1-14b-t2v"), without_q_norm, degree=2
        )
