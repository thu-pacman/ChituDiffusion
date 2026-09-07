from __future__ import annotations

import pytest
import torch
from torch import nn

from chitu_diffusion.models.zimage.tensor_parallel import ZIMAGE_TENSOR_PARALLEL_PLAN
from chitu_diffusion.parallel.tp import (
    PlainColumnParallelLinear,
    PlainRowParallelLinear,
    TensorParallelTopology,
    apply_tensor_parallel_plan,
    use_tensor_parallel_topology,
)

transformer_z_image = pytest.importorskip(
    "diffusers.models.transformers.transformer_z_image"
)

TINY = {
    "all_patch_size": [2],
    "all_f_patch_size": [1],
    "axes_dims": [4, 6, 6],
    "axes_lens": [64, 32, 32],
    "cap_feat_dim": 32,
    "dim": 64,
    "in_channels": 16,
    "n_heads": 4,
    "n_kv_heads": 4,
    "n_layers": 2,
    "n_refiner_layers": 1,
    "norm_eps": 1e-05,
    "qk_norm": True,
    "rope_theta": 256.0,
    "siglip_feat_dim": None,
    "t_scale": 1000.0,
}


def _model() -> nn.Module:
    with torch.device("meta"):
        return transformer_z_image.ZImageTransformer2DModel(**TINY)


def _topology(degree: int) -> TensorParallelTopology:
    return TensorParallelTopology(
        ranks=tuple(range(degree)),
        rank=0,
        rank_in_group=0,
        degree=degree,
        process_group=None,
    )


def test_the_plan_names_every_linear_in_the_graph():
    model = _model()

    rewrite = apply_tensor_parallel_plan(model, ZIMAGE_TENSOR_PARALLEL_PLAN, degree=1)

    linears = sum(1 for module in model.modules() if isinstance(module, nn.Linear))
    assert rewrite.sharded + len(rewrite.replicated) == linears


def test_only_the_hidden_dimension_projections_shard():
    rewrite = apply_tensor_parallel_plan(_model(), ZIMAGE_TENSOR_PARALLEL_PLAN, degree=1)

    blocks = TINY["n_layers"] + 2 * TINY["n_refiner_layers"]
    assert len(rewrite.column) == 5 * blocks
    assert len(rewrite.row) == 2 * blocks
    assert all(
        name.endswith(("to_q", "to_k", "to_v", "w1", "w3")) for name in rewrite.column
    )
    assert all(name.endswith(("to_out.0", "w2")) for name in rewrite.row)
    assert all(
        "adaLN_modulation" in name
        or name.startswith(("all_x_embedder", "all_final_layer", "cap_embedder", "t_embedder"))
        for name in rewrite.replicated
    )


def test_the_modulation_and_output_heads_stay_whole_so_blocks_need_no_gather():
    model = _model()
    with use_tensor_parallel_topology(_topology(2)):
        apply_tensor_parallel_plan(model, ZIMAGE_TENSOR_PARALLEL_PLAN, degree=2)

    block = model.layers[0]
    assert isinstance(block.attention.to_q, PlainColumnParallelLinear)
    assert isinstance(block.attention.to_out[0], PlainRowParallelLinear)
    assert isinstance(block.feed_forward.w2, PlainRowParallelLinear)
    assert isinstance(block.adaLN_modulation[0], nn.Linear)
    assert not isinstance(block.adaLN_modulation[0], PlainColumnParallelLinear)
    assert block.adaLN_modulation[0].out_features == 4 * TINY["dim"]
    assert block.attention.heads == TINY["n_heads"] // 2


def test_a_degree_that_does_not_divide_the_head_count_is_refused():
    with pytest.raises(ValueError, match="not divisible by tensor-parallel degree 3"):
        apply_tensor_parallel_plan(_model(), ZIMAGE_TENSOR_PARALLEL_PLAN, degree=3)
