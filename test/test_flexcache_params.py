import torch
import torch.nn as nn

from chitu_diffusion.core.models.backbone import BackboneMixin
from chitu_diffusion.flexcache.core.anchor_cache import AnchorCachePlanner
from chitu_diffusion.flexcache.flexcache_manager import FlexCacheManager
from chitu_diffusion.flexcache.params import (
    BlockDanceParams,
    CubicParams,
    DiTangoParams,
    PABParams,
    TaylorSeerParams,
    TeaCacheParams,
)
from chitu_diffusion.flexcache.strategy.blockdance import BlockDanceStrategy
from chitu_diffusion.flexcache.strategy.teacache import TeaCacheStrategy
from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend
from chitu_diffusion.runtime.task import DiffusionUserParams


def test_flexcache_strategies_are_accepted():
    for cls in (BlockDanceParams, TeaCacheParams, PABParams, CubicParams, TaylorSeerParams, DiTangoParams):
        params = DiffusionUserParams(
            flexcache_params=cls()
        ).resolve_flexcache_params()
        assert params.strategy == cls().strategy
        assert isinstance(params, cls)


def test_flexcache_param_dict_uses_concrete_fields():
    params = DiffusionUserParams(
        flexcache_params={
            "strategy": "pab",
            "skip_self_range": 4,
            "skip_cross_range": 6,
        }
    ).resolve_flexcache_params()
    assert isinstance(params, PABParams)
    assert params.skip_self_range == 4
    assert params.skip_cross_range == 6


def test_concrete_params_are_strategy_specific():
    blockdance = DiffusionUserParams(
        flexcache_params=BlockDanceParams(warmup=7, cooldown=3)
    ).resolve_flexcache_params()
    assert isinstance(blockdance, BlockDanceParams)
    assert not hasattr(blockdance, "tau_max")
    assert not hasattr(blockdance, "curvature_interval_power")

    cubic = DiffusionUserParams(
        flexcache_params=CubicParams(target_speedup=2.5, tau_max=6)
    ).resolve_flexcache_params()
    assert isinstance(cubic, CubicParams)
    assert not hasattr(cubic, "cache_ratio")
    assert not hasattr(cubic, "partition_mode")
    assert not hasattr(cubic, "anchor_interval")
    assert not hasattr(cubic, "alpha")
    assert cubic.target_speedup == 2.5
    assert cubic.tau_max == 6


def test_interval_plan_preserves_curvature_order():
    planner = AnchorCachePlanner(
        cache_ratio=0.8,
        warmup_steps=0,
        cooldown_steps=0,
        total_steps=10,
        tau_max=8,
        curvature_interval_power=1.0,
    )
    planner.curvature.curvature = {
        "high": 10.0,
        "medium": 1.0,
        "low": 0.1,
    }
    planner._refresh_intervals(["high", "medium", "low"])
    assert planner.intervals["high"] <= planner.intervals["medium"]
    assert planner.intervals["medium"] <= planner.intervals["low"]
    assert max(planner.intervals.values()) <= 8


def test_blockdance_reuse_skips_shallow_blocks(monkeypatch):
    class Buffer:
        current_step = 4

    class CurrentTask:
        task_id = "blockdance"
        buffer = Buffer()

    class Generator:
        current_task = CurrentTask()

    class ReqParams:
        num_inference_steps = 10

    class Req:
        params = ReqParams()

    class Task:
        req = Req()

    class FakeBlock(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
            self.calls = 0

        def forward(self, x, **kwargs):
            self.calls += 1
            return x + self.value

    class FakeModule(BackboneMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([FakeBlock(1), FakeBlock(10), FakeBlock(100), FakeBlock(1000)])

        def model_compute(self, tokens, **kwargs):
            x = tokens
            for block in self.blocks:
                x = block(x, **kwargs)
            return x

    monkeypatch.setattr(DiffusionBackend, "generator", Generator())
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)
    monkeypatch.setattr(DiffusionBackend, "flexcache", FlexCacheManager(max_cache_memory=20))

    module = FakeModule()
    strategy = BlockDanceStrategy(
        task=Task(),
        warmup_steps=0,
        cooldown_steps=0,
        boundary_block=1,
        group_size=2,
        start_fraction=0.4,
        end_fraction=0.95,
    )
    DiffusionBackend.flexcache.set_strategy(strategy)
    strategy.wrap_module_with_strategy(module)

    tokens = torch.zeros(1, 1)
    cache_output = module.model_compute(tokens, raw_e=torch.ones(1, 1, 1))
    assert cache_output.item() == 1111
    assert [block.calls for block in module.blocks] == [1, 1, 1, 1]

    Buffer.current_step = 5
    reuse_output = module.model_compute(torch.full((1, 1), 999.0), raw_e=torch.ones(1, 1, 1))
    assert reuse_output.item() == 1111
    assert [block.calls for block in module.blocks] == [1, 1, 2, 2]


def test_default_backbone_delta_accepts_cached_tensor():
    class FakeBlock(nn.Module):
        def forward(self, x, **kwargs):
            return x + 2

    class FakeModule(BackboneMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([FakeBlock()])

    module = FakeModule()
    state = module.backbone_make_state(torch.ones(1, 2))
    before = module.backbone_cache_state(state)
    after = module.backbone_run_block(module.backbone_blocks()[0], state)
    delta = module.backbone_block_delta(before, module.backbone_cache_state(after))
    restored = module.backbone_apply_block_delta(module.backbone_make_state(torch.ones(1, 2)), delta)

    assert torch.equal(delta, torch.full((1, 2), 2.0))
    assert torch.equal(module.backbone_state_tensor(restored), torch.full((1, 2), 3.0))


def test_teacache_cfg_branches_keep_independent_state(monkeypatch):
    class Buffer:
        current_step = 1

    class CurrentTask:
        buffer = Buffer()

    class Generator:
        current_task = CurrentTask()

    class ReqParams:
        num_inference_steps = 4

    class Req:
        params = ReqParams()

    class Task:
        req = Req()

    monkeypatch.setattr(DiffusionBackend, "generator", Generator())
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)

    strategy = TeaCacheStrategy(
        task=Task(),
        teacache_thresh=10.0,
        coefficients=[1.0, 0.0],
        warmup_steps=0,
        cooldown_steps=0,
    )

    e0_pos_0 = torch.ones(1, 1, 1)
    e0_pos_1 = e0_pos_0 * 1.1
    e0_neg_0 = e0_pos_0 * 3.0
    e0_neg_1 = e0_pos_0 * 3.3

    assert strategy.get_reuse_key(e0=e0_pos_0) is None
    assert strategy.get_store_key(e0=e0_pos_0) == "pos"

    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.NEG)
    assert strategy.get_reuse_key(e0=e0_neg_0) is None
    assert strategy.get_store_key(e0=e0_neg_0) == "neg"

    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)
    assert strategy.get_reuse_key(e0=e0_pos_1) == "pos"

    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.NEG)
    assert strategy.get_reuse_key(e0=e0_neg_1) == "neg"
