import torch

from chitu_diffusion.flexcache.core.anchor_cache import AnchorCachePlanner
from chitu_diffusion.flexcache.strategy.model import ModelStrategy
from chitu_diffusion.flexcache.strategy.teacache import TeaCacheStrategy
from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend
from chitu_diffusion.runtime.task import DiffusionUserParams, FlexCacheParams


def test_flexcache_strategies_are_accepted():
    for strategy in ("model", "layer", "attn", "seq", "teacache", "pab", "cubic", "taylorseer", "ditango"):
        params = DiffusionUserParams(
            flexcache_params=FlexCacheParams(strategy=strategy)
        ).resolve_flexcache_params()
        assert params.strategy == strategy


def test_strategy_params_are_preserved():
    params = DiffusionUserParams(
        flexcache_params=FlexCacheParams(
            strategy="pab",
            cache_ratio=0.99,
            strategy_params={"skip_self_range": 4, "skip_cross_range": 6},
        )
    ).resolve_flexcache_params()
    assert params.strategy_params == {"skip_self_range": 4, "skip_cross_range": 6}


def test_legacy_baseline_params_are_accepted_as_strategy_params():
    params = DiffusionUserParams(
        flexcache_params=FlexCacheParams(
            strategy="pab",
            baseline_params={"skip_self_range": 4, "skip_cross_range": 6},
        )
    ).resolve_flexcache_params()
    assert params.strategy_params == {"skip_self_range": 4, "skip_cross_range": 6}


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


def test_model_ratio_interval_uses_cache_ratio_directly(monkeypatch):
    class Buffer:
        current_step = 2

    class CurrentTask:
        buffer = Buffer()

    class Generator:
        current_task = CurrentTask()

    monkeypatch.setattr(DiffusionBackend, "generator", Generator())
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)

    planner = AnchorCachePlanner(
        cache_ratio=0.5,
        warmup_steps=2,
        cooldown_steps=2,
        total_steps=18,
        tau_max=99,
        mode="model_ratio",
    )
    assert planner.tau_max == 3
    planner.curvature.curvature = {"model": 1.0}
    planner._refresh_model_interval("model", curvature=1.0, is_anchor=True)
    assert planner.intervals["model"] == 3


def test_model_strategy_reuses_until_next_compute_step(monkeypatch):
    class Buffer:
        current_step = 2

    class CurrentTask:
        buffer = Buffer()

    class Generator:
        current_task = CurrentTask()

    monkeypatch.setattr(DiffusionBackend, "generator", Generator())
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)

    class ReqParams:
        num_inference_steps = 18

    class Req:
        params = ReqParams()

    class Task:
        req = Req()

    monkeypatch.setattr(DiffusionBackend, "generator", Generator())
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)

    strategy = ModelStrategy(
        task=Task(),
        cache_ratio=0.5,
        warmup_steps=2,
        cooldown_steps=2,
        tau_max=8,
        curvature_interval_power=1.0,
    )
    x0 = torch.ones(1, 2)
    x1 = x0 * 1.01
    x2 = x0 * 1.40

    assert strategy.get_reuse_key(x=x0) is None
    assert strategy.get_store_key(x=x0) == "pos"
    strategy.planner.mark_computed("pos", x0)

    Buffer.current_step = 3
    assert strategy.get_reuse_key(x=x1) == "pos"

    next_step = strategy.planner.next_compute_step["pos"]
    Buffer.current_step = next_step
    assert strategy.get_reuse_key(x=x2) is None


def test_model_strategy_keeps_cfg_branches_independent(monkeypatch):
    class Buffer:
        current_step = 2

    class CurrentTask:
        buffer = Buffer()

    class Generator:
        current_task = CurrentTask()

    monkeypatch.setattr(DiffusionBackend, "generator", Generator())
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)

    class ReqParams:
        num_inference_steps = 18

    class Req:
        params = ReqParams()

    class Task:
        req = Req()

    monkeypatch.setattr(DiffusionBackend, "generator", Generator())
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)

    strategy = ModelStrategy(
        task=Task(),
        cache_ratio=0.5,
        warmup_steps=2,
        cooldown_steps=2,
        tau_max=8,
        curvature_interval_power=1.0,
    )

    x0 = torch.ones(1, 2)
    assert strategy.get_reuse_key(x=x0) is None
    assert strategy.get_store_key(x=x0) == "pos"
    strategy.planner.mark_computed("pos", x0)

    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.NEG)
    assert strategy.get_reuse_key(x=x0 * 3) is None
    assert strategy.get_store_key(x=x0 * 3) == "neg"
    strategy.planner.mark_computed("neg", x0 * 3)

    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)
    Buffer.current_step = 3
    assert strategy.get_reuse_key(x=x0 * 1.01) == "pos"


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
