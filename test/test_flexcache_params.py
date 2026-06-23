import torch
import torch.nn as nn

from chitu_diffusion.models.backbone import BackboneMixin
from chitu_diffusion.flexcache.core.anchor_cache import AnchorCachePlanner
from chitu_diffusion.flexcache.flexcache_manager import FlexCacheManager
from chitu_diffusion.flexcache.params import (
    BlockDanceParams,
    CubicParams,
    DiTangoParams,
    MeanCacheParams,
    PABParams,
    TaylorSeerParams,
    TeaCacheParams,
)
from chitu_diffusion.flexcache.strategy.meancache import MeanCacheStrategy
from chitu_diffusion.flexcache.strategy.blockdance import BlockDanceStrategy
from chitu_diffusion.flexcache.strategy.taylorseer import TaylorSeerStrategy
from chitu_diffusion.flexcache.strategy.teacache import TeaCacheStrategy
from chitu_diffusion.runtime.generator import Generator
from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend
from chitu_diffusion.runtime.task import DiffusionUserParams


def test_flexcache_strategies_are_accepted():
    for cls in (BlockDanceParams, TeaCacheParams, PABParams, CubicParams, MeanCacheParams, TaylorSeerParams, DiTangoParams):
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


def test_ditango_param_dict_accepts_group_size_limit():
    params = DiffusionUserParams(
        flexcache_params={
            "strategy": "ditango",
            "cache_ratio": 0.6,
            "intra_group_size_limit": 2,
            "locality_group_compute_boost": 1.5,
            "groupwise_reuse_stale_kv": "true",
        }
    ).resolve_flexcache_params()
    assert isinstance(params, DiTangoParams)
    assert params.cache_ratio == 0.6
    assert params.intra_group_size_limit == 2
    assert params.locality_group_compute_boost == 1.5
    assert params.groupwise_reuse_stale_kv is True


def test_wan_meancache_strategy_can_be_built(monkeypatch):
    class ReqParams:
        num_inference_steps = 8

    class Req:
        params = ReqParams()

    class Task:
        req = Req()

    class Models:
        name = "Wan2.1-T2V-1.3B"

    class Args:
        models = Models()

    generator = object.__new__(Generator)
    generator.cp_size = 1
    generator.cfg_size = 2
    monkeypatch.setattr(DiffusionBackend, "args", Args())

    spec = MeanCacheParams(fresh_steps=10, warmup=1, cooldown=1)
    strategy = generator._build_flexcache_strategy(Task(), spec)
    assert isinstance(strategy, MeanCacheStrategy)
    assert strategy.fresh_steps == 10


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
            return super().model_compute(tokens, **kwargs)

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


def test_blockdance_cache_key_includes_cfg_branch_and_cp_rank(monkeypatch):
    class ReqParams:
        num_inference_steps = 10

    class Req:
        params = ReqParams()

    class Task:
        req = Req()

    class FakeCpGroup:
        group_size = 2
        rank_in_group = 1

    monkeypatch.setattr("chitu_diffusion.flexcache.strategy.blockdance.get_cp_group", lambda: FakeCpGroup())
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)

    strategy = BlockDanceStrategy(
        task=Task(),
        warmup_steps=0,
        cooldown_steps=0,
        boundary_block=3,
        group_size=2,
    )

    assert strategy._cache_key() == ("blockdance", "pos_cp1", 3)

    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.NEG)
    assert strategy._cache_key() == ("blockdance", "neg_cp1", 3)


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


def test_default_backbone_model_compute_routes_through_block_compute():
    class FakeBlock(nn.Module):
        def forward(self, x, **kwargs):
            return x + 1

    class FakeModule(BackboneMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([FakeBlock(), FakeBlock(), FakeBlock()])
            self.block_compute_calls = 0

        def block_compute(self, block_info, state):
            self.block_compute_calls += 1
            return super().block_compute(block_info, state)

    module = FakeModule()
    output = module.model_compute(torch.zeros(1, 1))

    assert output.item() == 3
    assert module.block_compute_calls == 3


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


def test_taylorseer_cfg_and_cp_branches_keep_independent_schedule(monkeypatch):
    class Buffer:
        current_step = 0

    class CurrentTask:
        task_id = "taylorseer"
        buffer = Buffer()

    class Generator:
        current_task = CurrentTask()

    class ReqParams:
        num_inference_steps = 6

    class Req:
        params = ReqParams()

    class Task:
        req = Req()

    class FakeCpGroup:
        group_size = 2
        rank_in_group = 1

    monkeypatch.setattr("chitu_diffusion.flexcache.strategy.taylorseer.get_cp_group", lambda: FakeCpGroup())
    monkeypatch.setattr(DiffusionBackend, "generator", Generator())

    strategy = TaylorSeerStrategy(
        task=Task(),
        fresh_threshold=2,
        max_order=1,
        first_enhance=1,
        warmup_steps=0,
        cooldown_steps=0,
    )

    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)
    assert strategy._current_type_for_step() == "full"
    assert strategy._cache_key(layer=0, module="self-attention") == (
        "taylorseer",
        "pos_cp1",
        0,
        "self-attention",
    )

    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.NEG)
    assert strategy._current_type_for_step() == "full"
    assert strategy._cache_key(layer=0, module="self-attention") == (
        "taylorseer",
        "neg_cp1",
        0,
        "self-attention",
    )

    Buffer.current_step = 1
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)
    assert strategy._current_type_for_step() == "Taylor"
    assert strategy._branch_states["pos_cp1"]["cache_counter"] == 1
    assert strategy._branch_states["neg_cp1"]["cache_counter"] == 0

    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.NEG)
    assert strategy._current_type_for_step() == "Taylor"
    assert strategy._branch_states["neg_cp1"]["cache_counter"] == 1


def test_taylorseer_wan_module_cache_reuses_local_outputs(monkeypatch):
    class Buffer:
        current_step = 0

    class CurrentTask:
        task_id = "taylorseer-block"
        buffer = Buffer()

    class Generator:
        current_task = CurrentTask()

    class ReqParams:
        num_inference_steps = 4

    class Req:
        params = ReqParams()

    class Task:
        req = Req()

    class FakeModuleOp(nn.Module):
        def __init__(self, delta):
            super().__init__()
            self.delta = delta
            self.calls = 0

        def forward(self, x, *args):
            self.calls += 1
            return x + self.delta

    class FakeWanBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.modulation = nn.Parameter(torch.zeros(1, 6, 1))
            self.norm1 = nn.Identity()
            self.self_attn = FakeModuleOp(1)
            self.norm3 = nn.Identity()
            self.cross_attn = FakeModuleOp(10)
            self.norm2 = nn.Identity()
            self.ffn = FakeModuleOp(100)

        def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
            e = (self.modulation + e).chunk(6, dim=1)
            y = self.self_attn(self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
            x = x + y * e[2]
            y = self.cross_attn(self.norm3(x), context, context_lens)
            x = x + y
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

    class FakeModule(BackboneMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([FakeWanBlock()])

        def backbone_make_state(self, tokens, **kwargs):
            return {
                "x": tokens,
                "kwargs": {
                    "e": kwargs["e"],
                    "seq_lens": None,
                    "grid_sizes": None,
                    "freqs": None,
                    "context": tokens,
                    "context_lens": None,
                },
            }

    monkeypatch.setattr("chitu_diffusion.flexcache.strategy.taylorseer.get_cp_group", lambda: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(DiffusionBackend, "generator", Generator())
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)
    monkeypatch.setattr(DiffusionBackend, "flexcache", FlexCacheManager(max_cache_memory=20))

    module = FakeModule()
    strategy = TaylorSeerStrategy(
        task=Task(),
        fresh_threshold=2,
        max_order=1,
        first_enhance=1,
        warmup_steps=0,
        cooldown_steps=0,
    )
    DiffusionBackend.flexcache.set_strategy(strategy)
    strategy.wrap_module_with_strategy(module)

    e = torch.ones(1, 6, 1)
    full_output = module.model_compute(torch.zeros(1, 1, 1), e=e)
    assert full_output.item() == 143
    assert module.blocks[0].self_attn.calls == 1
    assert module.blocks[0].cross_attn.calls == 1
    assert module.blocks[0].ffn.calls == 1
    assert set(DiffusionBackend.flexcache.cache) == {
        ("taylorseer", "pos_cp0", 0, "self-attention"),
        ("taylorseer", "pos_cp0", 0, "cross-attention"),
        ("taylorseer", "pos_cp0", 0, "ffn"),
    }

    Buffer.current_step = 1
    reuse_output = module.model_compute(torch.zeros(1, 1, 1), e=e)

    assert reuse_output.item() == 143
    assert module.blocks[0].self_attn.calls == 1
    assert module.blocks[0].cross_attn.calls == 1
    assert module.blocks[0].ffn.calls == 1


def test_taylorseer_flux_double_block_caches_ungated_modules(monkeypatch):
    class Buffer:
        current_step = 0

    class CurrentTask:
        task_id = "taylorseer-flux"
        buffer = Buffer()

    class Generator:
        current_task = CurrentTask()

    class ReqParams:
        num_inference_steps = 4

    class Req:
        params = ReqParams()

    class Task:
        req = Req()

    class FakeFluxNorm(nn.Module):
        def forward(self, x, emb):
            gate = torch.ones(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype)
            shift = torch.zeros_like(gate)
            scale = torch.zeros_like(gate)
            return x, gate, shift, scale, gate

    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states, encoder_hidden_states, image_rotary_emb=None, **kwargs):
            self.calls += 1
            return hidden_states + 1, encoder_hidden_states + 10

    class FakeFeedForward(nn.Module):
        def __init__(self, delta):
            super().__init__()
            self.delta = delta
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x + self.delta

    class FakeFluxDoubleBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = FakeFluxNorm()
            self.norm1_context = FakeFluxNorm()
            self.attn = FakeAttention()
            self.norm2 = nn.Identity()
            self.ff = FakeFeedForward(100)
            self.norm2_context = nn.Identity()
            self.ff_context = FakeFeedForward(1000)

        def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
            )
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
            ff_output = self.ff(self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None])
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
            encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
            context_ff_output = self.ff_context(
                self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            )
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            return encoder_hidden_states, hidden_states

    class FakeFluxModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = FakeFluxDoubleBlock()

        def backbone_blocks(self):
            from chitu_diffusion.models.backbone import BackboneBlockInfo

            return [BackboneBlockInfo(index=0, name="block", module=self.block)]

    monkeypatch.setattr("chitu_diffusion.flexcache.strategy.taylorseer.get_cp_group", lambda: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(DiffusionBackend, "generator", Generator())
    monkeypatch.setattr(DiffusionBackend, "cfg_type", CFGType.POS)
    monkeypatch.setattr(DiffusionBackend, "flexcache", FlexCacheManager(max_cache_memory=20))

    module = FakeFluxModule()
    strategy = TaylorSeerStrategy(
        task=Task(),
        fresh_threshold=2,
        max_order=1,
        first_enhance=1,
        warmup_steps=0,
        cooldown_steps=0,
    )
    DiffusionBackend.flexcache.set_strategy(strategy)
    strategy.wrap_module_with_strategy(module)

    encoder_out, hidden_out = module.block(
        hidden_states=torch.zeros(1, 1, 1),
        encoder_hidden_states=torch.zeros(1, 1, 1),
        temb=torch.zeros(1, 1),
    )
    assert hidden_out.item() == 102
    assert encoder_out.item() == 1020
    assert module.block.attn.calls == 1
    assert module.block.ff.calls == 1
    assert module.block.ff_context.calls == 1
    assert set(DiffusionBackend.flexcache.cache) == {
        ("taylorseer", "pos_cp0", 0, "img_attn"),
        ("taylorseer", "pos_cp0", 0, "img_mlp"),
        ("taylorseer", "pos_cp0", 0, "txt_attn"),
        ("taylorseer", "pos_cp0", 0, "txt_mlp"),
    }

    Buffer.current_step = 1
    encoder_out, hidden_out = module.block(
        hidden_states=torch.zeros(1, 1, 1),
        encoder_hidden_states=torch.zeros(1, 1, 1),
        temb=torch.zeros(1, 1),
    )
    assert hidden_out.item() == 102
    assert encoder_out.item() == 1020
    assert module.block.attn.calls == 1
    assert module.block.ff.calls == 1
    assert module.block.ff_context.calls == 1
