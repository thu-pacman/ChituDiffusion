from __future__ import annotations

import argparse
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from chitu_diffusion.commands.cache_args import (
    add_cache_arguments,
    cache_config_from_args,
)
from chitu_diffusion.flexcache import (
    CacheSession,
    FlexCacheModelSpec,
    active_cache_session,
)
from chitu_diffusion.flexcache.config import (
    CacheCommonConfig,
    CacheConfig,
    MagCacheConfig,
    MeanCacheConfig,
    PABConfig,
    TaylorSeerConfig,
    TeaCacheConfig,
)
from chitu_diffusion.flexcache.strategies import create_cache_strategy


class _Leaf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states + 1


class _Block(nn.Module):
    """Mimics a gated block: the leaf output is scaled before the residual."""

    def __init__(self, attention_name: str, feedforward_name: str) -> None:
        super().__init__()
        setattr(self, attention_name, _Leaf())
        setattr(self, feedforward_name, _Leaf())
        self.attention_name = attention_name
        self.feedforward_name = feedforward_name
        self.gate = 1.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention = getattr(self, self.attention_name)(hidden_states)
        hidden_states = hidden_states + self.gate * attention
        feedforward = getattr(self, self.feedforward_name)(hidden_states)
        return hidden_states + self.gate * feedforward


class _ToyTransformer(nn.Module):
    def __init__(
        self,
        group_name: str,
        attention_name: str = "attn",
        feedforward_name: str = "ff",
    ) -> None:
        super().__init__()
        setattr(
            self, group_name, nn.ModuleList([_Block(attention_name, feedforward_name)])
        )
        self.group_name = group_name
        self.calls = 0
        # Lets the zimage modulation probe resolve to the raw timestep.
        self.t_embedder = nn.Identity()
        self.t_scale = 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        for block in getattr(self, self.group_name):
            hidden_states = block(hidden_states)
        return hidden_states + timestep


class _RecursiveZImageTransformer(_ToyTransformer):
    def __init__(self) -> None:
        super().__init__("layers", "attention", "feed_forward")
        self.epe = SimpleNamespace(
            parallel=SimpleNamespace(active=SimpleNamespace(width=2))
        )

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        del cap_feats
        if len(x) > 1:
            return [
                self.forward([sample], t[index : index + 1], [sample])[0]
                for index, sample in enumerate(x)
            ]
        hidden_states = x[0]
        for block in self.layers:
            hidden_states = block(hidden_states)
        return [hidden_states + t.reshape(-1, 1)]


class _FluxNorm(nn.Module):
    def __init__(self, *, single: bool = False) -> None:
        super().__init__()
        self.single = single
        self.calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        self.calls += 1
        gate = emb
        if self.single:
            return hidden_states, gate
        zeros = torch.zeros_like(gate)
        return hidden_states, gate, zeros, zeros, gate


class _FluxAttention(nn.Module):
    def __init__(self, *, joint: bool) -> None:
        super().__init__()
        self.joint = joint
        self.include_ip = False
        self.calls = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        self.calls += 1
        if self.joint:
            assert encoder_hidden_states is not None
            if self.include_ip:
                return hidden_states + 1, encoder_hidden_states + 2, hidden_states
            return hidden_states + 1, encoder_hidden_states + 2
        return hidden_states + 3


class _FluxProjection(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states[..., : self.dim] + hidden_states[..., self.dim :]


class _CountingIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states


class _FluxDoubleBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = _FluxNorm()
        self.norm1_context = _FluxNorm()
        self.norm2 = _CountingIdentity()
        self.norm2_context = _CountingIdentity()
        self.attn = _FluxAttention(joint=True)
        self.ff = _Leaf()
        self.ff_context = _Leaf()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_norm, gate, _, _, mlp_gate = self.norm1(hidden_states, emb=temb)
        encoder_norm, encoder_gate, _, _, encoder_mlp_gate = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        attention, encoder_attention = self.attn(
            hidden_norm, encoder_hidden_states=encoder_norm
        )
        hidden_states = hidden_states + gate.unsqueeze(1) * attention
        hidden_states = hidden_states + mlp_gate.unsqueeze(1) * self.ff(
            self.norm2(hidden_states)
        )
        encoder_hidden_states = (
            encoder_hidden_states
            + encoder_gate.unsqueeze(1) * encoder_attention
            + encoder_mlp_gate.unsqueeze(1)
            * self.ff_context(self.norm2_context(encoder_hidden_states))
        )
        return encoder_hidden_states, hidden_states


class _FluxSingleBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = _FluxNorm(single=True)
        self.attn = _FluxAttention(joint=False)
        self.proj_mlp = _Leaf()
        self.act_mlp = nn.Identity()
        self.proj_out = _FluxProjection(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        merged = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        normalized, gate = self.norm(merged, emb=temb)
        attention = self.attn(normalized)
        mlp = self.act_mlp(self.proj_mlp(normalized))
        projected = self.proj_out(torch.cat([attention, mlp], dim=-1))
        merged = merged + gate.unsqueeze(1) * projected
        return merged[:, :text_seq_len], merged[:, text_seq_len:]


class _FluxToyTransformer(nn.Module):
    def __init__(self, dim: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.transformer_blocks = nn.ModuleList([_FluxDoubleBlock()])
        self.single_transformer_blocks = nn.ModuleList([_FluxSingleBlock(dim)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del pooled_projections
        assert timestep is not None
        temb = torch.ones(
            hidden_states.shape[0],
            self.dim,
            dtype=hidden_states.dtype,
        ) * (1 + timestep)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, temb
            )
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, temb
            )
        return encoder_hidden_states, hidden_states


def _session(config: CacheConfig, spec: FlexCacheModelSpec, steps: int) -> CacheSession:
    return CacheSession(
        config=config,
        model_spec=spec,
        strategy=create_cache_strategy(config),
        total_steps=steps,
    )


@pytest.mark.parametrize(
    ("family", "group", "attention", "feedforward", "kind"),
    [
        ("zimage", "layers", "attention", "feed_forward", "joint"),
        ("flux1", "transformer_blocks", "attn", "ff", "joint"),
        ("qwen_image", "transformer_blocks", "attn", "img_mlp", "joint"),
        ("wan", "blocks", "attn1", "ffn", "self"),
    ],
)
def test_model_specs_expose_stable_block_and_leaf_sites(
    family: str,
    group: str,
    attention: str,
    feedforward: str,
    kind: str,
) -> None:
    model = _ToyTransformer(group, attention, feedforward)
    spec = FlexCacheModelSpec.discover(model, family=family)

    assert spec.blocks[0].site_id == f"{family}.block.{group}.0"
    assert [leaf.site_id for leaf in spec.leaves] == [
        f"{family}.block.{group}.0.{attention}",
        f"{family}.block.{group}.0.{feedforward}",
    ]
    assert [leaf.kind for leaf in spec.attentions] == [kind]
    assert (
        spec.step_probe((torch.zeros(1),), {spec.timestep_arg: torch.ones(1)})
        is not None
    )


def test_meancache_uses_official_model_specific_schedules() -> None:
    qwen = _ToyTransformer("transformer_blocks", "attn", "img_mlp")
    qwen_spec = FlexCacheModelSpec.discover(qwen, family="qwen_image")
    qwen_strategy = create_cache_strategy(
        CacheConfig(
            strategy="meancache",
            params=MeanCacheConfig(fresh_steps=10),
        )
    )
    qwen_strategy.begin(total_steps=50, model_spec=qwen_spec)
    assert qwen_strategy.fresh_indices == frozenset(
        {0, 1, 3, 7, 14, 21, 28, 35, 42, 49}
    )

    zimage = _ToyTransformer("layers", "attention", "feed_forward")
    zimage_spec = FlexCacheModelSpec.discover(zimage, family="zimage")
    zimage_strategy = create_cache_strategy(
        CacheConfig(
            strategy="meancache",
            params=MeanCacheConfig(fresh_steps=13),
        )
    )
    zimage_strategy.begin(total_steps=50, model_spec=zimage_spec)
    assert zimage_strategy.fresh_indices == frozenset(
        {0, 1, 2, 3, 5, 9, 15, 22, 29, 36, 43, 47, 49}
    )

    with pytest.raises(ValueError, match="supported values"):
        create_cache_strategy(
            CacheConfig(
                strategy="meancache",
                params=MeanCacheConfig(fresh_steps=20),
            )
        ).begin(total_steps=50, model_spec=qwen_spec)
    with pytest.raises(ValueError, match="exactly 50"):
        zimage_strategy.begin(total_steps=49, model_spec=zimage_spec)


def test_magcache_official_profiles_reproduce_shared_schedules() -> None:
    flux_spec = FlexCacheModelSpec.discover(_FluxToyTransformer(), family="flux1")
    flux = create_cache_strategy(CacheConfig(strategy="magcache"))
    flux.begin(total_steps=28, model_spec=flux_spec)
    assert {i for i, reuse in enumerate(flux.reuse_schedule) if reuse} == {
        3,
        4,
        6,
        7,
        8,
        10,
        12,
        13,
        14,
        15,
        16,
        18,
        19,
        20,
        21,
        22,
        24,
        25,
        27,
    }

    qwen = _ToyTransformer("transformer_blocks", "attn", "img_mlp")
    qwen_spec = FlexCacheModelSpec.discover(qwen, family="qwen_image")
    qwen_strategy = create_cache_strategy(CacheConfig(strategy="magcache"))
    qwen_strategy.begin(total_steps=50, model_spec=qwen_spec)
    assert {i for i, reuse in enumerate(qwen_strategy.reuse_schedule) if reuse} == {
        16,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        33,
        35,
        36,
        38,
        40,
    }

    wan = _ToyTransformer("blocks", "attn1", "ffn")
    wan.config = SimpleNamespace(
        num_attention_heads=12,
        attention_head_dim=128,
        image_dim=None,
    )
    wan_spec = FlexCacheModelSpec.discover(wan, family="wan")
    wan_strategy = create_cache_strategy(CacheConfig(strategy="magcache"))
    wan_strategy.begin(total_steps=50, model_spec=wan_spec)
    assert {i for i, reuse in enumerate(wan_strategy.reuse_schedule) if reuse} == {
        10,
        11,
        12,
        13,
        15,
        16,
        17,
        18,
        20,
        21,
        22,
        23,
        25,
        26,
        27,
        28,
        30,
        31,
        32,
        33,
        35,
        36,
        37,
        39,
        40,
        42,
        43,
        45,
        47,
    }
    assert wan_strategy.stats.strategy["branch_profiles"] == 2
    assert wan_strategy.stats.strategy["shared_branch_schedule"] is True


def test_magcache_reuses_complete_backbone_residual() -> None:
    model = _ToyTransformer("blocks", "attn1", "ffn")
    spec = FlexCacheModelSpec.discover(model, family="wan")
    config = CacheConfig(
        strategy="magcache",
        params=MagCacheConfig(
            threshold=1.0,
            max_skip_steps=2,
            retention_ratio=0.0,
            ratios=(1.0, 1.0),
            reference_steps=2,
        ),
    )
    block = model.blocks[0]

    with _session(config, spec, 2) as session:
        fresh = model(torch.tensor([[1.0]]), torch.tensor(0.0))
        calls = (block.attn1.calls, block.ffn.calls)
        reused = model(torch.tensor([[5.0]]), torch.tensor(1.0))

    residual = fresh - torch.tensor([[1.0]])
    assert torch.equal(reused, torch.tensor([[5.0]]) + residual + 1.0)
    assert (block.attn1.calls, block.ffn.calls) == calls
    assert session.stats.step_hits == 1
    assert session.stats.block_hits == 1


def test_magcache_rejects_custom_profiles_for_heterogeneous_zimage_backbone() -> None:
    model = _ToyTransformer("layers", "attention", "feed_forward")
    spec = FlexCacheModelSpec.discover(model, family="zimage")
    config = CacheConfig(
        strategy="magcache",
        params=MagCacheConfig(
            threshold=0.1,
            max_skip_steps=2,
            retention_ratio=0.2,
            ratios=(1.0, 1.0),
            reference_steps=2,
        ),
    )

    with pytest.raises(NotImplementedError, match="single compatible"):
        create_cache_strategy(config).begin(total_steps=2, model_spec=spec)


def test_flux_magcache_reuses_atomic_double_single_backbone() -> None:
    model = _FluxToyTransformer()
    spec = FlexCacheModelSpec.discover(model, family="flux1")
    config = CacheConfig(
        strategy="magcache",
        params=MagCacheConfig(
            threshold=1.0,
            max_skip_steps=2,
            retention_ratio=0.0,
            ratios=(1.0, 1.0),
            reference_steps=2,
        ),
    )
    hidden0 = torch.zeros(1, 2, 2)
    encoder0 = torch.ones(1, 1, 2)
    hidden1 = torch.full((1, 2, 2), 4.0)
    encoder1 = torch.full((1, 1, 2), 3.0)
    double = model.transformer_blocks[0]
    single = model.single_transformer_blocks[0]

    with _session(config, spec, 2) as session:
        fresh_encoder, fresh_hidden = model(
            hidden0,
            encoder0,
            timestep=torch.tensor(0.0),
        )
        calls = (
            double.attn.calls,
            double.ff.calls,
            double.ff_context.calls,
            single.attn.calls,
            single.proj_mlp.calls,
            single.proj_out.calls,
        )
        reused_encoder, reused_hidden = model(
            hidden1,
            encoder1,
            timestep=torch.tensor(1.0),
        )

    assert torch.equal(reused_hidden, hidden1 + (fresh_hidden - hidden0))
    assert torch.equal(reused_encoder, encoder1)
    assert (
        double.attn.calls,
        double.ff.calls,
        double.ff_context.calls,
        single.attn.calls,
        single.proj_mlp.calls,
        single.proj_out.calls,
    ) == calls
    assert session.stats.block_hits == 2


class _MeanScheduler:
    def __init__(self) -> None:
        self.sigmas = torch.linspace(1.0, 0.0, 51)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        del return_dict
        index = int(timestep.item())
        interval = self.sigmas[index + 1] - self.sigmas[index]
        return (sample + interval * model_output,)


class _MeanParallelContext:
    world_size = 1

    @contextmanager
    def activate(self, lane_ranks: tuple[int, ...]):
        assert lane_ranks == (0,)
        yield


def test_meancache_matches_official_sigma_jvp_trajectory() -> None:
    model = _ToyTransformer("transformer_blocks", "attn", "img_mlp")
    spec = FlexCacheModelSpec.discover(model, family="qwen_image")
    strategy = create_cache_strategy(
        CacheConfig(
            strategy="meancache",
            params=MeanCacheConfig(fresh_steps=10),
        )
    )
    strategy.begin(total_steps=50, model_spec=spec)
    state = SimpleNamespace(
        latents=torch.tensor([[0.25, -0.5]]),
        scheduler=_MeanScheduler(),
        timesteps=torch.arange(50),
        step_index=0,
    )
    pipeline = SimpleNamespace(parallel_context=_MeanParallelContext())

    ref_latents = state.latents.clone()
    ref_pre: list[torch.Tensor] = []
    ref_post: list[torch.Tensor] = []
    ref_sigma_pre: list[torch.Tensor] = []
    ref_sigma_post: list[torch.Tensor] = []
    ref_velocity: list[torch.Tensor] = []

    def full_step() -> SimpleNamespace:
        index = state.step_index
        velocity = state.latents.square() * 0.1 + index / 50
        state.latents = state.scheduler.step(
            velocity,
            state.timesteps[index],
            state.latents,
            return_dict=False,
        )[0]
        state.step_index += 1
        return state

    for index in range(50):
        strategy.denoise_step(
            pipeline,
            state,
            lane_ranks=(0,),
            call_original=full_step,
        )

        ref_latent_pre = ref_latents.clone()
        if index in strategy.fresh_indices:
            velocity = ref_latents.square() * 0.1 + index / 50
        else:
            order = strategy.edge_orders[index - 1]
            actual = min(order, len(ref_velocity))
            start = len(ref_velocity) - actual
            denominator = ref_sigma_post[-1] - ref_sigma_pre[start]
            average = (ref_post[-1] - ref_pre[start]) / denominator
            jvp = (ref_velocity[start] - average) / denominator
            interval = state.scheduler.sigmas[index + 1] - state.scheduler.sigmas[index]
            velocity = ref_velocity[-1] - jvp * interval
        ref_latents = state.scheduler.step(
            velocity,
            state.timesteps[index],
            ref_latents,
            return_dict=False,
        )[0]
        ref_pre.append(ref_latent_pre)
        ref_post.append(ref_latents.clone())
        ref_sigma_pre.append(state.scheduler.sigmas[index])
        ref_sigma_post.append(state.scheduler.sigmas[index + 1])
        ref_velocity.append(velocity)

        assert torch.allclose(state.latents, ref_latents)

    assert strategy.stats.step_misses == 10
    assert strategy.stats.step_hits == 40


def test_cache_config_round_trips_discriminated_params() -> None:
    config = CacheConfig.from_mapping(
        {
            "strategy": "teacache",
            "common": {"warmup_steps": 2, "cooldown_steps": 1},
            "params": {"threshold": 0.3},
        }
    )

    assert isinstance(config.params, TeaCacheConfig)
    assert CacheConfig.from_mapping(config.to_dict()) == config
    mean = CacheConfig.from_mapping(
        {"strategy": "meancache", "params": {"fresh_steps": 17}}
    )
    assert mean == CacheConfig(
        strategy="meancache",
        params=MeanCacheConfig(fresh_steps=17),
    )
    with pytest.raises(ValueError, match="exactly 50"):
        mean.validate_steps(20)
    with pytest.raises(ValueError, match="warmup/cooldown"):
        CacheConfig(
            strategy="meancache",
            common=CacheCommonConfig(warmup_steps=1),
        )
    with pytest.raises(ValueError, match="requires MeanCacheConfig"):
        CacheConfig(strategy="meancache", params=TeaCacheConfig())
    with pytest.raises(ValueError, match="must be a mapping"):
        CacheConfig.from_mapping([])
    with pytest.raises(ValueError, match="finite numbers"):
        CacheConfig.from_mapping(
            {"strategy": "magcache", "params": {"ratios": [1.0, None]}}
        )
    with pytest.raises(ValueError, match="finite numbers"):
        TeaCacheConfig(coefficients=(1.0, float("nan")))


def test_cache_cli_constructs_only_the_selected_strategy() -> None:
    parser = argparse.ArgumentParser()
    add_cache_arguments(parser)
    args = parser.parse_args(["--cache-strategy", "none", "--pab-self-interval", "0"])

    assert cache_config_from_args(args) == CacheConfig()


def test_teacache_refuses_uncalibrated_checkpoints() -> None:
    model = _ToyTransformer("layers", "attention", "feed_forward")
    spec = FlexCacheModelSpec.discover(model, family="zimage")
    config = CacheConfig(strategy="teacache")

    with pytest.raises(NotImplementedError, match="no calibrated polynomial"):
        with _session(config, spec, 4):
            model(torch.zeros(1, 2), torch.tensor(1.0))


def test_teacache_runs_uncalibrated_with_explicit_coefficients() -> None:
    model = _ToyTransformer("layers", "attention", "feed_forward")
    spec = FlexCacheModelSpec.discover(model, family="zimage")
    config = CacheConfig(
        strategy="teacache",
        params=TeaCacheConfig(threshold=1e9, coefficients=(1.0, 0.0)),
    )

    session = _session(config, spec, 4)
    with session:
        for step in range(4):
            model(torch.zeros(1, 2), torch.tensor(1.0 - 0.1 * step))

    # First and last steps are always recomputed, the middle two are reused.
    assert session.stats.step_hits == 2
    assert session.stats.step_misses == 2


def test_serial_cfg_branches_share_a_denoise_step_and_isolate_state() -> None:
    model = _ToyTransformer("blocks", "attn1", "ffn")
    spec = FlexCacheModelSpec.discover(model, family="wan")
    config = CacheConfig(
        strategy="taylorseer",
        params=TaylorSeerConfig(fresh_threshold=2, first_enhance=1),
    )

    session = _session(config, spec, 4)
    with session:
        for step in range(4):
            timestep = torch.tensor(1.0 - 0.1 * step)
            model(torch.zeros(1, 2), timestep=timestep)
            model(torch.ones(1, 2), timestep=timestep)

    assert session.steps_seen == 4
    assert session.branches_seen == 2
    # Both branches follow the same schedule, so hits scale with branch count.
    assert session.stats.step_hits == 4
    assert session.stats.step_misses == 4


def test_recursive_zimage_cp_cfg_branches_are_isolated() -> None:
    model = _RecursiveZImageTransformer()
    spec = FlexCacheModelSpec.discover(model, family="zimage")
    config = CacheConfig(
        strategy="pab",
        params=PABConfig(self_interval=2, cross_interval=2, joint_interval=2),
    )

    session = _session(config, spec, 2)
    with session:
        model(
            [torch.zeros(1, 2), torch.full((1, 2), 10.0)],
            torch.tensor([1.0, 1.0]),
            [torch.zeros(1, 2), torch.zeros(1, 2)],
        )
        model(
            [torch.ones(1, 2), torch.full((1, 2), 20.0)],
            torch.tensor([0.0, 0.0]),
            [torch.zeros(1, 2), torch.zeros(1, 2)],
        )

    assert session.steps_seen == 2
    assert session.branches_seen == 2
    assert session.stats.leaf_hits == 2
    assert session.stats.leaf_misses == 2


def test_taylorseer_extrapolates_pre_gate_leaf_outputs() -> None:
    model = _ToyTransformer("blocks", "attn1", "ffn")
    spec = FlexCacheModelSpec.discover(model, family="wan")
    config = CacheConfig(
        strategy="taylorseer",
        params=TaylorSeerConfig(fresh_threshold=2, first_enhance=1),
    )

    session = _session(config, spec, 4)
    with session:
        for step in range(4):
            model(torch.zeros(1, 2), timestep=torch.tensor(float(step)))

    # Steps 0 and 2 recompute; 1 and 3 reuse both leaves of the single block.
    assert model.calls == 4
    assert model.blocks[0].attn1.calls == 2
    assert model.blocks[0].ffn.calls == 2
    assert session.stats.leaf_hits == 4
    assert session.stats.leaf_misses == 4


def test_flux_taylorseer_reuse_bypasses_whole_blocks() -> None:
    model = _FluxToyTransformer()
    spec = FlexCacheModelSpec.discover(model, family="flux1")
    config = CacheConfig(
        strategy="taylorseer",
        params=TaylorSeerConfig(fresh_threshold=3, first_enhance=1),
    )
    hidden = torch.zeros(1, 2, 2)
    encoder = torch.ones(1, 1, 2)

    session = _session(config, spec, 2)
    with session:
        model(hidden, encoder, timestep=torch.tensor(0.0))
        double = model.transformer_blocks[0]
        single = model.single_transformer_blocks[0]
        fresh_calls = (
            double.attn.calls,
            double.ff.calls,
            double.ff_context.calls,
            single.attn.calls,
            single.proj_mlp.calls,
            single.proj_out.calls,
        )
        model(hidden, encoder, timestep=torch.tensor(1.0))

    assert fresh_calls == (1, 1, 1, 1, 1, 1)
    assert (
        double.attn.calls,
        double.ff.calls,
        double.ff_context.calls,
        single.attn.calls,
        single.proj_mlp.calls,
        single.proj_out.calls,
    ) == fresh_calls
    assert double.norm1.calls == 2
    assert double.norm1_context.calls == 2
    assert double.norm2.calls == 1
    assert double.norm2_context.calls == 1
    assert single.norm.calls == 2
    assert session.stats.block_hits == 2
    assert session.stats.leaf_hits == 4


def test_flux_taylorseer_rejects_ip_attention_outputs() -> None:
    model = _FluxToyTransformer()
    model.transformer_blocks[0].attn.include_ip = True
    spec = FlexCacheModelSpec.discover(model, family="flux1")
    config = CacheConfig(
        strategy="taylorseer",
        params=TaylorSeerConfig(fresh_threshold=3, first_enhance=1),
    )

    with pytest.raises(NotImplementedError, match="IP-Adapter"):
        with _session(config, spec, 2):
            model(
                torch.zeros(1, 2, 2),
                torch.ones(1, 1, 2),
                timestep=torch.tensor(0.0),
            )


def test_taylorseer_second_order_differs_from_first_order() -> None:
    def run(max_order: int) -> torch.Tensor:
        torch.manual_seed(0)
        model = _ToyTransformer("blocks", "attn1", "ffn")
        spec = FlexCacheModelSpec.discover(model, family="wan")
        config = CacheConfig(
            strategy="taylorseer",
            params=TaylorSeerConfig(
                fresh_threshold=3,
                max_order=max_order,
                first_enhance=3,
            ),
        )
        outputs = []
        with _session(config, spec, 9):
            for step in range(9):
                hidden = torch.full((1, 2), float(step) ** 2)
                outputs.append(model(hidden, timestep=torch.tensor(float(step))))
        return torch.stack(outputs)

    first_order = run(1)
    second_order = run(2)
    # The reference does not bootstrap a second derivative before the first
    # three enhanced anchors have completed.
    assert torch.allclose(second_order[3], first_order[3])
    assert not torch.allclose(second_order, first_order)


def test_pab_reuses_attention_leaves_only() -> None:
    model = _ToyTransformer("blocks", "attn1", "ffn")
    spec = FlexCacheModelSpec.discover(model, family="wan")
    config = CacheConfig(
        strategy="pab",
        params=PABConfig(self_interval=2, cross_interval=3, joint_interval=2),
    )

    session = _session(config, spec, 4)
    with session:
        for step in range(4):
            model(torch.zeros(1, 2), torch.tensor(float(step)))

    assert model.blocks[0].attn1.calls == 2
    assert model.blocks[0].ffn.calls == 4
    assert session.stats.leaf_hits == 2
    assert session.stats.leaf_misses == 2


def test_cache_sessions_do_not_share_request_state() -> None:
    model = _ToyTransformer("layers", "attention", "feed_forward")
    spec = FlexCacheModelSpec.discover(model, family="zimage")
    config = CacheConfig(
        strategy="teacache",
        params=TeaCacheConfig(threshold=1e9, coefficients=(1.0, 0.0)),
    )

    hits = []
    for _ in range(2):
        session = _session(config, spec, 3)
        with session:
            for step in range(3):
                model(torch.zeros(1, 2), torch.tensor(1.0 - 0.1 * step))
        hits.append(session.stats.step_hits)

    assert hits == [1, 1]
    assert active_cache_session() is None


def test_request_level_cache_patches_only_the_pipeline_instance() -> None:
    class _Pipeline:
        parallel_context = SimpleNamespace(world_size=1)

        def denoise_step(
            self,
            state: object,
            *,
            lane_ranks: tuple[int, ...],
        ) -> object:
            del lane_ranks
            return state

    model = _ToyTransformer("transformer_blocks", "attn", "img_mlp")
    spec = FlexCacheModelSpec.discover(model, family="qwen_image")
    config = CacheConfig(
        strategy="meancache",
        params=MeanCacheConfig(fresh_steps=10),
    )
    pipeline = _Pipeline()
    class_forward = _Pipeline.denoise_step
    session = CacheSession(
        config=config,
        model_spec=spec,
        strategy=create_cache_strategy(config),
        total_steps=50,
        pipeline=pipeline,
    )

    with session:
        assert _Pipeline.denoise_step is class_forward
        assert "denoise_step" in pipeline.__dict__

    assert _Pipeline.denoise_step is class_forward
    assert active_cache_session() is None


def test_serve_rejects_generate_only_cache_strategies() -> None:
    with pytest.raises(NotImplementedError, match="generate-only"):
        CacheConfig(strategy="pab").require_serve_available()
    CacheConfig().require_serve_available()
