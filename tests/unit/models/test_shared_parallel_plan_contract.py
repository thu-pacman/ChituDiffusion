"""Every serve factory reads the common plans instead of private arguments."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from chitu_diffusion.epe.contracts import (
    ContextParallelPlan,
    ExecutorBuildContext,
    ModelParallelPlan,
    StageWorldSpec,
    VaeParallelPlan,
)
from chitu_diffusion.models.flux1 import executor as flux1_executor
from chitu_diffusion.models.hunyuan_image3 import executor as hunyuan_image3_executor
from chitu_diffusion.models.minimax_h3 import executor as minimax_h3_executor
from chitu_diffusion.models.qwen_image import executor as qwen_image_executor
from chitu_diffusion.models.wan import executor as wan_executor
from chitu_diffusion.models.zimage import executor as zimage_executor
from chitu_diffusion.parallel.vae import create_vae_parallel_placement

FACTORIES = {
    "zimage": (zimage_executor, "ZImageExecutorFactory", "agkv"),
    "flux1": (flux1_executor, "Flux1ExecutorFactory", "agkv"),
    "qwen-image": (qwen_image_executor, "QwenImageExecutorFactory", "agkv"),
    "wan": (wan_executor, "WanExecutorFactory", "agkv"),
    "minimax-h3": (minimax_h3_executor, "MiniMaxH3ExecutorFactory", "ulysses"),
    "hunyuan-image3": (
        hunyuan_image3_executor,
        "HunyuanImage3ExecutorFactory",
        "agkv",
    ),
}


class _ContextCaptured(Exception):
    def __init__(self, context: ExecutorBuildContext, overrides: dict[str, object]):
        super().__init__("captured")
        self.context = context
        self.overrides = overrides


def _context(attention_mode: str) -> ExecutorBuildContext:
    return ExecutorBuildContext(
        world=StageWorldSpec(
            stage_name="stage",
            rank=0,
            world_size=4,
            local_device="cpu",
            physical_device_ids=(0, 1, 2, 3),
        ),
        pool=SimpleNamespace(policy="static_cp", allowed_lane_widths=(1, 2, 4)),
        cp=ContextParallelPlan(
            world_size=4,
            attention_mode=attention_mode,
            ulysses_degree=4 if attention_mode == "ulysses" else None,
            agkv_transport="fast",
        ),
        vae=VaeParallelPlan(degree=None, halo=6),
        model=ModelParallelPlan(),
    )


@pytest.mark.parametrize("factory_name", sorted(FACTORIES))
def test_every_factory_builds_its_lane_from_the_common_cp_plan(
    factory_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, attribute, attention_mode = FACTORIES[factory_name]
    _capture_context(module, monkeypatch)
    factory = getattr(module, attribute)("/models/unused")
    context = _context(attention_mode)

    with pytest.raises(_ContextCaptured) as captured:
        factory.build(context)

    assert captured.value.context is context
    if factory_name == "hunyuan-image3":
        # The DiT attention mode is separate from the lane layout, which this
        # model always builds as a Ulysses group of its context-parallel degree.
        # Transports stay on the common CP plan and must not be overwritten.
        assert captured.value.overrides == {
            "attention_mode": "ulysses",
            "ulysses_degree": 4,
        }
    else:
        assert captured.value.overrides == {}


def _capture_context(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def capture(context: ExecutorBuildContext, **overrides: object) -> None:
        raise _ContextCaptured(context, overrides)

    monkeypatch.setattr(module, "build_stage_parallel_context", capture)


def test_the_vae_plan_reaches_decode_without_passing_through_a_factory() -> None:
    plan = _context("agkv").vae
    placement = create_vae_parallel_placement(plan.degree, halo=plan.halo)

    assert placement.halo == 6
    assert placement.degree is None
    assert not placement.is_stage_scoped
