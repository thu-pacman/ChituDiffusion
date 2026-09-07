from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import yaml

from chitu_diffusion.cli import build_parser
from chitu_diffusion.epe.contracts import (
    ContextParallelPlan,
    ExecutorBuildContext,
    ModelParallelPlan,
    StageWorldSpec,
    VaeParallelPlan,
)
from chitu_diffusion.models.hunyuan_image3 import (
    HunyuanImage3Executor,
    HunyuanImage3ExecutorFactory,
)
from chitu_diffusion.serve.config import HotSwitchPoolConfig, StageServiceConfig

EXAMPLE = "examples/stage-hunyuan-image3.yaml"


def _pool(**overrides: object) -> HotSwitchPoolConfig:
    fields: dict[str, object] = {
        "policy": "static_cp",
        "allowed_lane_widths": [1, 2, 4],
        "warmup_resolutions": [1024],
        "warmup_steps": 3,
    }
    fields.update(overrides)
    world_size = max(fields["allowed_lane_widths"])  # type: ignore[arg-type]
    return HotSwitchPoolConfig.from_mapping(fields, world_size=world_size)


def _context(
    pool: HotSwitchPoolConfig,
    *,
    tensor_parallel_degree: int = 2,
    cfg_parallel_degree: int = 2,
    expert_parallel_degree: int | None = None,
    vae_degree: int | None = 8,
) -> ExecutorBuildContext:
    world = StageWorldSpec(
        stage_name="hunyuan_image3",
        rank=0,
        world_size=8,
        local_device="cpu",
        physical_device_ids=tuple(range(8)),
        owns_process_group=False,
    )
    return ExecutorBuildContext(
        world=world,
        pool=pool,
        cp=ContextParallelPlan(
            world_size=8 // tensor_parallel_degree, attention_mode="agkv"
        ),
        vae=VaeParallelPlan(degree=vae_degree),
        model=ModelParallelPlan(
            tensor_parallel_degree=tensor_parallel_degree,
            cfg_parallel_degree=cfg_parallel_degree,
            expert_parallel_degree=expert_parallel_degree,
        ),
    )


def test_the_stage_example_describes_the_supported_geometry() -> None:
    config = StageServiceConfig.from_mapping(yaml.safe_load(open(EXAMPLE)))
    parallelism = config.parallelism

    assert config.factory == "hunyuan-image3"
    assert parallelism.model.tensor_parallel_degree == 2
    assert parallelism.model.cfg_parallel_degree == 2
    assert parallelism.model.expert_parallel_degree == 2
    assert parallelism.cp.world_size == 4
    assert parallelism.cp.attention_mode == "agkv"
    assert parallelism.vae.degree == 8
    assert parallelism.scheduler.policy == "static_cp"
    assert max(parallelism.scheduler.allowed_lane_widths) == 4
    assert len(config.gpu) == 8


def test_the_cli_exposes_the_distributed_generate_command() -> None:
    parser = build_parser()
    namespace, remaining = parser.parse_known_args(
        ["generate", "--model", "hunyuan-image3", "--prompt", "a red cube"]
    )

    assert namespace.model == "hunyuan-image3"
    assert remaining == ["--prompt", "a red cube"]


def _cli_plan(argv: list[str], world_size: int):
    import sys
    from unittest.mock import patch

    from chitu_diffusion.commands.generate.hunyuan_image3 import build_plan, parse_args

    with patch.object(sys, "argv", ["hunyuan-image3", "--model-path", "/x", *argv]):
        return build_plan(parse_args(), world_size)


def test_the_distributed_cli_defaults_to_the_validated_eight_gpu_topology() -> None:
    plan = _cli_plan([], 8)

    assert plan.describe() == "TP2xCFG2xCP2xEP4"
    assert plan.attention_mode == "agkv"


@pytest.mark.parametrize(
    ("argv", "world_size", "topology"),
    (
        (["--cfg-parallel-degree", "1"], 8, "TP2xCFG1xCP4xEP4"),
        (["--tensor-parallel-degree", "4"], 8, "TP4xCFG2xCP1xEP2"),
        (["--expert-parallel-degree", "1"], 8, "TP2xCFG2xCP2xEP1"),
        (["--tensor-parallel-degree", "1"], 4, "TP1xCFG2xCP2xEP4"),
    ),
)
def test_the_distributed_cli_accepts_other_single_node_topologies(
    argv: list[str], world_size: int, topology: str
) -> None:
    assert _cli_plan(argv, world_size).describe() == topology


def test_the_distributed_cli_rejects_degrees_the_launch_cannot_fill() -> None:
    with pytest.raises(SystemExit, match="unsupported Hunyuan Image 3 topology"):
        _cli_plan(["--context-parallel-degree", "1"], 8)


def test_the_factory_requires_a_static_context_parallel_lane() -> None:
    factory = HunyuanImage3ExecutorFactory(model_path="/nonexistent")

    with pytest.raises(ValueError, match="static_cp"):
        factory.build(_context(_pool(policy="elastic")))


def test_the_factory_requires_the_widest_lane_to_be_the_tp_plane_width() -> (
    None
):
    factory = HunyuanImage3ExecutorFactory(model_path="/nonexistent")
    pool = HotSwitchPoolConfig.from_mapping(
        {
            "policy": "static_cp",
            "allowed_lane_widths": [1, 2],
            "warmup_resolutions": [1024],
            "warmup_steps": 3,
        },
        world_size=2,
    )

    with pytest.raises(ValueError, match="TP-plane width 4"):
        factory.build(_context(pool))


@pytest.mark.parametrize(
    ("model_fields", "topology"),
    (
        ({}, "TP2xCFG2xCP2xEP4"),
        ({"cfg_parallel_degree": 1}, "TP2xCFG1xCP4xEP4"),
        ({"expert_parallel_degree": 2}, "TP2xCFG2xCP2xEP2"),
        ({"tensor_parallel_degree": 4}, "TP4xCFG2xCP1xEP2"),
        (
            {"tensor_parallel_degree": 1, "cfg_parallel_degree": 1},
            "TP1xCFG1xCP8xEP8",
        ),
    ),
)
def test_the_factory_derives_the_plan_from_the_stage_world(
    model_fields: dict[str, object], topology: str
) -> None:
    factory = HunyuanImage3ExecutorFactory(model_path="/nonexistent")

    plan = factory.resolve_plan(_context(_pool(), **model_fields))

    assert plan.describe() == topology
    assert plan.world_size == 8


def test_the_factory_requires_every_configured_degree_to_have_a_lane() -> None:
    # EP2 collects over rank pairs, so a pool that only groups whole planes
    # would leave that dispatch without a process group.
    factory = HunyuanImage3ExecutorFactory(model_path="/nonexistent")
    pool = HotSwitchPoolConfig.from_mapping(
        {
            "policy": "static_cp",
            "allowed_lane_widths": [1, 4],
            "warmup_resolutions": [1024],
            "warmup_steps": 3,
        },
        world_size=4,
    )

    with pytest.raises(ValueError, match=r"allowed_lane_widths to include \[2\]"):
        factory.build(
            _context(pool, cfg_parallel_degree=1, expert_parallel_degree=2)
        )


def test_finalize_activates_the_completing_lane_for_lane_scoped_decode() -> None:
    class ParallelContext:
        active: tuple[int, ...] = (7,)

        @contextmanager
        def activate(self, ranks: tuple[int, ...]):
            previous = self.active
            self.active = ranks
            try:
                yield self
            finally:
                self.active = previous

    class Pipeline:
        def __init__(self) -> None:
            self.parallel_context = ParallelContext()
            self.decode_lane: tuple[int, ...] | None = None

        def decode_request(self, state: object, *, timings: dict[str, object]):
            del state, timings
            self.decode_lane = self.parallel_context.active
            return "decoded"

    pipeline = Pipeline()
    executor = object.__new__(HunyuanImage3Executor)
    executor.pipeline = pipeline

    result = executor.finalize_gpu(
        SimpleNamespace(output_type="pil"),
        lane_ranks=(0, 1),
        timings={},
    )

    assert result == "decoded"
    assert pipeline.decode_lane == (0, 1)
    assert pipeline.parallel_context.active == (7,)
