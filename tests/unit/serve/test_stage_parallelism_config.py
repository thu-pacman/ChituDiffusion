from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chitu_diffusion.serve.config import (
    StageServiceConfig,
    load_stage_service_config,
)

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


def _stage(**parallelism: Any) -> dict[str, Any]:
    return {
        "name": "stage",
        "factory": "zimage",
        "gpu": [0, 1, 2, 3],
        "parallelism": parallelism,
        "factory_args": {"model_path": "/models/Z-Image"},
    }


def _config(**parallelism: Any) -> StageServiceConfig:
    return StageServiceConfig.from_mapping(_stage(**parallelism))


def test_the_shipped_examples_describe_their_documented_geometry() -> None:
    zimage = load_stage_service_config(EXAMPLES / "stage-zimage.yaml")
    h3 = load_stage_service_config(EXAMPLES / "stage-minimax-h3.yaml")
    hyi3 = load_stage_service_config(EXAMPLES / "stage-hunyuan-image3.yaml")

    assert zimage.parallelism.cp.world_size == 4
    assert zimage.parallelism.model.tensor_parallel_degree == 1
    assert zimage.parallelism.vae.degree is None

    assert h3.parallelism.cp.world_size == 2
    assert h3.parallelism.cp.attention_mode == "ulysses"
    assert h3.parallelism.model.tensor_parallel_degree == 4
    assert h3.parallelism.vae.degree is None

    assert hyi3.parallelism.cp.world_size == 4
    assert hyi3.parallelism.model.tensor_parallel_degree == 2
    assert hyi3.parallelism.model.cfg_parallel_degree == 2
    assert hyi3.parallelism.model.expert_parallel_degree == 2
    assert hyi3.parallelism.vae.degree == 8

    for config in (zimage, h3, hyi3):
        assert config.parallelism.gpu_count == len(config.gpu)


def test_an_empty_parallelism_section_fills_the_whole_stage_with_cp() -> None:
    config = _config()

    assert config.parallelism.cp.world_size == 4
    assert config.parallelism.cp.attention_mode == "agkv"
    assert config.parallelism.cp.ulysses_degree is None
    assert config.parallelism.cp.ulysses_transport is None
    assert config.parallelism.cp.agkv_transport is None
    assert config.parallelism.vae.degree is None
    assert config.parallelism.vae.halo == 8
    assert config.parallelism.scheduler.policy == "elastic"
    assert config.parallelism.scheduler.allowed_lane_widths == (1, 2, 4)
    # Z-Image splits guidance by default, and the CP world absorbs the rest.
    assert config.parallelism.model.cfg_parallel_degree == 2
    assert config.parallelism.model.tensor_parallel_degree == 1


def test_the_cp_world_and_tensor_parallel_degree_must_cover_every_gpu() -> None:
    with pytest.raises(ValueError, match="must equal the number of stage GPUs"):
        StageServiceConfig.from_mapping(
            {
                "name": "h3",
                "factory": "minimax-h3",
                "gpu": list(range(8)),
                "parallelism": {
                    "cp": {"world_size": 4, "attention_mode": "ulysses"},
                    "model": {"tensor_parallel_degree": 4},
                    "scheduler": {"allowed_lane_widths": [1, 2, 4]},
                },
                "factory_args": {"model_path": "/models/MiniMax-H3"},
            }
        )


def test_a_tensor_parallel_degree_that_cannot_tile_the_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="must divide the number of stage GPUs"):
        StageServiceConfig.from_mapping(
            {
                "name": "h3",
                "factory": "minimax-h3",
                "gpu": list(range(8)),
                "parallelism": {"model": {"tensor_parallel_degree": 3}},
                "factory_args": {"model_path": "/models/MiniMax-H3"},
            }
        )


@pytest.mark.parametrize(
    "transports",
    [
        {"ulysses_transport": "auto", "agkv_transport": "auto"},
        {"ulysses_transport": "nccl", "agkv_transport": "fast"},
        {"ulysses_transport": "fast", "agkv_transport": "nccl"},
    ],
)
def test_transport_selections_normalize_auto_to_the_runtime_default(
    transports: dict[str, str],
) -> None:
    plan = _config(cp={"world_size": 4, **transports}).parallelism.cp

    for key, value in transports.items():
        expected = None if value == "auto" else value
        assert getattr(plan, key) == expected


def test_an_unknown_transport_names_the_supported_values() -> None:
    with pytest.raises(ValueError, match="parallelism.cp.agkv_transport must be"):
        _config(cp={"world_size": 4, "agkv_transport": "nvshmem"})


@pytest.mark.parametrize("degree", [1, 2, 4])
def test_a_static_cp_stage_may_pin_decode_to_a_fixed_group(degree: int) -> None:
    config = _config(
        vae={"degree": degree},
        scheduler={"policy": "static_cp", "allowed_lane_widths": [1, 2, 4]},
    )

    assert config.parallelism.vae.degree == degree
    assert config.parallelism.vae.is_stage_scoped is (degree > 1)


def test_a_fixed_decode_group_is_refused_while_lanes_decode_concurrently() -> None:
    with pytest.raises(ValueError, match="requires .*policy=static_cp"):
        _config(vae={"degree": 4})


def test_a_decode_group_wider_than_the_stage_is_refused() -> None:
    with pytest.raises(ValueError, match=r"vae.degree must be in \[1, 4\]"):
        _config(
            vae={"degree": 8},
            scheduler={"policy": "static_cp", "allowed_lane_widths": [1, 2, 4]},
        )


def test_disabling_the_vae_group_normalizes_to_a_single_decoding_rank() -> None:
    assert _config(vae={"enabled": False}).parallelism.vae.degree == 1

    with pytest.raises(ValueError, match="must be unset or 1 when vae is disabled"):
        _config(vae={"enabled": False, "degree": 4})


def test_a_negative_halo_is_refused() -> None:
    with pytest.raises(ValueError, match="halo must be non-negative"):
        _config(vae={"halo": -1})


@pytest.mark.parametrize(
    ("factory", "model", "message"),
    [
        ("flux1", {"cfg_parallel_degree": 2}, "supported axes: none"),
        ("zimage", {"tensor_parallel_degree": 2}, "cfg_parallel_degree"),
        ("minimax-h3", {"expert_parallel_degree": 2}, "tensor_parallel_degree"),
    ],
)
def test_a_model_only_accepts_the_axes_it_implements(
    factory: str, model: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message) as failure:
        StageServiceConfig.from_mapping(
            {
                "name": "stage",
                "factory": factory,
                "gpu": [0, 1, 2, 3],
                "parallelism": {"model": model},
                "factory_args": {"model_path": "/models/whatever"},
            }
        )

    assert "does not implement parallelism.model keys" in str(failure.value)


def test_an_unknown_parallelism_section_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown parallelism keys: ep"):
        _config(ep={"degree": 2})


def test_an_unknown_key_inside_a_section_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown parallelism.vae keys: tile"):
        _config(vae={"tile": 512})


@pytest.mark.parametrize(
    ("section", "field", "value", "replacement"),
    [
        ("parallelism", "sp", 4, "parallelism.cp.world_size"),
        ("parallelism", "tp", 2, "parallelism.model.tensor_parallel_degree"),
        ("parallelism", "chitu_pool", {}, "parallelism.scheduler"),
        (
            "factory_args",
            "attention_mode",
            "ulysses",
            "parallelism.cp.attention_mode",
        ),
        ("factory_args", "ulysses_degree", 2, "parallelism.cp.ulysses_degree"),
        (
            "factory_args",
            "cfg_parallel",
            True,
            "parallelism.model.cfg_parallel_degree",
        ),
        (
            "factory_args",
            "expert_parallel_degree",
            4,
            "parallelism.model.expert_parallel_degree",
        ),
        ("factory_args", "parallel_vae", True, "parallelism.vae.enabled"),
        ("factory_args", "vae_parallel_degree", 8, "parallelism.vae.degree"),
        ("factory_args", "vae_parallel_halo", 8, "parallelism.vae.halo"),
    ],
)
def test_every_retired_field_reports_where_it_moved(
    section: str, field: str, value: Any, replacement: str
) -> None:
    raw = _stage(cp={"world_size": 4})
    raw.setdefault(section, {})[field] = value

    with pytest.raises(ValueError, match=f"moved to {replacement}"):
        StageServiceConfig.from_mapping(raw)
