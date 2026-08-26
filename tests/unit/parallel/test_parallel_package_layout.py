from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "chitu_diffusion.parallel.cp",
        "chitu_diffusion.parallel.cp.fast",
        "chitu_diffusion.parallel.cp.nccl",
        "chitu_diffusion.parallel.tp",
        "chitu_diffusion.parallel.vae",
    ),
)
def test_parallel_domains_are_importable(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize(
    "module_name",
    (
        "chitu_diffusion.parallel.groups",
        "chitu_diffusion.parallel.fast_cp",
        "chitu_diffusion.parallel.linear",
        "chitu_diffusion.parallel.nccl",
        "chitu_diffusion.parallel.tensor_parallel",
        "chitu_diffusion.parallel.topology",
        "chitu_diffusion.parallel.tp_loader",
    ),
)
def test_removed_parallel_paths_fail_explicitly(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_parallel_root_does_not_reexport_domain_symbols() -> None:
    parallel = importlib.import_module("chitu_diffusion.parallel")
    assert not hasattr(parallel, "EpeParallelContext")
    assert not hasattr(parallel, "RowParallelLinear")
    assert not hasattr(parallel, "parallel_tiled_vae_decode")
